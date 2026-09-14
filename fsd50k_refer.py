#!/usr/bin/env python
# =============================================================================
# FSD50K fine-tuning: AudioMAE (tiny/small/base/large)
# Multi-label sound event classification (200 classes), mAP metric
# =============================================================================

import sys
import os
import gc
import types
import collections
import collections.abc
import math
import random
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score

import copy
import timm
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple
from timm.utils import ModelEmaV2

# =========================================================
# Environment patches
# =========================================================
print("🔧 Applying environment patches...")
try:
    import torch._six
except ImportError:
    pass

_six = types.ModuleType('torch._six')
_six.container_abcs = collections.abc
_six.string_classes = str
sys.modules['torch._six'] = _six

# =========================================================
# Model Config
# =========================================================
def get_audiomae_config(model_size):
    configs = {
        "tiny":  {"embed_dim": 192,  "depth": 12, "num_heads": 3,  "drop_path_rate": 0.0},
        "small": {"embed_dim": 384,  "depth": 12, "num_heads": 6,  "drop_path_rate": 0.1},
        "base":  {"embed_dim": 768,  "depth": 12, "num_heads": 12, "drop_path_rate": 0.2},
        "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "drop_path_rate": 0.3},
    }
    if model_size.lower() not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    return configs[model_size.lower()]

# =========================================================
# 1. Augmentation utilities, Mixup/CutMix, and ASL loss
# =========================================================
class WaveformAugment:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, waveform):
        if random.random() < 0.5:
            gain = random.uniform(0.7, 1.3)
            waveform = waveform * gain
        if random.random() < 0.5:
            noise_level = random.uniform(0.0005, 0.005)
            noise = torch.randn_like(waveform)
            waveform = waveform + noise * noise_level
        if random.random() < 0.5:
            shift = random.randint(0, waveform.shape[1])
            waveform = torch.roll(waveform, shifts=shift, dims=1)
        if random.random() < 0.5:
            waveform = -waveform
        return waveform

class MixupCutmix:
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=0.5, cutmix_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.cutmix_prob = cutmix_prob

    def __call__(self, x, target):
        if random.random() > (self.mix_prob + self.cutmix_prob):
            return x, target

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        use_cutmix = random.random() < (self.cutmix_prob / (self.mix_prob + self.cutmix_prob))

        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            H, W = x.shape[2], x.shape[3]
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            target_mixed = target * lam + target[index, :] * (1. - lam)
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            x = lam * x + (1 - lam) * x[index, :]
            target_mixed = lam * target + (1 - lam) * target[index, :]

        return x, target_mixed

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()

# =========================================================
# 2. AudioMAE model definition
# =========================================================
class PatchEmbed_new(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.patch_hw = (img_size[0] // stride[0], img_size[1] // stride[1])
        self.num_patches = self.patch_hw[0] * self.patch_hw[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])

    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000 ** omega
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class AudioMAEClassifier(nn.Module):
    def __init__(self, img_size=(1024, 128), patch_size=16, stride=16, in_chans=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 num_classes=200, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 init_method="xavier_uniform"):
        super().__init__()
        self.init_method = init_method
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size,
                                          in_chans=in_chans, embed_dim=embed_dim, stride=stride)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        if self.init_method == "xavier_uniform":
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        elif self.init_method == "xavier_normal":
            torch.nn.init.xavier_normal_(w.view([w.shape[0], -1]))
        elif self.init_method == "trunc_normal_002":
            torch.nn.init.trunc_normal_(w, std=0.02)
        elif self.init_method == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(w.view([w.shape[0], -1]), nonlinearity="relu")
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")
        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.init_method == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            elif self.init_method == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            elif self.init_method == "trunc_normal_002":
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
            elif self.init_method == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown init_method: {self.init_method}")
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained(self, checkpoint_path, logger=None):
        log = logger.info if logger else print
        log(f"📥 Loading pretrained weights from {checkpoint_path}...")
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            state_dict = ckpt.get('model_state') or ckpt.get('model') or ckpt
            new_dict = {k: v for k, v in state_dict.items() if 'decoder' not in k and 'mask_token' not in k}
            msg = self.load_state_dict(new_dict, strict=False)
            log(f"✅ Loaded. {msg}")
        except Exception as e:
            log(f"❌ Load failed: {e}")

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 1:, :].mean(dim=1))


def get_parameter_groups_with_llrd(model, weight_decay, lr, layer_decay):
    """Create decoupled AdamW groups with ViT layer-wise LR decay."""
    max_layer_id = len(model.blocks) + 1
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("patch_embed.") or name in {"cls_token", "pos_embed"}:
            layer_id = 0
        elif name.startswith("blocks."):
            layer_id = int(name.split(".", 2)[1]) + 1
        else:
            layer_id = max_layer_id
        no_decay = parameter.ndim == 1 or name.endswith(".bias") or name.endswith("cls_token")
        key = (layer_id, no_decay)
        if key not in groups:
            scale = layer_decay ** (max_layer_id - layer_id)
            groups[key] = {
                "params": [],
                "lr": lr * scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "lr_scale": scale,
            }
        groups[key]["params"].append(parameter)
    return [groups[key] for key in sorted(groups)]

# =========================================================
# 3. FSD50K Dataset
# =========================================================
class FSD50KDataset(Dataset):
    def __init__(self, root_dir, split='train', target_length=1024, sr=16000,
                 freq_mask=24, time_mask=96):
        self.root_dir = root_dir
        self.split = split
        self.target_sr = sr
        self.target_len = int(sr * 10.24)
        self.target_length = target_length

        # Support both local (FSD50K.ground_truth/FSD50K.dev_audio) and original (labels/clips) layouts
        gt_dir = os.path.join(root_dir, 'FSD50K.ground_truth')
        if not os.path.isdir(gt_dir):
            gt_dir = os.path.join(root_dir, 'labels')
        vocab_path = os.path.join(gt_dir, 'vocabulary.csv')

        vocab_df = pd.read_csv(vocab_path, header=None, dtype=str)
        valid_labels = []
        for idx, row in vocab_df.iterrows():
            vals = [str(x).strip() for x in row.values if pd.notna(x)]
            if any(v.lower() == 'label' for v in vals): continue
            label_candidates = [v for v in vals if not v.startswith('/m/') and not v.isdigit()]
            if label_candidates:
                valid_labels.append(label_candidates[0])

        self.label2id = {l: i for i, l in enumerate(valid_labels)}

        if split in ['train', 'val']:
            csv_path = os.path.join(gt_dir, 'dev.csv')
            # dev.csv has header: fname,labels,mids,split
            df = pd.read_csv(csv_path, dtype=str)
            df['split'] = df['split'].str.strip()
            self.df = df[df['split'] == split]

            dev_audio = os.path.join(root_dir, 'FSD50K.dev_audio')
            if not os.path.isdir(dev_audio):
                dev_audio = os.path.join(root_dir, 'clips', 'dev')
            self.audio_dir = dev_audio
        else:
            csv_path = os.path.join(gt_dir, 'eval.csv')
            try:
                df = pd.read_csv(csv_path, dtype=str)
            except Exception:
                df = pd.read_csv(csv_path, header=None,
                                 names=['fname', 'labels', 'mids', 'split'], dtype=str)
            self.df = df

            eval_audio = os.path.join(root_dir, 'FSD50K.eval_audio')
            if not os.path.isdir(eval_audio):
                eval_audio = os.path.join(root_dir, 'clips', 'eval')
            self.audio_dir = eval_audio

        self.fnames = self.df['fname'].tolist()
        self.raw_labels = self.df['labels'].tolist()

        self.wave_aug = WaveformAugment(sample_rate=self.target_sr)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        raw_label_str = self.raw_labels[idx]

        labels = [l.strip() for l in raw_label_str.split(',')]
        target = torch.zeros(len(self.label2id))
        for l in labels:
            if l in self.label2id:
                target[self.label2id[l]] = 1.0

        wav_path = os.path.join(self.audio_dir, f"{fname}.wav")
        try:
            waveform, sr = torchaudio.load(wav_path)
        except Exception:
            return torch.zeros(1, self.target_length, 128), target

        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        curr_len = waveform.shape[1]
        if curr_len < self.target_len:
            repeat_num = int(self.target_len / curr_len) + 1
            waveform = waveform.repeat(1, repeat_num)
            waveform = waveform[:, :self.target_len]
        else:
            if self.split == 'train':
                start = random.randint(0, curr_len - self.target_len)
            else:
                start = (curr_len - self.target_len) // 2
            waveform = waveform[:, start:start+self.target_len]

        if self.split == 'train':
            waveform = self.wave_aug(waveform)

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=self.target_sr, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10
        )

        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
        elif p < 0:
            fbank = fbank[:self.target_length, :]

        if self.split == 'train':
            fbank = fbank.transpose(0, 1).unsqueeze(0)
            for _ in range(2):
                fbank = self.freq_mask(fbank)
                fbank = self.time_mask(fbank)
            fbank = fbank.squeeze(0).transpose(0, 1)

        fbank = (fbank - (-4.26)) / (4.56 * 2)
        return fbank.unsqueeze(0), target

# =========================================================
# 4. Training logic
# =========================================================
def train_one_epoch(model, model_ema, loader, optimizer, criterion, device, mix_fn):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    for img, target in pbar:
        img, target = img.to(device), target.to(device)
        if mix_fn is not None:
            img, target = mix_fn(img, target)

        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        model_ema.update(model)

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    criterion = AsymmetricLoss()

    for img, target in tqdm(loader, desc="Validating", leave=False):
        img, target = img.to(device), target.to(device)
        logits = model(img)
        loss = criterion(logits, target)
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    try:
        mAP = average_precision_score(all_targets, all_preds, average='macro')
    except:
        mAP = 0.0
    return total_loss / len(loader), mAP

# =========================================================
# 5. Main entry point
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="FSD50K AudioMAE Fine-tuning")
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["tiny", "small", "base", "large"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--llrd_decay", type=float, default=0.75,
                        help="Layer-wise LR decay; 1.0 disables LLRD")
    parser.add_argument("--fold", type=int, default=1)  # unused, kept for compat
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_path", type=str,
                        default="datasets/FSD50K")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="ft_checkpoints/fsd50k")
    parser.add_argument("--init_method", type=str, default="xavier_uniform",
                        choices=["xavier_uniform", "xavier_normal", "trunc_normal_002", "kaiming_uniform"])
    parser.add_argument("--mixup_prob", type=float, default=0.5)
    parser.add_argument("--cutmix_prob", type=float, default=0.5)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0 < args.llrd_decay <= 1:
        parser.error("--llrd_decay must lie in (0, 1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use the repository-relative checkpoint convention when no path is given.
    if not args.pretrained:
        args.pretrained = os.path.join(
            "checkpoints_audiomae_v3", f"vit_{args.model_size}", "final.pth"
        )

    # Large model anti-overfit tuning
    if args.model_size == "large":
        args.lr = 5e-5
        args.ema_decay = 0.9995
        args.mixup_prob = 0.3
        args.cutmix_prob = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join(args.output_dir, f"vit_{args.model_size}", "ema")
    os.makedirs(exp_dir, exist_ok=True)

    # Logging
    logger = logging.getLogger(f"fsd50k_{args.model_size}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(exp_dir, "train.log"), mode='w')
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info(f"{'='*50}")
    logger.info(f"FSD50K ViT-{args.model_size.upper()}")
    logger.info(f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    logger.info(f"Mixup prob: {args.mixup_prob} | CutMix prob: {args.cutmix_prob}")
    logger.info(f"EMA decay: {args.ema_decay} | LLRD decay: {args.llrd_decay}")
    logger.info(f"Pretrained: {args.pretrained}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Init method: {args.init_method}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {exp_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"{'='*50}")

    # Dataset
    train_ds = FSD50KDataset(args.data_path, split='train')
    val_ds = FSD50KDataset(args.data_path, split='val')
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {len(train_ds.label2id)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=True)

    # Mixup + CutMix
    mix_fn = MixupCutmix(mixup_alpha=0.8, cutmix_alpha=1.0,
                         mix_prob=args.mixup_prob, cutmix_prob=args.cutmix_prob)

    # Model
    model_config = get_audiomae_config(args.model_size)
    model = AudioMAEClassifier(num_classes=len(train_ds.label2id), stride=16,
                               mlp_ratio=4.0, init_method=args.init_method, **model_config)

    if os.path.exists(args.pretrained):
        model.load_pretrained(args.pretrained, logger)
    else:
        logger.warning(f"Pretrained weights not found at {args.pretrained}. Training from scratch!")

    model.to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Parameters: {param_count:.1f}M")

    # EMA
    model_ema = ModelEmaV2(model, decay=args.ema_decay)

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    optimizer = optim.AdamW(
        get_parameter_groups_with_llrd(model, weight_decay=1e-2, lr=args.lr, layer_decay=args.llrd_decay)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mAP = 0.0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, model_ema, train_loader, optimizer, criterion, device, mix_fn)
        val_loss, val_mAP = validate(model_ema.module, val_loader, device)

        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val mAP: {val_mAP:.4f}")

        if val_mAP > best_mAP:
            best_mAP = val_mAP
            torch.save(model_ema.module.state_dict(), os.path.join(exp_dir, "best.pth"))
            logger.info("✅ New Best Model Saved!")

    logger.info(f"🏁 Training Finished. Best mAP ({args.model_size}): {best_mAP:.4f}")

    del model, model_ema, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()
