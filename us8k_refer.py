#!/usr/bin/env python
# =============================================================================
# UrbanSound8K fine-tuning: AudioMAE (tiny/small/base/large) + EMA + 200 epochs
# UrbanSound8K dataset with the project EMA, logging, and argparse protocol.
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
import copy
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

import timm
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

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
# EMA
# =========================================================
class ModelEma:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.module = copy.deepcopy(model)
        self.module.eval()
        self._param_pairs = list(zip(self.module.parameters(), model.parameters()))

    def update(self):
        with torch.no_grad():
            for ema_p, model_p in self._param_pairs:
                ema_p.mul_(self.decay).add_(model_p, alpha=1 - self.decay)

# =========================================================
# Waveform Augmentation
# =========================================================
class WaveformAugment:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, waveform):
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain
        if random.random() < 0.5:
            noise_level = random.uniform(0.001, 0.01)
            noise = torch.randn_like(waveform)
            waveform = waveform + noise * noise_level
        if random.random() < 0.5:
            shift = random.randint(0, waveform.shape[1])
            waveform = torch.roll(waveform, shifts=shift, dims=1)
        return waveform

# =========================================================
# Model Components
# =========================================================
class PatchEmbed_new(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        _, _, h, w = self.get_output_shape(img_size)
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

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
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

class AudioMAEClassifier(nn.Module):
    def __init__(self, img_size=(1024, 128), patch_size=16, stride=16, in_chans=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 num_classes=10, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, init_method="xavier_uniform", **kwargs):
        super().__init__()
        self.init_method = init_method
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size,
                                          in_chans=in_chans, embed_dim=embed_dim, stride=stride)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1],
                                                     self.patch_embed.patch_hw, cls_token=True)
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
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:, :].mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def load_pretrained(self, checkpoint_path, logger=None):
        log = logger.info if logger else print
        log(f"📥 Loading pretrained from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = ckpt.get('model_state') or ckpt.get('model') or ckpt
        new_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if 'decoder' in k or 'mask_token' in k:
                continue
            new_dict[k] = v
        msg = self.load_state_dict(new_dict, strict=False)
        missing = [k for k in msg.missing_keys if 'head' not in k]
        log(f"✅ Loaded. Missing (expected): {missing}")

# =========================================================
# UrbanSound8K dataset implementation.
# =========================================================
class UrbanSoundDataset(Dataset):
    def __init__(self, root_dir, test_fold=10, train=True, target_length=1024):
        self.root_dir = root_dir
        self.train = train
        self.target_sr = 16000
        self.target_samples = int(self.target_sr * 10.24)
        self.target_length = target_length

        meta_csv = os.path.join(root_dir, 'metadata', 'UrbanSound8K.csv')
        if not os.path.exists(meta_csv):
            meta_csv = os.path.join(root_dir, 'UrbanSound8K.csv')
        df = pd.read_csv(meta_csv)

        if train:
            self.df = df[df['fold'] != test_fold]
            self.wave_aug = WaveformAugment(sample_rate=self.target_sr)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=30)
            self.time_mask = T.TimeMasking(time_mask_param=80)
        else:
            self.df = df[df['fold'] == test_fold]

        self.files = self.df['slice_file_name'].tolist()
        self.folds = self.df['fold'].tolist()
        self.labels = self.df['classID'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.files[idx]
        fold = self.folds[idx]
        label = self.labels[idx]

        wav_path = os.path.join(self.root_dir, 'audio', f'fold{fold}', filename)
        try:
            waveform, sr = torchaudio.load(wav_path)
        except Exception:
            return torch.zeros(1, self.target_length, 128), label

        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.shape[1] < self.target_samples:
            repeat_num = int(self.target_samples / waveform.shape[1]) + 1
            waveform = waveform.repeat(1, repeat_num)
        if self.train:
            start = random.randint(0, waveform.shape[1] - self.target_samples)
        else:
            start = 0
        waveform = waveform[:, start:start + self.target_samples]

        if self.train:
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

        if self.train:
            fbank = fbank.transpose(0, 1).unsqueeze(0)
            fbank = self.freq_mask(fbank)
            fbank = self.time_mask(fbank)
            fbank = fbank.squeeze(0).transpose(0, 1)

        fbank = (fbank - (-4.26)) / (4.56 * 2)
        return fbank.unsqueeze(0), label

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="UrbanSound8K AudioMAE Fine-tuning + EMA")
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["tiny", "small", "base", "large"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--llrd_decay", type=float, default=0.75,
                        help="Layer-wise LR decay; 1.0 disables LLRD")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--fold", type=int, default=10, help="Test fold (1-10, default: 10)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_path", type=str,
                        default="datasets/UrbanSound8K")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="ft_checkpoints/us8k")
    parser.add_argument("--init_method", type=str, default="xavier_uniform",
                        choices=["xavier_uniform", "xavier_normal", "trunc_normal_002", "kaiming_uniform"])
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

    if not args.pretrained:
        args.pretrained = os.path.join(
            "checkpoints_audiomae_v3", f"vit_{args.model_size}", "final.pth"
        )

    # Large model anti-overfit tuning
    if args.model_size == "large":
        args.lr = 1e-4
        args.ema_decay = 0.9999

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join(args.output_dir, f"vit_{args.model_size}", "ema")
    os.makedirs(exp_dir, exist_ok=True)

    # Logging
    logger = logging.getLogger(f"us8k_{args.model_size}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(exp_dir, "train.log"), mode='w')
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info(f"{'='*50}")
    logger.info(f"UrbanSound8K ViT-{args.model_size.upper()} + EMA")
    logger.info(f"Fold: {args.fold} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    logger.info(f"EMA decay: {args.ema_decay} | LLRD decay: {args.llrd_decay}")
    logger.info(f"Pretrained: {args.pretrained}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Init method: {args.init_method}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {exp_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"{'='*50}")

    # Dataset
    train_ds = UrbanSoundDataset(args.data_path, test_fold=args.fold, train=True)
    val_ds = UrbanSoundDataset(args.data_path, test_fold=args.fold, train=False)
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True,
    )

    # Mixup
    mixup_prob = 0.5 if args.model_size == "large" else 1.0
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, prob=mixup_prob, switch_prob=0.5,
        mode='batch', label_smoothing=0.1, num_classes=10,
    )

    # Model
    model_config = get_audiomae_config(args.model_size)
    model = AudioMAEClassifier(
        num_classes=10, stride=16, init_method=args.init_method, **model_config
    )
    model.to(device)
    if os.path.exists(args.pretrained):
        model.load_pretrained(args.pretrained, logger)
    else:
        logger.warning(f"Pretrained weights not found at {args.pretrained}. Training from scratch!")

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Parameters: {param_count:.1f}M")

    # EMA
    model_ema = ModelEma(model, decay=args.ema_decay)

    # Optimizer & AMP
    optimizer = optim.AdamW(
        get_parameter_groups_with_llrd(model, weight_decay=0.05, lr=args.lr, layer_decay=args.llrd_decay)
    )
    criterion_train = SoftTargetCrossEntropy()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[US8K-{args.model_size}] Epoch {epoch+1}/{args.epochs}")

        for img, label in pbar:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if mixup_fn is not None:
                img, label = mixup_fn(img, label)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(img)
                loss = criterion_train(logits, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model_ema.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation — use EMA weights
        model_ema.module.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for img, label in tqdm(val_loader, desc=f"[US8K-{args.model_size}] Val {epoch+1}/{args.epochs}", leave=False):
                img = img.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    preds = model_ema.module(img).argmax(dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)
        val_acc = 100 * correct / total

        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_ema.module.state_dict(), os.path.join(exp_dir, "best.pth"))

    logger.info(f"🏁 Complete. Best Val Acc ({args.model_size}, fold {args.fold}): {best_acc:.2f}%")

    del model, model_ema, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
