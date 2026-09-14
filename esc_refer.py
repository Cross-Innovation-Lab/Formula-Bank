#!/usr/bin/env python
# =============================================================================
# Parameterized canonical ESC-50 fine-tuning: AudioMAE + EMA + LLRD + OneCycleLR.
# Its defaults intentionally match esc_formal_version.py for new experiments.
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
from functools import partial
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import timm
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
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
            noise_level = random.uniform(0.001, 0.015)
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

# =========================================================
# Core Model
# =========================================================
class AudioMAEClassifier(nn.Module):
    def __init__(self, img_size=(1024, 128), patch_size=16, stride=16, in_chans=1,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 num_classes=50, drop_path_rate=0.1,
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

    def load_pretrained_mae(self, checkpoint_path, logger=None):
        log = logger.info if logger else print
        log(f"📥 Loading pretrained weights from {checkpoint_path}...")
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

            # Robust key finding: supports model_state (pretrain.py), model, state_dict, or raw
            if 'model_state' in ckpt:
                state_dict = ckpt['model_state']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt

            new_dict = {}
            for k, v in state_dict.items():
                # Strip common prefixes
                if k.startswith('module.'):
                    k = k[7:]
                if k.startswith('backbone.'):
                    k = k[9:]
                if k.startswith('encoder.'):
                    k = k[8:]

                # Exclude decoder and mask tokens
                if 'decoder' in k or 'mask_token' in k:
                    continue
                new_dict[k] = v

            msg = self.load_state_dict(new_dict, strict=False)
            log(f"✅ Loaded successfully! Msg: {msg}")
        except Exception as e:
            log(f"❌ Load failed: {e}")

# =========================================================
# Model config & LLRD
# =========================================================
def get_audiomae_config(model_size):
    """Dynamic hyperparameters for different scaling sizes"""
    configs = {
        "tiny":  {"embed_dim": 192,  "depth": 12, "num_heads": 3,  "drop_path_rate": 0.0},
        "small": {"embed_dim": 384,  "depth": 12, "num_heads": 6,  "drop_path_rate": 0.1},
        "base":  {"embed_dim": 768,  "depth": 12, "num_heads": 12, "drop_path_rate": 0.2},
        "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "drop_path_rate": 0.3},
    }
    if model_size.lower() not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    return configs[model_size.lower()]

def get_parameter_groups_with_llrd(model, weight_decay=0.05, lr=2e-4, layer_decay=0.75):
    """Layer-wise Learning Rate Decay"""
    parameter_group_names = {}
    parameter_group_vars = {}
    num_layers = model.depth

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith(".bias") or name in ['cls_token', 'pos_embed']:
            wd = 0.0
        else:
            wd = weight_decay

        if name.startswith('patch_embed') or name in ['cls_token', 'pos_embed']:
            layer_id = 0
        elif name.startswith('blocks.'):
            layer_id = int(name.split('.')[1]) + 1
        else:
            layer_id = num_layers + 1

        scale = layer_decay ** (num_layers + 1 - layer_id)
        group_name = f"layer_{layer_id}_{wd}"

        if group_name not in parameter_group_names:
            parameter_group_names[group_name] = {
                "weight_decay": wd,
                "params": [],
                "lr": lr * scale
            }
            parameter_group_vars[group_name] = parameter_group_names[group_name]

        parameter_group_vars[group_name]["params"].append(param)

    return list(parameter_group_vars.values())

# =========================================================
# ESC-50 Dataset
# =========================================================
class ESC50Dataset(Dataset):
    def __init__(self, root_dir, fold=1, train=True, target_length=1024):
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, 'audio')
        self.meta_path = os.path.join(root_dir, 'meta', 'esc50.csv')
        self.target_length = target_length
        self.train = train
        self.sr = 16000
        self.target_samples = int(self.sr * 10.24)

        df = pd.read_csv(self.meta_path)
        if train:
            self.df = df[df['fold'] != fold]
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=30)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=80)
            self.wave_aug = WaveformAugment(sample_rate=self.sr)
        else:
            self.df = df[df['fold'] == fold]

        self.categories = sorted(df['category'].unique())
        self.cat2idx = {cat: i for i, cat in enumerate(self.categories)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.audio_dir, row['filename'])
        label = self.cat2idx[row['category']]

        waveform, sr = torchaudio.load(wav_path)

        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.shape[1] < self.target_samples:
            repeat_num = (self.target_samples // waveform.shape[1]) + 1
            waveform = waveform.repeat(1, repeat_num)
        waveform = waveform[:, :self.target_samples]

        if self.train:
            waveform = self.wave_aug(waveform)

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=self.sr, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10
        )

        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
        else:
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
    parser = argparse.ArgumentParser(description="ESC-50 AudioMAE Fine-tuning + EMA")
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["tiny", "small", "base", "large"],
                        help="ViT variant (default: base)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_path", type=str,
                        default="datasets/ESC-50/ESC-50-master")
    parser.add_argument("--pretrained", type=str, default="",
                        help="Path to pretrained checkpoint. Auto-generated if empty.")
    parser.add_argument("--init_method", type=str, default="xavier_uniform",
                        choices=["xavier_uniform", "xavier_normal", "trunc_normal_002", "kaiming_uniform"],
                        help="Initialization used when training from scratch or for newly initialized heads.")
    parser.add_argument("--llrd_decay", type=float, default=0.75,
                        help="Layer-wise LR decay; 1.0 disables LLRD")
    parser.add_argument("--output_dir", type=str, default="ft_checkpoints/esc50",
                        help="Root output directory")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

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

    if not 0 < args.llrd_decay <= 1:
        parser.error("--llrd_decay must lie in (0, 1]")

    # Large model anti-overfit tuning
    if args.model_size == "large":
        args.lr = 1e-4
        args.ema_decay = 0.9999

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.join(args.output_dir, f"vit_{args.model_size}", "ema")
    os.makedirs(exp_dir, exist_ok=True)

    # Logging
    logger = logging.getLogger(f"esc50_{args.model_size}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(exp_dir, "train.log"), mode='w')
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info(f"{'='*50}")
    logger.info(f"ESC-50 ViT-{args.model_size.upper()} + EMA + LLRD + OneCycleLR")
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
    train_ds = ESC50Dataset(args.data_path, fold=args.fold, train=True)
    val_ds = ESC50Dataset(args.data_path, fold=args.fold, train=False)
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

    mixup_prob = 0.5 if args.model_size == "large" else 1.0
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, prob=mixup_prob,
        switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=50,
    )

    # Model
    logger.info(f"Building AudioMAE Classifier ({args.model_size})...")
    model_config = get_audiomae_config(args.model_size)
    model = AudioMAEClassifier(
        num_classes=50,
        stride=16,
        init_method=args.init_method,
        **model_config
    )

    if os.path.exists(args.pretrained):
        model.load_pretrained_mae(args.pretrained, logger)
    else:
        logger.warning(f"Pretrained weights not found at {args.pretrained}. Training from scratch!")

    model.to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Parameters: {param_count:.1f}M")

    # AMP
    scaler = torch.amp.GradScaler('cuda')

    # EMA (timm ModelEmaV2)
    model_ema = ModelEmaV2(model, decay=args.ema_decay)

    # Optimizer with LLRD
    param_groups = get_parameter_groups_with_llrd(
        model, weight_decay=0.05, lr=args.lr, layer_decay=args.llrd_decay
    )
    optimizer = optim.AdamW(param_groups)

    # OneCycleLR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group['lr'] for group in optimizer.param_groups],
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    criterion = SoftTargetCrossEntropy()
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for img, label in pbar:
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)

            if mixup_fn is not None:
                img, label = mixup_fn(img, label)

            with torch.amp.autocast('cuda'):
                logits = model(img)
                loss = criterion(logits, label)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            model_ema.update(model)

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[-1]['lr']:.6f}"})

        # Validation — use EMA weights
        model_ema.module.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for img, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model_ema.module(img)
                _, pred = torch.max(logits, 1)
                val_correct += (pred == label).sum().item()
                total_val += label.size(0)

        val_acc = 100 * val_correct / total_val
        logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss/len(train_loader):.4f} | EMA Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_ema.module.state_dict(), os.path.join(exp_dir, "best.pth"))

    logger.info(f"🏁 Training Finished. Best Val Acc ({args.model_size}): {best_acc:.2f}%")

    del model, model_ema, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
