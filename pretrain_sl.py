import argparse
import collections
import collections.abc
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import random
import socket
import sys
import time
import types
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.functional as F
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



def make_grad_scaler(enabled):
    """Compatibility wrapper for new and old PyTorch AMP APIs."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_cuda(enabled):
    """Compatibility wrapper for new and old PyTorch autocast APIs."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)

# =========================================================
# Environment compatibility patches required by the pinned dependencies.
# =========================================================
print("🔧 Applying environment patches...")
try:
    import torch._six
except ImportError:
    _six = types.ModuleType('torch._six')
    _six.container_abcs = collections.abc
    _six.string_classes = str
    sys.modules['torch._six'] = _six
    torch._six = _six

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models.vision_transformer import Block
try:
    from timm.layers import to_2tuple
except ImportError:
    from timm.models.layers import to_2tuple

# =========================================================
# Model components (PatchEmbed and positional embeddings).
# =========================================================
class PatchEmbed_new(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
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

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

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
# AudioMAE model definition with the pinned timm compatibility fix.
# =========================================================
class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=10, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 audio_exp=False, decoder_mode=0, 
                 use_custom_patch=False, pos_trainable=False, **kwargs):
        super().__init__()

        self.audio_exp = audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        # Encoder
        # Pre-training uses patch_size as the stride, so patches do not overlap.
        if use_custom_patch:
            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                              embed_dim=embed_dim, stride=stride)
        else:
            self.patch_embed = PatchEmbed_new(img_size, patch_size, in_chans, embed_dim, stride=patch_size)

        self.use_custom_patch = use_custom_patch
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)
        
        # Compatibility fix: qk_scale is omitted for the pinned timm version.
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=pos_trainable)
        self.decoder_mode = decoder_mode
        self.norm_pix_loss = norm_pix_loss
        
        # Compatibility fix: qk_scale is omitted for the pinned timm version.
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
            decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True)
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True)
            
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        # Handling for audio (1024, 128)
        if self.audio_exp:
             # Regular grid patchify
            h = imgs.shape[2] // p
            w = imgs.shape[3] // p
            x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = 1024 // p
        w = 128 // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)
        pred = pred[:, 1:, :]
        return pred

    def forward_loss(self, imgs, pred, mask, norm_pix_loss=False):
        target = self.patchify(imgs)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        emb_enc, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(emb_enc, ids_restore)
        loss = self.forward_loss(imgs, pred, mask, norm_pix_loss=self.norm_pix_loss)
        return loss, pred, mask

# =========================================================
# 🏗️ AudioMAE ViT Variant Factories
# =========================================================
# Model variants use 16x16 patches. The Large variant follows the common
# ViT-L configuration, while the decoder remains lightweight for pre-training.

def mae_vit_tiny_patch16(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def mae_vit_small_patch16(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def mae_vit_base_patch16(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def mae_vit_large_patch16(**kwargs):
    return MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


MODEL_FACTORIES = {
    "tiny": mae_vit_tiny_patch16,
    "small": mae_vit_small_patch16,
    "base": mae_vit_base_patch16,
    "large": mae_vit_large_patch16,
}

# Conservative single-GPU batch-size defaults keep the Large variant runnable.
DEFAULT_BATCH_SIZE = {
    "tiny": 64,
    "small": 32,
    "base": 16,
    "large": 4,
}

DEFAULT_SYNTH_CONFIG = {
    "name": "baseline_current",
    "f0_distribution": "uniform",
    "f0_range_hz": [50.0, 2000.0],
    "source_types": ["harmonic", "inharmonic", "modal", "fm", "chirp", "pulse", "noise_texture"],
    "source_type_weights": [0.24, 0.16, 0.16, 0.16, 0.10, 0.10, 0.08],
    # Optional fixed-count controls. When unset, retain the legacy weighted
    # source/event sampling below.
    "source_count": None,
    "total_event_count": None,
    "events_per_source": None,
    "event_allocation": "legacy",
    "source_count_values": [1, 2, 3, 4, 5, 6],
    "source_count_weights": [0.22, 0.25, 0.22, 0.16, 0.10, 0.05],
    "event_count_values": [1, 2, 3, 5, 8, 12, 20],
    "event_count_weights": [0.18, 0.20, 0.20, 0.17, 0.12, 0.08, 0.05],
    "onset_modes": ["random", "clustered", "rhythmic", "dense"],
    "onset_mode_weights": [0.25, 0.25, 0.25, 0.25],
    "local_f0_mult": [0.35, 3.0],
    "event_f0_mult": [0.7, 1.4],
    "event_duration": [0.03, 4.0],
    "event_gain_db": [-18.0, 0.0],
    "source_gain_db": [-12.0, 0.0],
    "transient_prob": 0.65,
    "event_spectral_shape_prob": 0.80,
    "source_tremolo_prob": 0.45,
    "source_spectral_shape_prob": 1.0,
    "background_noise_prob": 0.95,
    "snr_db": [5.0, 35.0],
    "channel_reflection_prob": 0.55,
    "channel_bandlimit_prob": 0.50,
    "channel_saturation_prob": 0.35,
}

SYNTH_CONFIG = dict(DEFAULT_SYNTH_CONFIG)
SYNTH_CONFIG_HASH = "default"


def _merge_synth_config(user_config):
    cfg = dict(DEFAULT_SYNTH_CONFIG)
    if user_config:
        cfg.update(user_config)
    return cfg


def _load_synth_config(path):
    if not path:
        return dict(DEFAULT_SYNTH_CONFIG), "default"
    with open(path, "r", encoding="utf-8") as f:
        user_config = json.load(f)
    cfg = _merge_synth_config(user_config)
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    cfg_hash = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return cfg, cfg_hash


def set_synth_config(cfg, cfg_hash="default"):
    global SYNTH_CONFIG, SYNTH_CONFIG_HASH
    SYNTH_CONFIG = cfg
    SYNTH_CONFIG_HASH = cfg_hash

# =========================================================
# Procedural audio synthesizer with configurable acoustic complexity.
# Acoustic-Complexity Procedural Synthesizer
#
# The synthesizer increases low-level acoustic complexity without semantic labels.
# It supports multiple sources, event layouts, diverse source families, transients,
# spectral coloration, and lightweight channel perturbations.
#
# Describe this as an acoustically motivated procedural synthesizer or stochastic
# spectral coloration, rather than a complete physical propagation simulation.
# =========================================================
def _normalize_audio(x, eps=1e-8):
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean()
    peak = x.abs().max()
    if peak > eps:
        x = x / peak
    return x.clamp(-1.0, 1.0)


def _rms(x, eps=1e-8):
    return torch.sqrt(torch.mean(x.float() ** 2) + eps)


def _apply_db_gain(x, db):
    return x * (10.0 ** (db / 20.0))


def _safe_biquad(fn, x, sample_rate, *args, **kwargs):
    """Wrap torchaudio biquads and guard against invalid random parameters."""
    try:
        y = fn(x.view(1, -1), sample_rate, *args, **kwargs).squeeze(0)
        if torch.isfinite(y).all():
            return y
        return x
    except Exception:
        return x


def _colored_noise(num_samples, sample_rate, color=None):
    """Generate white, dark, bright, or brown noise for texture or noise floors."""
    x = torch.randn(num_samples)

    if color is None:
        color = random.choice(["white", "dark", "bright", "brown"])

    if color == "dark":
        cutoff = random.uniform(200, 2500)
        x = _safe_biquad(F.lowpass_biquad, x, sample_rate, cutoff_freq=cutoff)
    elif color == "bright":
        cutoff = random.uniform(1000, 5000)
        x = _safe_biquad(F.highpass_biquad, x, sample_rate, cutoff_freq=cutoff)
    elif color == "brown":
        x = torch.cumsum(x, dim=0)

    return _normalize_audio(x)


def _make_envelope(n, sample_rate):
    """Generate an envelope family with attack, hold, decay, release, and power curves."""
    if n <= 8:
        return torch.ones(n)

    attack = max(1, int(n * random.uniform(0.001, 0.25)))
    hold = int(n * random.uniform(0.0, 0.25))
    decay = max(1, int(n * random.uniform(0.05, 0.45)))
    release = max(1, n - attack - hold - decay)
    sustain = random.uniform(0.1, 0.9)

    a = torch.linspace(0, 1, attack)
    h = torch.ones(hold) if hold > 0 else torch.empty(0)
    d = torch.linspace(1, sustain, decay)
    r = torch.linspace(sustain, 0, release)
    env = torch.cat([a, h, d, r])

    if len(env) < n:
        env = torch.cat([env, torch.zeros(n - len(env))])
    env = env[:n].clamp(min=0.0)

    # Power curves cover percussive and sustained envelope shapes.
    if random.random() < 0.55:
        env = env ** random.uniform(0.45, 3.0)

    return env


def _harmonic_event(n, sample_rate, f0):
    """Synthesize additive harmonics with vibrato, drift, and partial-level AM."""
    t = torch.arange(n).float() / sample_rate
    x = torch.zeros(n)

    k_max = random.randint(3, 30)
    rolloff = random.uniform(0.6, 2.0)

    vibrato_rate = random.uniform(0.2, 8.0)
    vibrato_depth = random.uniform(0.0, 0.025)
    drift = random.uniform(-0.04, 0.04)

    f_base = f0 * (
        1.0
        + vibrato_depth * torch.sin(2 * math.pi * vibrato_rate * t)
        + drift * torch.linspace(-0.5, 0.5, n)
    )
    f_base = torch.clamp(f_base, 20.0, sample_rate * 0.45)
    base_phase = 2 * math.pi * torch.cumsum(f_base, dim=0) / sample_rate

    for k in range(1, k_max + 1):
        if f0 * k > sample_rate * 0.45:
            break
        amp = random.uniform(0.4, 1.2) / (k ** rolloff)
        detune = random.uniform(0.995, 1.005)
        phase = base_phase * k * detune + random.uniform(0, 2 * math.pi)

        if random.random() < 0.5:
            mod_rate = random.uniform(0.1, 5.0)
            mod_depth = random.uniform(0.05, 0.5)
            partial_env = 1.0 + mod_depth * torch.sin(
                2 * math.pi * mod_rate * t + random.uniform(0, 2 * math.pi)
            )
        else:
            partial_env = 1.0

        x += amp * partial_env * torch.sin(phase)

    return _normalize_audio(x)


def _inharmonic_event(n, sample_rate, f0):
    """Synthesize inharmonic partial ratios with exponential decay."""
    t = torch.arange(n).float() / sample_rate
    x = torch.zeros(n)

    n_partials = random.randint(4, 24)
    stretch = random.uniform(1.01, 1.25)

    for i in range(1, n_partials + 1):
        ratio = (i ** stretch) * random.uniform(0.97, 1.03)
        freq = f0 * ratio
        if freq > sample_rate * 0.45:
            continue
        amp = random.uniform(0.2, 1.0) / (i ** random.uniform(0.7, 1.7))
        decay = random.uniform(0.2, 5.0)
        phase = random.uniform(0, 2 * math.pi)
        partial_env = torch.exp(-decay * t)
        x += amp * partial_env * torch.sin(2 * math.pi * freq * t + phase)

    return _normalize_audio(x)


def _modal_event(n, sample_rate, f0):
    """Synthesize generic modal resonances without semantic class assumptions."""
    t = torch.arange(n).float() / sample_rate
    x = torch.zeros(n)

    n_modes = random.randint(3, 18)
    decay_global = random.uniform(0.3, 8.0)

    for _ in range(n_modes):
        ratio = random.uniform(0.7, 12.0)
        freq = f0 * ratio * random.uniform(0.97, 1.05)
        if freq > sample_rate * 0.45:
            continue
        amp = random.uniform(0.2, 1.0) / math.sqrt(ratio)
        decay = decay_global * random.uniform(0.5, 2.5)
        phase = random.uniform(0, 2 * math.pi)
        x += amp * torch.exp(-decay * t) * torch.sin(2 * math.pi * freq * t + phase)

    return _normalize_audio(x)


def _fm_event(n, sample_rate, f0):
    """Synthesize two-operator FM with a time-varying modulation index."""
    t = torch.arange(n).float() / sample_rate

    carrier = f0 * random.uniform(0.5, 2.0)
    carrier = max(20.0, min(carrier, sample_rate * 0.4))
    ratio = random.choice([0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, math.pi])
    mod_freq = carrier * ratio
    index = random.uniform(0.2, 12.0)

    index_env = torch.linspace(random.uniform(0.2, index), random.uniform(0.2, index), n)
    mod = index_env * torch.sin(2 * math.pi * mod_freq * t + random.uniform(0, 2 * math.pi))
    x = torch.sin(2 * math.pi * carrier * t + mod)

    return _normalize_audio(x)


def _chirp_event(n, sample_rate, f0):
    """Synthesize linear or nonlinear sweeps with optional wobble."""
    t = torch.arange(n).float() / sample_rate

    start = f0 * random.uniform(0.3, 2.0)
    end = f0 * random.uniform(0.3, 4.0)
    start = max(20.0, min(start, sample_rate * 0.4))
    end = max(20.0, min(end, sample_rate * 0.4))

    curve = random.uniform(0.5, 3.0)
    u = torch.linspace(0, 1, n) ** curve
    f = start * (1 - u) + end * u

    if random.random() < 0.5:
        f = f * (1 + random.uniform(0.005, 0.04) * torch.sin(2 * math.pi * random.uniform(2, 20) * t))

    phase = 2 * math.pi * torch.cumsum(f, dim=0) / sample_rate
    x = torch.sin(phase + random.uniform(0, 2 * math.pi))
    return _normalize_audio(x)


def _pulse_event(n, sample_rate, f0):
    """Synthesize soft square, saw, or pulse excitation without physical-source claims."""
    t = torch.arange(n).float() / sample_rate
    mode = random.choice(["square_soft", "saw_soft", "pulse_train"])
    f = max(20.0, min(f0 * random.uniform(0.5, 4.0), sample_rate * 0.35))

    if mode == "square_soft":
        x = torch.tanh(random.uniform(2.0, 8.0) * torch.sin(2 * math.pi * f * t))
    elif mode == "saw_soft":
        phase = (t * f) % 1.0
        x = 2.0 * phase - 1.0
        x = torch.tanh(random.uniform(1.0, 4.0) * x)
    else:
        x = torch.zeros(n)
        period = max(1, int(sample_rate / f))
        width = random.randint(1, max(2, period // 8))
        for pos in range(0, n, period):
            end = min(n, pos + width)
            x[pos:end] = 1.0
        x = x + 0.02 * torch.randn(n)

    return _normalize_audio(x)


def _noise_texture_event(n, sample_rate):
    """Synthesize colored-noise textures with slow modulation."""
    x = _colored_noise(n, sample_rate)
    t = torch.arange(n).float() / sample_rate

    for _ in range(random.randint(1, 3)):
        rate = random.uniform(0.1, 12.0)
        depth = random.uniform(0.05, 0.7)
        x = x * (1.0 + depth * torch.sin(2 * math.pi * rate * t + random.uniform(0, 2 * math.pi)))

    return _normalize_audio(x)


def _render_event(n, sample_rate, f0, source_type=None):
    """Render one primitive event.

    ``source_type`` is an opt-in override for experiments that need to retain
    the primitive identity as a label.  Leaving it as ``None`` deliberately
    keeps the legacy random draw and RNG order unchanged.
    """
    cfg = SYNTH_CONFIG
    if source_type is None:
        source_type = random.choices(
            cfg["source_types"],
            weights=cfg["source_type_weights"],
        )[0]
    elif source_type not in cfg["source_types"]:
        raise ValueError(f"Unsupported source_type override: {source_type!r}")

    if source_type == "harmonic":
        x = _harmonic_event(n, sample_rate, f0)
    elif source_type == "inharmonic":
        x = _inharmonic_event(n, sample_rate, f0)
    elif source_type == "modal":
        x = _modal_event(n, sample_rate, f0)
    elif source_type == "fm":
        x = _fm_event(n, sample_rate, f0)
    elif source_type == "chirp":
        x = _chirp_event(n, sample_rate, f0)
    elif source_type == "pulse":
        x = _pulse_event(n, sample_rate, f0)
    else:
        x = _noise_texture_event(n, sample_rate)

    return x, source_type


def _add_transient_complex(x, sample_rate):
    """Synthesize complex transients from decaying noise bursts near onset."""
    n = len(x)
    y = x.clone()

    if random.random() < 0.75:
        n_bursts = random.randint(1, 8)
        max_start = min(n - 1, int(0.25 * sample_rate))

        for _ in range(n_bursts):
            start = random.randint(0, max(0, max_start))
            available = n - start
            if available <= 0:
                continue

            length = int(random.uniform(0.002, 0.08) * sample_rate)
            # Important: do not force length >= 4 after clipping to available samples.
            # Otherwise a tail slice like y[-1:] has shape [1] while burst has shape [4].
            length = max(1, min(length, available))

            burst = torch.randn(length)
            decay = torch.exp(-torch.linspace(0, random.uniform(2, 12), length))
            burst = burst * decay

            if length >= 4 and random.random() < 0.5:
                burst = _safe_biquad(F.highpass_biquad, burst, sample_rate, cutoff_freq=random.uniform(800, 6000))

            y[start:start + length] += burst[:length] * random.uniform(0.05, 0.8)

    return _normalize_audio(y)


def _spectral_shape(x, sample_rate):
    """Apply a random spectral-coloration filter chain."""
    y = x.clone()

    n_filters = random.randint(0, 4)
    for _ in range(n_filters):
        mode = random.choice(["lp", "hp", "bp", "notch_like"])

        if mode == "lp":
            cutoff = random.uniform(300, sample_rate * 0.45)
            y = _safe_biquad(F.lowpass_biquad, y, sample_rate, cutoff_freq=cutoff)
        elif mode == "hp":
            cutoff = random.uniform(30, 3000)
            y = _safe_biquad(F.highpass_biquad, y, sample_rate, cutoff_freq=cutoff)
        elif mode == "bp":
            center = random.uniform(100, sample_rate * 0.4)
            q = random.uniform(0.3, 5.0)
            y = _safe_biquad(F.bandpass_biquad, y, sample_rate, central_freq=center, Q=q)
        else:
            # Approximate notch-like coloration when notch_biquad is unavailable.
            center = random.uniform(200, sample_rate * 0.4)
            q = random.uniform(0.5, 8.0)
            band = _safe_biquad(F.bandpass_biquad, y, sample_rate, central_freq=center, Q=q)
            y = y - random.uniform(0.2, 0.9) * band

    return _normalize_audio(y)


def _apply_channel_effects(x, sample_rate):
    """Apply lightweight channel effects such as reflections and saturation."""
    cfg = SYNTH_CONFIG
    y = x.clone()

    # Early reflections / short multi-tap delay
    if random.random() < float(cfg.get("channel_reflection_prob", 0.55)):
        dry = y.clone()
        for _ in range(random.randint(1, 5)):
            delay = int(random.uniform(0.003, 0.08) * sample_rate)
            gain = random.uniform(0.03, 0.35)
            if delay < len(y):
                y[delay:] += gain * dry[:-delay]

    # Random band limitation
    if random.random() < float(cfg.get("channel_bandlimit_prob", 0.50)):
        hp = random.uniform(20, 300)
        lp = random.uniform(2500, sample_rate * 0.45)
        y = _safe_biquad(F.highpass_biquad, y, sample_rate, cutoff_freq=hp)
        y = _safe_biquad(F.lowpass_biquad, y, sample_rate, cutoff_freq=lp)

    # Mild saturation / soft clipping
    if random.random() < float(cfg.get("channel_saturation_prob", 0.35)):
        drive = random.uniform(1.0, 4.0)
        y = torch.tanh(drive * y)

    return _normalize_audio(y)


def _place(dst, src, start):
    end = min(len(dst), start + len(src))
    if end > start:
        dst[start:end] += src[:end - start]
    return dst


def _sample_event_starts(num_samples, sample_rate, count):
    """Generate random, clustered, rhythmic, or dense temporal event layouts."""
    cfg = SYNTH_CONFIG
    mode = random.choices(
        cfg.get("onset_modes", ["random", "clustered", "rhythmic", "dense"]),
        weights=cfg.get("onset_mode_weights", [0.25, 0.25, 0.25, 0.25]),
    )[0]

    if mode == "random":
        return [random.randint(0, num_samples - 1) for _ in range(count)]

    if mode == "clustered":
        center = random.randint(0, num_samples - 1)
        spread = int(random.uniform(0.05, 1.5) * sample_rate)
        starts = []
        for _ in range(count):
            s = center + random.randint(-spread, spread)
            starts.append(max(0, min(num_samples - 1, s)))
        return starts

    if mode == "rhythmic":
        interval = int(random.uniform(0.08, 0.8) * sample_rate)
        start = random.randint(0, max(0, min(num_samples - 1, interval)))
        starts = []
        pos = start
        while pos < num_samples and len(starts) < count:
            jitter = int(random.uniform(-0.08, 0.08) * interval)
            starts.append(max(0, min(num_samples - 1, pos + jitter)))
            pos += interval
        return starts

    return sorted([random.randint(0, num_samples - 1) for _ in range(count)])


def _require_positive_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if result < 1 or result != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def _resolve_source_event_counts(cfg):
    """Resolve per-source event counts while preserving legacy configs."""
    fixed_source_count = cfg.get("source_count")
    if fixed_source_count is None:
        n_sources = random.choices(
            cfg["source_count_values"],
            weights=cfg["source_count_weights"],
        )[0]
    else:
        n_sources = _require_positive_int(fixed_source_count, "source_count")

    total_event_count = cfg.get("total_event_count")
    events_per_source = cfg.get("events_per_source")
    if total_event_count is not None:
        total_event_count = _require_positive_int(total_event_count, "total_event_count")
        allocation = str(cfg.get("event_allocation", "legacy"))
        if allocation == "legacy":
            allocation = "equal_per_source"
        if allocation != "equal_per_source":
            raise ValueError(
                "total_event_count currently requires event_allocation='equal_per_source'"
            )
        if total_event_count % n_sources != 0:
            raise ValueError(
                "total_event_count must be divisible by source_count for equal_per_source "
                f"allocation, got {total_event_count} and {n_sources}"
            )
        resolved_events_per_source = total_event_count // n_sources
        if events_per_source is not None:
            configured_events_per_source = _require_positive_int(
                events_per_source, "events_per_source"
            )
            if configured_events_per_source != resolved_events_per_source:
                raise ValueError(
                    "events_per_source disagrees with total_event_count/source_count: "
                    f"{configured_events_per_source} != {resolved_events_per_source}"
                )
        return n_sources, [resolved_events_per_source] * n_sources

    if events_per_source is not None:
        resolved_events_per_source = _require_positive_int(events_per_source, "events_per_source")
        return n_sources, [resolved_events_per_source] * n_sources

    return n_sources, [
        random.choices(
            cfg["event_count_values"],
            weights=cfg["event_count_weights"],
        )[0]
        for _ in range(n_sources)
    ]


def advanced_physics_synth(
    f0_hz,
    duration=10.24,
    sample_rate=16000,
    return_metadata=False,
    event_counts_override=None,
    primitive_sequence=None,
):
    """
    AudioPG complex synthesizer v3.

    Legacy-compatible inputs are supported:
        f0_hz: sampled fundamental frequency.
        duration: default 10.24 s, corresponding to 1024 log-Mel frames.
        sample_rate: default 16 kHz.

    The default output is a ``torch.Tensor`` with shape
    ``[duration * sample_rate]``. With ``return_metadata=True``, the function
    returns ``(waveform, metadata)`` for primitive-labeled streams.
    """
    cfg = SYNTH_CONFIG
    num_samples = int(duration * sample_rate)
    mix = torch.zeros(num_samples)

    # Multiple independent sources are optional controls and do not introduce
    # semantic labels; default random-number consumption is preserved.
    if event_counts_override is None:
        _, event_counts = _resolve_source_event_counts(cfg)
    else:
        event_counts = [
            _require_positive_int(count, "event_counts_override item")
            for count in event_counts_override
        ]
        if not event_counts:
            raise ValueError("event_counts_override must contain at least one source")

    total_events = sum(event_counts)
    if primitive_sequence is not None:
        primitive_sequence = list(primitive_sequence)
        if len(primitive_sequence) != total_events:
            raise ValueError(
                "primitive_sequence length must equal the overridden total event count: "
                f"{len(primitive_sequence)} != {total_events}"
            )
        unknown = set(primitive_sequence) - set(cfg["source_types"])
        if unknown:
            raise ValueError(f"primitive_sequence contains unsupported primitives: {sorted(unknown)}")

    event_metadata = [] if return_metadata else None
    primitive_index = 0

    for source_index, n_events in enumerate(event_counts):
        source_bus = torch.zeros(num_samples)

        local_f0_min, local_f0_max = cfg.get("local_f0_mult", [0.35, 3.0])
        local_f0 = float(f0_hz) * random.uniform(float(local_f0_min), float(local_f0_max))
        local_f0 = max(30.0, min(local_f0, sample_rate * 0.25))

        starts = _sample_event_starts(num_samples, sample_rate, n_events)

        for event_index, start in enumerate(starts):
            remaining = num_samples - start
            # Labelled sparse scenes must materialize every requested primitive.
            # The legacy branch intentionally preserves its historical tail skip.
            if event_counts_override is not None and remaining <= 8:
                start = max(0, num_samples - 16)
                remaining = num_samples - start
            if remaining <= 8:
                continue

            event_dur_min, event_dur_max = cfg.get("event_duration", [0.03, 4.0])
            event_dur = random.uniform(float(event_dur_min), min(duration, float(event_dur_max)))
            event_n = int(event_dur * sample_rate)
            event_n = min(max(16, event_n), remaining)

            event_f0_min, event_f0_max = cfg.get("event_f0_mult", [0.7, 1.4])
            event_f0 = local_f0 * random.uniform(float(event_f0_min), float(event_f0_max))
            primitive = None if primitive_sequence is None else primitive_sequence[primitive_index]
            event, source_type = _render_event(event_n, sample_rate, event_f0, primitive)
            primitive_index += 1

            env = _make_envelope(event_n, sample_rate)
            event = event * env

            if random.random() < float(cfg.get("transient_prob", 0.65)):
                event = _add_transient_complex(event, sample_rate)

            if random.random() < float(cfg.get("event_spectral_shape_prob", 0.80)):
                event = _spectral_shape(event, sample_rate)

            event_gain_min, event_gain_max = cfg.get("event_gain_db", [-18.0, 0.0])
            event = _apply_db_gain(event, random.uniform(float(event_gain_min), float(event_gain_max)))
            if return_metadata:
                event_metadata.append({
                    "source_index": source_index,
                    "event_index": event_index,
                    "primitive": source_type,
                    "start_sample": int(start),
                    "num_samples": int(event_n),
                    "f0_hz": float(event_f0),
                })
            source_bus = _place(source_bus, event, start)

        # source-level slow tremolo / modulation
        if random.random() < float(cfg.get("source_tremolo_prob", 0.45)):
            t = torch.arange(num_samples).float() / sample_rate
            rate = random.uniform(0.05, 8.0)
            depth = random.uniform(0.05, 0.6)
            trem = 1.0 + depth * torch.sin(2 * math.pi * rate * t + random.uniform(0, 2 * math.pi))
            source_bus = source_bus * trem

        if random.random() < float(cfg.get("source_spectral_shape_prob", 1.0)):
            source_bus = _spectral_shape(source_bus, sample_rate)
        source_gain_min, source_gain_max = cfg.get("source_gain_db", [-12.0, 0.0])
        source_bus = _apply_db_gain(source_bus, random.uniform(float(source_gain_min), float(source_gain_max)))
        mix += source_bus

    # Background colored noise floor with randomized SNR
    if random.random() < float(cfg.get("background_noise_prob", 0.95)):
        noise = _colored_noise(num_samples, sample_rate)
        snr_min, snr_max = cfg.get("snr_db", [5.0, 35.0])
        snr_db = random.uniform(float(snr_min), float(snr_max))
        noise = noise * (_rms(mix) / (_rms(noise) * (10.0 ** (snr_db / 20.0))))
        mix = mix + noise

    # Global lightweight channel perturbation
    mix = _apply_channel_effects(mix, sample_rate)

    # Avoid rare silent or extremely weak samples.
    if _rms(mix) < 1e-4:
        mix = _noise_texture_event(num_samples, sample_rate)

    waveform = _normalize_audio(mix)
    if return_metadata:
        return waveform, {
            "duration": float(duration),
            "sample_rate": int(sample_rate),
            "source_event_counts": list(event_counts),
            "events": event_metadata,
        }
    return waveform

@contextmanager
def isolated_sample_rng(seed):
    """Render one synthetic sample without advancing the worker RNG stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class PhysicsDataset(torch.utils.data.Dataset):
    def __init__(self, epoch_len=5000, seed=None):  # Expanded stream length.
        self.epoch_len = epoch_len
        self.target_length = 1024
        self.seed = seed
        self._epoch = mp.Value("q", 0)

    def set_epoch(self, epoch):
        self._epoch.value = int(epoch)

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        def render_waveform():
            f0_min, f0_max = map(float, SYNTH_CONFIG.get("f0_range_hz", [50.0, 2000.0]))
            if not 0.0 < f0_min < f0_max:
                raise ValueError(f"Invalid f0_range_hz: {[f0_min, f0_max]}")

            distribution = str(SYNTH_CONFIG.get("f0_distribution", "uniform")).replace("-", "_")
            if distribution == "uniform":
                f0 = np.random.uniform(f0_min, f0_max)
            elif distribution == "log_uniform":
                f0 = np.exp(np.random.uniform(np.log(f0_min), np.log(f0_max)))
            else:
                raise ValueError(f"Unsupported f0_distribution: {distribution}")
            return advanced_physics_synth(f0)

        if self.seed is None:
            waveform = render_waveform()
        else:
            sample_seed = (
                int(self.seed) * 1_000_003
                + self._epoch.value * 10_000_019
                + int(idx)
            ) % (2**31 - 1)
            with isolated_sample_rng(sample_seed):
                waveform = render_waveform()

        # To Mel
        waveform = waveform.unsqueeze(0)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10
        )

        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[:self.target_length, :]

        fbank = (fbank - (-4.26)) / (4.56 * 2)

        return fbank.unsqueeze(0), torch.tensor(0)

# =========================================================
# Formula-driven supervised learning
# =========================================================
FORMULA_MODEL_CONFIGS = {
    "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3, "drop_path_rate": 0.0},
    "small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "drop_path_rate": 0.1},
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "drop_path_rate": 0.2},
    "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "drop_path_rate": 0.3},
}

FORMAL_SUBSET_SIZES = {
    "C56A": 56,
    "C56B": 56,
    "C56C": 56,
    "C112": 112,
    "C224": 224,
}

# These subsets are not embedded in the immutable v1.0 Formula Bank.  They are
# supplied by a frozen scale-subset registry that is cryptographically bound to
# the v1.0 manifest.  Keeping them separate preserves the provenance of all
# completed v1.0 experiments.
SCALE_SUBSET_SIZES = {
    "S14": 14,
    "S28": 28,
}

IDENTITY_REQUIRED_FIELDS = {
    "harmonic": ("partial_indices", "envelope"),
    "inharmonic": ("partial_ratios", "envelope"),
    "modal": ("mode_ratios", "relative_amplitudes", "damping", "envelope"),
    "fm": ("carrier_ratio", "modulator_ratio", "modulation_index", "envelope"),
    "chirp": ("start_ratio", "end_ratio", "curve_exponent", "envelope"),
    "pulse": ("waveform", "duty_cycle", "envelope"),
    "noise_texture": ("spectral_slope", "am_rate_hz", "am_depth", "envelope"),
}

RENDERER_ALIASES = {
    "harmonic": {"harmonic", "additive_harmonic"},
    "inharmonic": {"inharmonic", "additive_inharmonic"},
    "modal": {"modal", "modal_resonator"},
    "fm": {"fm", "two_operator_fm"},
    "chirp": {"chirp", "normalized_chirp"},
    "pulse": {"pulse", "pulse_oscillator"},
    "noise_texture": {"noise_texture", "filtered_noise_texture"},
}


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_seed(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _require_numeric_list(value, name, min_length=1):
    if not isinstance(value, list) or len(value) < min_length:
        raise ValueError(f"{name} must be a list with at least {min_length} items")
    return [_require_number(item, f"{name} item") for item in value]


def _normalize_nuisance_rules(rules):
    normalized = dict(rules)
    if "f0_hz" in normalized and "frequency_anchor_hz" in normalized:
        raise ValueError("allowed_instance_variation cannot define both f0_hz and frequency_anchor_hz")
    if "f0_hz" not in normalized and "frequency_anchor_hz" in normalized:
        normalized["f0_hz"] = normalized.pop("frequency_anchor_hz")
    required = ("f0_hz", "duration_s", "gain_db", "onset_s")
    missing = [key for key in required if key not in normalized]
    if missing:
        raise ValueError(f"allowed_instance_variation is missing required rules: {missing}")
    if "phase_rad" in normalized and "global_phase" in normalized:
        raise ValueError("allowed_instance_variation cannot define both phase_rad and global_phase")
    if "phase_rad" not in normalized and "global_phase" not in normalized:
        raise ValueError("allowed_instance_variation must define phase_rad or global_phase")
    return normalized


def _validate_distribution_rule(rule, name):
    if isinstance(rule, (int, float)) and not isinstance(rule, bool):
        _require_number(rule, name)
        return
    if not isinstance(rule, dict):
        raise ValueError(f"{name} distribution rule must be a number or object")
    distribution = str(rule.get("distribution", "uniform")).replace("-", "_")
    if distribution == "fixed":
        _require_number(rule.get("value"), f"{name}.value")
        return
    if distribution == "choice":
        _require_numeric_list(rule.get("values"), f"{name}.values")
        return
    if distribution in {"uniform_fit", "stratified_uniform_fit"}:
        if not name.endswith(".onset_s"):
            raise ValueError(f"{distribution} is only valid for onset_s")
        margin = _require_number(rule.get("edge_margin_s"), f"{name}.edge_margin_s")
        if margin < 0.0:
            raise ValueError(f"{name}.edge_margin_s must be nonnegative")
        if rule.get("condition_on", "duration_s") != "duration_s":
            raise ValueError(f"{name}.{distribution} must condition on duration_s")
        if distribution == "stratified_uniform_fit":
            strata = rule.get("num_strata")
            if isinstance(strata, bool) or not isinstance(strata, int) or strata < 2:
                raise ValueError(f"{name}.num_strata must be an integer >= 2")
            if rule.get("stratum_assignment") != "instance_id_modulo":
                raise ValueError(
                    f"{name}.stratified_uniform_fit requires stratum_assignment='instance_id_modulo'"
                )
        return
    if distribution not in {"uniform", "log_uniform"}:
        raise ValueError(f"Unsupported distribution for {name}: {distribution!r}")
    bounds = rule.get("range")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"{name} must define a two-item range")
    low = _require_number(bounds[0], f"{name}.range[0]")
    high = _require_number(bounds[1], f"{name}.range[1]")
    if low > high:
        raise ValueError(f"{name} range is reversed: {bounds}")
    if distribution == "log_uniform" and low <= 0:
        raise ValueError(f"{name} log_uniform range must be positive")


def _maximum_rule_value(rule, name):
    """Return the largest possible scalar value for a validated rule."""
    if isinstance(rule, (int, float)) and not isinstance(rule, bool):
        return _require_number(rule, name)
    distribution = str(rule.get("distribution", "uniform")).replace("-", "_")
    if distribution == "fixed":
        return _require_number(rule.get("value"), f"{name}.value")
    if distribution == "choice":
        return max(_require_numeric_list(rule.get("values"), f"{name}.values"))
    if distribution in {"uniform", "log_uniform"}:
        bounds = rule.get("range")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"{name} must define a two-item range")
        return _require_number(bounds[1], f"{name}.range[1]")
    raise ValueError(f"{name} does not have a finite scalar maximum")


def _validate_conditional_onset_rule(rules, clip_duration_s, name):
    onset_rule = rules["onset_s"]
    if not isinstance(onset_rule, dict):
        return
    distribution = str(onset_rule.get("distribution", "uniform")).replace("-", "_")
    if distribution not in {"uniform_fit", "stratified_uniform_fit"}:
        return
    margin = _require_number(onset_rule["edge_margin_s"], f"{name}.onset_s.edge_margin_s")
    maximum_duration = _maximum_rule_value(rules["duration_s"], f"{name}.duration_s")
    if maximum_duration + 2.0 * margin > clip_duration_s + 1e-9:
        raise ValueError(
            f"{name}.onset_s uniform_fit cannot fit max duration {maximum_duration} "
            f"with {margin} s edge margins in a {clip_duration_s} s clip"
        )


def _validate_identity(template):
    family = template["family"]
    if family not in IDENTITY_REQUIRED_FIELDS:
        raise ValueError(f"Unsupported formula family: {family!r}")
    identity = template["identity"]
    renderer = str(template["renderer"])
    if renderer not in RENDERER_ALIASES[family]:
        raise ValueError(
            f"{template['class_id']} renderer {renderer!r} is not valid for family {family!r}; "
            f"expected one of {sorted(RENDERER_ALIASES[family])}"
        )
    missing = [key for key in IDENTITY_REQUIRED_FIELDS[family] if key not in identity]
    if missing:
        raise ValueError(f"{template['class_id']} identity is missing {missing}")

    envelope = identity["envelope"]
    if not isinstance(envelope, dict) or "type" not in envelope:
        raise ValueError(f"{template['class_id']} identity.envelope must contain a type")
    envelope_type = str(envelope["type"]).replace("-", "_")
    if envelope_type == "flat":
        fade_ratio = _require_number(envelope.get("fade_ratio", 0.02), "envelope.fade_ratio")
        if not 0.0 <= fade_ratio < 0.5:
            raise ValueError("flat envelope fade_ratio must lie in [0, 0.5)")
    elif envelope_type == "percussive":
        attack_ratio = _require_number(envelope.get("attack_ratio", 0.01), "envelope.attack_ratio")
        release_ratio = _require_number(envelope.get("release_ratio", 0.03), "envelope.release_ratio")
        decay = _require_number(envelope.get("decay", 5.0), "envelope.decay")
        if not 0.0 <= attack_ratio < 0.5 or not 0.0 <= release_ratio < 0.5 or decay < 0.0:
            raise ValueError(
                "percussive attack/release ratios must lie in [0, 0.5) and decay must be nonnegative"
            )
    elif envelope_type == "adsr":
        attack_ratio = _require_number(envelope.get("attack_ratio"), "envelope.attack_ratio")
        decay_ratio = _require_number(envelope.get("decay_ratio"), "envelope.decay_ratio")
        release_ratio = _require_number(envelope.get("release_ratio"), "envelope.release_ratio")
        sustain_level = _require_number(envelope.get("sustain_level"), "envelope.sustain_level")
        if min(attack_ratio, decay_ratio, release_ratio) < 0.0:
            raise ValueError("ADSR ratios must be nonnegative")
        if attack_ratio + decay_ratio + release_ratio > 1.0 or not 0.0 <= sustain_level <= 1.0:
            raise ValueError("ADSR ratios must sum to at most 1 and sustain_level must lie in [0, 1]")
    else:
        raise ValueError(f"Unsupported envelope type: {envelope_type!r}")

    if family == "harmonic":
        partials = _require_numeric_list(identity["partial_indices"], "partial_indices")
        if any(int(item) != item or item <= 0 for item in partials):
            raise ValueError("harmonic partial_indices must be positive integers")
        if "relative_amplitudes" not in identity and "rolloff_exponent" not in identity:
            raise ValueError("harmonic identity needs relative_amplitudes or rolloff_exponent")
        _relative_amplitudes(identity, len(partials), partials)
    elif family == "inharmonic":
        ratios = _require_numeric_list(identity["partial_ratios"], "partial_ratios")
        if any(item <= 0 for item in ratios):
            raise ValueError("inharmonic partial_ratios must be positive")
        if "relative_amplitudes" not in identity and "rolloff_exponent" not in identity:
            raise ValueError("inharmonic identity needs relative_amplitudes or rolloff_exponent")
        _relative_amplitudes(identity, len(ratios), ratios)
        damping = _expand_per_component(identity.get("damping", 0.0), len(ratios), "damping")
        if any(value < 0.0 for value in damping):
            raise ValueError("inharmonic damping must be nonnegative")
    elif family == "modal":
        ratios = _require_numeric_list(identity["mode_ratios"], "mode_ratios")
        amplitudes = _require_numeric_list(identity["relative_amplitudes"], "relative_amplitudes")
        if len(ratios) != len(amplitudes):
            raise ValueError("modal mode_ratios and relative_amplitudes must have equal length")
        damping = _expand_per_component(identity["damping"], len(ratios), "damping")
        if any(value <= 0.0 for value in ratios) or any(value < 0.0 for value in damping):
            raise ValueError("modal ratios must be positive and damping must be nonnegative")
    elif family == "fm":
        carrier_ratio = _require_number(identity["carrier_ratio"], "carrier_ratio")
        modulator_ratio = _require_number(identity["modulator_ratio"], "modulator_ratio")
        if carrier_ratio <= 0.0 or modulator_ratio <= 0.0:
            raise ValueError("FM carrier_ratio and modulator_ratio must be positive")
        modulation_index = identity["modulation_index"]
        if isinstance(modulation_index, list):
            values = _require_numeric_list(modulation_index, "modulation_index", min_length=2)
            if len(values) != 2:
                raise ValueError("FM modulation_index trajectory must have exactly two endpoints")
        else:
            _require_number(modulation_index, "modulation_index")
    elif family == "chirp":
        start_ratio = _require_number(identity["start_ratio"], "start_ratio")
        end_ratio = _require_number(identity["end_ratio"], "end_ratio")
        curve_exponent = _require_number(identity["curve_exponent"], "curve_exponent")
        if min(start_ratio, end_ratio, curve_exponent) <= 0.0:
            raise ValueError("chirp ratios and curve_exponent must be positive")
        if identity.get("trajectory", "linear") not in {"linear", "exponential"}:
            raise ValueError("chirp trajectory must be linear or exponential")
    elif family == "pulse":
        supported_waveforms = {
            "square",
            "saw",
            "pulse_train",
            "bipolar_pulse",
            "double_pulse",
            "alternating_pulse",
        }
        if identity["waveform"] not in supported_waveforms:
            raise ValueError(
                f"pulse waveform must be one of {sorted(supported_waveforms)}"
            )
        duty = _require_number(identity["duty_cycle"], "duty_cycle")
        if not 0.0 < duty < 1.0:
            raise ValueError("pulse duty_cycle must lie in (0, 1)")
    else:
        _require_number(identity["spectral_slope"], "spectral_slope")
        am_rate = _require_number(identity["am_rate_hz"], "am_rate_hz")
        am_depth = _require_number(identity["am_depth"], "am_depth")
        if am_rate < 0.0 or not 0.0 <= am_depth <= 1.0:
            raise ValueError("noise_texture am_rate_hz must be nonnegative and am_depth must lie in [0, 1]")


def _load_scale_subset_registry(registry_path, manifest_hash, template_ids):
    """Load a frozen additive scale lineage without mutating the base Bank."""
    path = Path(registry_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scale subset registry not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Scale subset registry schema_version must equal 1")
    if payload.get("status") != "frozen":
        raise ValueError("Scale subset registry status must equal 'frozen'")
    if payload.get("base_manifest_sha256") != manifest_hash:
        raise ValueError("Scale subset registry is not bound to this Formula Bank hash")
    subsets = payload.get("subsets")
    if not isinstance(subsets, dict) or set(subsets) != set(SCALE_SUBSET_SIZES):
        raise ValueError(
            f"Scale subset registry must define exactly {sorted(SCALE_SUBSET_SIZES)}"
        )

    template_id_set = set(template_ids)
    memberships_by_class = {}
    for subset_name, expected_size in SCALE_SUBSET_SIZES.items():
        members = subsets[subset_name]
        if not isinstance(members, list) or len(members) != expected_size:
            raise ValueError(
                f"Scale subset {subset_name} must contain exactly {expected_size} class IDs"
            )
        if any(not isinstance(class_id, str) or not class_id for class_id in members):
            raise ValueError(f"Scale subset {subset_name} contains an invalid class ID")
        if len(set(members)) != len(members):
            raise ValueError(f"Scale subset {subset_name} contains duplicate class IDs")
        unknown = set(members) - template_id_set
        if unknown:
            raise ValueError(
                f"Scale subset {subset_name} contains unknown class IDs: {sorted(unknown)}"
            )
        for class_id in members:
            memberships_by_class.setdefault(class_id, set()).add(subset_name)

    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
        "scale_id": str(payload.get("scale_id", "")).strip(),
        "payload": payload,
        "memberships_by_class": memberships_by_class,
    }


class FormulaBank:
    def __init__(self, manifest_path, subset, subset_registry_path=""):
        self.path = Path(manifest_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Formula Bank manifest not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as handle:
            self.payload = json.load(handle)
        if self.payload.get("schema_version") != 1:
            raise ValueError("Formula Bank schema_version must equal 1")
        self.bank_id = str(self.payload.get("bank_id", "")).strip()
        if not self.bank_id:
            raise ValueError("Formula Bank must define a non-empty bank_id")
        self.sample_rate = int(self.payload.get("sample_rate", 16000))
        self.clip_duration_s = _require_number(
            self.payload.get("clip_duration_s", 10.24), "clip_duration_s"
        )
        if self.sample_rate != 16000 or abs(self.clip_duration_s - 10.24) > 1e-9:
            raise ValueError("Formal frontend requires sample_rate=16000 and clip_duration_s=10.24")
        templates = self.payload.get("templates")
        if not isinstance(templates, list) or not templates:
            raise ValueError("Formula Bank must contain a non-empty templates list")

        self.bank_hash = hashlib.sha256(_canonical_json(self.payload).encode("utf-8")).hexdigest()
        template_ids = [str(template.get("class_id", "")).strip() for template in templates if isinstance(template, dict)]
        self.subset_registry = None
        self.formal_subset_sizes = dict(FORMAL_SUBSET_SIZES)
        registry_memberships_by_class = {}
        if subset_registry_path:
            self.subset_registry = _load_scale_subset_registry(
                subset_registry_path,
                self.bank_hash,
                template_ids,
            )
            registry_memberships_by_class = self.subset_registry["memberships_by_class"]
            self.formal_subset_sizes.update(SCALE_SUBSET_SIZES)

        selected = []
        seen_ids = set()
        seen_indices = set()
        nuisance_signature = None
        phase_signatures = {}
        subset_memberships = {name: [] for name in self.formal_subset_sizes}
        for raw_template in templates:
            if not isinstance(raw_template, dict):
                raise ValueError("Every template must be an object")
            required = (
                "class_id", "global_index", "family", "renderer", "identity",
                "allowed_instance_variation", "validity_constraints", "subsets",
            )
            missing = [key for key in required if key not in raw_template]
            if missing:
                raise ValueError(f"Template is missing required fields: {missing}")
            class_id = str(raw_template["class_id"]).strip()
            global_index = int(raw_template["global_index"])
            if not class_id or class_id in seen_ids:
                raise ValueError(f"Duplicate or empty class_id: {class_id!r}")
            if global_index in seen_indices:
                raise ValueError(f"Duplicate global_index: {global_index}")
            seen_ids.add(class_id)
            seen_indices.add(global_index)
            if not isinstance(raw_template["identity"], dict) or not raw_template["identity"]:
                raise ValueError(f"{class_id} must define a non-empty identity")
            if not isinstance(raw_template["validity_constraints"], dict) or not raw_template["validity_constraints"]:
                raise ValueError(f"{class_id} must define non-empty validity_constraints")
            if not isinstance(raw_template["subsets"], list) or any(
                not isinstance(name, str) or not name for name in raw_template["subsets"]
            ):
                raise ValueError(f"{class_id}.subsets must be a list")
            if len(set(raw_template["subsets"])) != len(raw_template["subsets"]):
                raise ValueError(f"{class_id}.subsets contains duplicate names")
            effective_subsets = list(raw_template["subsets"])
            effective_subsets.extend(sorted(registry_memberships_by_class.get(class_id, set())))
            if len(set(effective_subsets)) != len(effective_subsets):
                raise ValueError(f"{class_id} has duplicate effective subset memberships")
            for subset_name in effective_subsets:
                if subset_name in subset_memberships:
                    subset_memberships[subset_name].append(raw_template)
            _validate_identity(raw_template)
            normalized_rules = _normalize_nuisance_rules(raw_template["allowed_instance_variation"])
            for rule_name, rule in normalized_rules.items():
                _validate_distribution_rule(rule, f"{class_id}.{rule_name}")
            _validate_conditional_onset_rule(
                normalized_rules, self.clip_duration_s, class_id
            )
            if subset in effective_subsets:
                shared_rules = {
                    key: normalized_rules[key]
                    for key in ("f0_hz", "duration_s", "gain_db", "onset_s")
                }
                signature = _canonical_json(shared_rules)
                if nuisance_signature is None:
                    nuisance_signature = signature
                elif nuisance_signature != signature:
                    raise ValueError(
                        f"Selected templates do not share matched f0/duration/gain/onset rules; failed at {class_id}"
                    )
                phase_rule = normalized_rules.get("phase_rad", normalized_rules.get("global_phase"))
                phase_signature = _canonical_json(phase_rule)
                family = raw_template["family"]
                previous_phase_signature = phase_signatures.setdefault(family, phase_signature)
                if previous_phase_signature != phase_signature:
                    raise ValueError(
                        f"Selected {family} templates do not share a matched phase rule; failed at {class_id}"
                    )
                template = dict(raw_template)
                template["subsets"] = effective_subsets
                template["allowed_instance_variation"] = normalized_rules
                selected.append(template)

        selected.sort(key=lambda item: item["global_index"])
        if not selected:
            raise ValueError(f"No templates belong to subset {subset!r}")
        self.bank_version = str(self.payload.get("bank_version", "")).strip()
        self.generator_commit = str(self.payload.get("generator_commit", "")).strip()
        self.generator_config_hash = str(self.payload.get("generator_config_hash", "")).strip()
        self.admission_report = self.payload.get("admission_report", {})
        if subset in self.formal_subset_sizes:
            for field_name, field_value in (
                ("bank_version", self.bank_version),
                ("generator_commit", self.generator_commit),
                ("generator_config_hash", self.generator_config_hash),
            ):
                if not field_value:
                    raise ValueError(f"Formal Formula Bank must define non-empty {field_name}")
            if not isinstance(self.admission_report, dict):
                raise ValueError("Formal Formula Bank admission_report must be an object")
            if self.admission_report.get("status") != "passed":
                raise ValueError("Formal Formula Bank admission_report.status must equal 'passed'")
            if not str(self.admission_report.get("report_hash", "")).strip():
                raise ValueError("Formal Formula Bank admission_report must define report_hash")
            for subset_name, expected_size in self.formal_subset_sizes.items():
                members = subset_memberships[subset_name]
                if len(members) != expected_size:
                    raise ValueError(
                        f"Formal manifest subset {subset_name} must contain {expected_size} templates, got {len(members)}"
                    )
                expected_per_family = expected_size // len(IDENTITY_REQUIRED_FIELDS)
                family_counts = {
                    family: sum(member["family"] == family for member in members)
                    for family in IDENTITY_REQUIRED_FIELDS
                }
                if any(count != expected_per_family for count in family_counts.values()):
                    raise ValueError(
                        f"Formal subset {subset_name} is not family-balanced: {family_counts}"
                    )
            membership_ids = {
                name: {member["class_id"] for member in members}
                for name, members in subset_memberships.items()
            }
            if not membership_ids["C56A"] < membership_ids["C112"] < membership_ids["C224"]:
                raise ValueError("Formal subsets must satisfy strict nesting C56A < C112 < C224")
            if self.subset_registry is not None and not (
                membership_ids["S14"] < membership_ids["S28"]
                < membership_ids["C56A"] < membership_ids["C112"] < membership_ids["C224"]
            ):
                raise ValueError("Scale subsets must satisfy S14 < S28 < C56A < C112 < C224")
            if not membership_ids["C56B"].issubset(membership_ids["C224"]):
                raise ValueError("C56B must be a subset of C224")
            if not membership_ids["C56C"].issubset(membership_ids["C224"]):
                raise ValueError("C56C must be a subset of C224")
            c224_indices = sorted(member["global_index"] for member in subset_memberships["C224"])
            if c224_indices != list(range(self.formal_subset_sizes["C224"])):
                raise ValueError("C224 global_index values must be exactly 0..223")
        self.subset = subset
        self.is_formal_subset = subset in self.formal_subset_sizes
        self.templates = selected
        self.label_mapping = {
            template["class_id"]: label for label, template in enumerate(self.templates)
        }

    @property
    def num_classes(self):
        return len(self.templates)


def _sample_rule(rule, name):
    if isinstance(rule, (int, float)) and not isinstance(rule, bool):
        return _require_number(rule, name)
    if not isinstance(rule, dict):
        raise ValueError(f"{name} distribution rule must be a number or object")
    distribution = str(rule.get("distribution", "uniform")).replace("-", "_")
    if distribution == "fixed":
        return _require_number(rule["value"], f"{name}.value")
    if distribution == "choice":
        values = _require_numeric_list(rule.get("values"), f"{name}.values")
        return float(random.choice(values))
    bounds = rule.get("range")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"{name} must define a two-item range")
    low = _require_number(bounds[0], f"{name}.range[0]")
    high = _require_number(bounds[1], f"{name}.range[1]")
    if not low <= high:
        raise ValueError(f"{name} range is reversed: {bounds}")
    if distribution == "uniform":
        return random.uniform(low, high)
    if distribution == "log_uniform":
        if low <= 0:
            raise ValueError(f"{name} log_uniform range must be positive")
        return math.exp(random.uniform(math.log(low), math.log(high)))
    raise ValueError(f"Unsupported distribution for {name}: {distribution!r}")


def _render_identity_envelope(n, spec):
    kind = str(spec.get("type", "")).replace("-", "_")
    u = torch.linspace(0.0, 1.0, n)
    if kind == "flat":
        fade_ratio = _require_number(spec.get("fade_ratio", 0.02), "envelope.fade_ratio")
        fade_ratio = min(max(fade_ratio, 0.0), 0.49)
        envelope = torch.ones(n)
        fade_n = max(1, int(round(n * fade_ratio)))
        envelope[:fade_n] = torch.linspace(0.0, 1.0, fade_n)
        envelope[-fade_n:] = torch.linspace(1.0, 0.0, fade_n)
        return envelope
    if kind == "cosine":
        # Used by the property-ablation renderer only: a shared differentiable
        # attack/release shape that removes template-specific onset edges.
        fade_ratio = _require_number(spec.get("fade_ratio", 0.02), "envelope.fade_ratio")
        fade_ratio = min(max(fade_ratio, 0.0), 0.49)
        envelope = torch.ones(n)
        fade_n = max(1, int(round(n * fade_ratio)))
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0.0, math.pi, fade_n))
        envelope[:fade_n] = ramp
        envelope[-fade_n:] = torch.flip(ramp, dims=[0])
        return envelope
    if kind == "percussive":
        attack_ratio = _require_number(spec.get("attack_ratio", 0.01), "envelope.attack_ratio")
        release_ratio = _require_number(spec.get("release_ratio", 0.03), "envelope.release_ratio")
        decay = _require_number(spec.get("decay", 5.0), "envelope.decay")
        attack_n = max(1, min(n, int(round(n * attack_ratio))))
        release_n = max(1, min(n, int(round(n * release_ratio))))
        envelope = torch.exp(-decay * u)
        envelope[:attack_n] *= torch.linspace(0.0, 1.0, attack_n)
        envelope[-release_n:] *= torch.linspace(1.0, 0.0, release_n)
        return envelope
    if kind == "adsr":
        attack_ratio = _require_number(spec.get("attack_ratio"), "envelope.attack_ratio")
        decay_ratio = _require_number(spec.get("decay_ratio"), "envelope.decay_ratio")
        release_ratio = _require_number(spec.get("release_ratio"), "envelope.release_ratio")
        sustain = _require_number(spec.get("sustain_level"), "envelope.sustain_level")
        if min(attack_ratio, decay_ratio, release_ratio) < 0 or attack_ratio + decay_ratio + release_ratio > 1:
            raise ValueError("ADSR ratios must be nonnegative and sum to at most 1")
        if not 0 <= sustain <= 1:
            raise ValueError("ADSR sustain_level must lie in [0, 1]")
        attack_n = max(1, int(round(n * attack_ratio)))
        decay_n = max(1, int(round(n * decay_ratio)))
        release_n = max(1, int(round(n * release_ratio)))
        sustain_n = max(0, n - attack_n - decay_n - release_n)
        envelope = torch.cat([
            torch.linspace(0.0, 1.0, attack_n),
            torch.linspace(1.0, sustain, decay_n),
            torch.full((sustain_n,), sustain),
            torch.linspace(sustain, 0.0, release_n),
        ])
        return envelope[:n] if len(envelope) >= n else torch.nn.functional.pad(envelope, (0, n - len(envelope)))
    raise ValueError(f"Unsupported envelope type: {kind!r}")


def _relative_amplitudes(identity, count, spectral_positions=None):
    if "relative_amplitudes" in identity:
        amplitudes = _require_numeric_list(identity["relative_amplitudes"], "relative_amplitudes")
        if len(amplitudes) != count:
            raise ValueError("relative_amplitudes length must match the partial/mode count")
        return amplitudes
    exponent = _require_number(identity["rolloff_exponent"], "rolloff_exponent")
    if spectral_positions is None:
        spectral_positions = list(range(1, count + 1))
    if len(spectral_positions) != count or any(position <= 0 for position in spectral_positions):
        raise ValueError("spectral_positions must be positive and match the component count")
    return [1.0 / (float(position) ** exponent) for position in spectral_positions]


def _relative_phases(identity, count):
    if bool(identity.get("relative_phase_is_identity", False)):
        phases = _require_numeric_list(identity.get("relative_phases"), "relative_phases")
        if len(phases) != count:
            raise ValueError("relative_phases length must match the oscillator count")
        return phases
    return [random.uniform(0.0, 2.0 * math.pi) for _ in range(count)]


def _expand_per_component(value, count, name):
    if isinstance(value, list):
        values = _require_numeric_list(value, name)
        if len(values) != count:
            raise ValueError(f"{name} length must equal {count}")
        return values
    scalar = _require_number(value, name)
    return [scalar] * count


def _bandlimited_rectangular_series(phase_cycles, base_frequency, nyquist_guard, components):
    """Render periodic rectangular components without sampling a discontinuity.

    Each component is ``(offset_cycles, width_cycles, amplitude)``.  The DC
    term is intentionally omitted because the shared waveform normalization
    removes it.  A 32-partial cap keeps online rendering bounded while the
    Nyquist guard prevents aliased harmonics.
    """
    max_harmonic = min(32, int(nyquist_guard // max(base_frequency, 1e-8)))
    if max_harmonic < 1:
        raise ValueError("pulse base frequency leaves no partial below the Nyquist guard")
    event = torch.zeros_like(phase_cycles)
    for harmonic in range(1, max_harmonic + 1):
        harmonic_value = float(harmonic)
        for offset, width, amplitude in components:
            coefficient = (
                2.0
                * amplitude
                * math.sin(math.pi * harmonic_value * width)
                / (math.pi * harmonic_value)
            )
            center = offset + width / 2.0
            event += coefficient * torch.cos(
                2.0 * math.pi * harmonic_value * (phase_cycles - center)
            )
    return event


def _bandlimited_saw_series(phase_cycles, base_frequency, nyquist_guard):
    max_harmonic = min(32, int(nyquist_guard // max(base_frequency, 1e-8)))
    if max_harmonic < 1:
        raise ValueError("saw base frequency leaves no partial below the Nyquist guard")
    event = torch.zeros_like(phase_cycles)
    for harmonic in range(1, max_harmonic + 1):
        event -= torch.sin(2.0 * math.pi * harmonic * phase_cycles) / harmonic
    return event


def _render_formula_event(template, n, sample_rate, anchor_hz, phase_rad):
    family = template["family"]
    identity = template["identity"]
    t = torch.arange(n, dtype=torch.float32) / sample_rate
    u = torch.linspace(0.0, 1.0, n)
    nyquist_guard = sample_rate * 0.45

    if family == "harmonic":
        partials = [int(item) for item in identity["partial_indices"]]
        amplitudes = _relative_amplitudes(identity, len(partials), partials)
        phases = _relative_phases(identity, len(partials))
        if anchor_hz * max(partials) > nyquist_guard:
            raise ValueError("harmonic identity exceeds the Nyquist guard")
        event = torch.zeros(n)
        for partial, amplitude, offset in zip(partials, amplitudes, phases):
            event += amplitude * torch.sin(2 * math.pi * anchor_hz * partial * t + phase_rad + offset)
    elif family == "inharmonic":
        ratios = _require_numeric_list(identity["partial_ratios"], "partial_ratios")
        amplitudes = _relative_amplitudes(identity, len(ratios), ratios)
        phases = _relative_phases(identity, len(ratios))
        damping = _expand_per_component(identity.get("damping", 0.0), len(ratios), "damping")
        if anchor_hz * max(ratios) > nyquist_guard:
            raise ValueError("inharmonic identity exceeds the Nyquist guard")
        event = torch.zeros(n)
        for ratio, amplitude, offset, decay in zip(ratios, amplitudes, phases, damping):
            event += amplitude * torch.exp(-decay * u) * torch.sin(
                2 * math.pi * anchor_hz * ratio * t + phase_rad + offset
            )
    elif family == "modal":
        ratios = _require_numeric_list(identity["mode_ratios"], "mode_ratios")
        amplitudes = _require_numeric_list(identity["relative_amplitudes"], "relative_amplitudes")
        damping = _expand_per_component(identity["damping"], len(ratios), "damping")
        phases = _relative_phases(identity, len(ratios))
        if anchor_hz * max(ratios) > nyquist_guard:
            raise ValueError("modal identity exceeds the Nyquist guard")
        event = torch.zeros(n)
        for ratio, amplitude, decay, offset in zip(ratios, amplitudes, damping, phases):
            event += amplitude * torch.exp(-decay * u) * torch.sin(
                2 * math.pi * anchor_hz * ratio * t + phase_rad + offset
            )
    elif family == "fm":
        carrier = anchor_hz * _require_number(identity["carrier_ratio"], "carrier_ratio")
        modulator = carrier * _require_number(identity["modulator_ratio"], "modulator_ratio")
        index_value = identity["modulation_index"]
        if isinstance(index_value, list):
            endpoints = _require_numeric_list(index_value, "modulation_index", min_length=2)
            if len(endpoints) != 2:
                raise ValueError("modulation_index trajectory must contain exactly two endpoints")
            index_env = torch.linspace(endpoints[0], endpoints[1], n)
            max_index = max(abs(endpoints[0]), abs(endpoints[1]))
        else:
            scalar_index = _require_number(index_value, "modulation_index")
            max_index = abs(scalar_index)
            index_env = torch.full((n,), scalar_index)
        if carrier + max_index * modulator > nyquist_guard:
            raise ValueError("FM sideband estimate exceeds the Nyquist guard")
        mod_phase = _require_number(identity.get("modulator_phase", 0.0), "modulator_phase")
        modulation = index_env * torch.sin(2 * math.pi * modulator * t + mod_phase)
        event = torch.sin(2 * math.pi * carrier * t + phase_rad + modulation)
    elif family == "chirp":
        start = anchor_hz * _require_number(identity["start_ratio"], "start_ratio")
        end = anchor_hz * _require_number(identity["end_ratio"], "end_ratio")
        if max(start, end) > nyquist_guard or min(start, end) < 20.0:
            raise ValueError("chirp trajectory lies outside the valid frequency range")
        curve = _require_number(identity["curve_exponent"], "curve_exponent")
        shaped_u = u ** curve
        trajectory = str(identity.get("trajectory", "linear"))
        if trajectory == "linear":
            frequency = start * (1.0 - shaped_u) + end * shaped_u
        elif trajectory == "exponential":
            frequency = start * torch.pow(torch.full_like(shaped_u, end / start), shaped_u)
        else:
            raise ValueError(f"Unsupported chirp trajectory: {trajectory!r}")
        instantaneous_phase = 2 * math.pi * torch.cumsum(frequency, dim=0) / sample_rate
        event = torch.sin(instantaneous_phase + phase_rad)
    elif family == "pulse":
        frequency_ratio = _require_number(identity.get("frequency_ratio", 1.0), "frequency_ratio")
        frequency = anchor_hz * frequency_ratio
        if not 20.0 <= frequency <= sample_rate * 0.40:
            raise ValueError("pulse frequency lies outside the valid range")
        duty = _require_number(identity["duty_cycle"], "duty_cycle")
        waveform = identity["waveform"]
        if waveform == "alternating_pulse":
            base_frequency = frequency / 2.0
            phase_cycles = base_frequency * t + phase_rad / (2 * math.pi)
            components = [(0.0, duty / 2.0, 1.0), (0.5, duty / 2.0, -1.0)]
        elif waveform == "saw":
            base_frequency = frequency
            phase_cycles = base_frequency * t + phase_rad / (2 * math.pi)
            event = _bandlimited_saw_series(phase_cycles, base_frequency, nyquist_guard)
            components = None
        elif waveform == "pulse_train":
            base_frequency = frequency
            phase_cycles = base_frequency * t + phase_rad / (2 * math.pi)
            components = [(0.0, duty, 1.0)]
        elif waveform == "double_pulse":
            base_frequency = frequency
            phase_cycles = base_frequency * t + phase_rad / (2 * math.pi)
            components = [(0.0, duty / 2.0, 1.0), (0.5, duty / 2.0, 1.0)]
        else:
            base_frequency = frequency
            phase_cycles = base_frequency * t + phase_rad / (2 * math.pi)
            if waveform == "square":
                components = [(0.0, duty, 1.0)]
            else:
                components = [(0.0, duty / 2.0, 1.0), (0.5, duty / 2.0, -1.0)]
        if components is not None:
            event = _bandlimited_rectangular_series(
                phase_cycles, base_frequency, nyquist_guard, components
            )
    else:
        spectral_slope = _require_number(identity["spectral_slope"], "spectral_slope")
        white = torch.randn(n)
        spectrum = torch.fft.rfft(white)
        frequencies = torch.fft.rfftfreq(n, d=1.0 / sample_rate)
        reference = max(anchor_hz, 20.0)
        relative_frequency = torch.clamp(frequencies / reference, min=0.05)
        spectral_gain = torch.pow(relative_frequency, -0.5 * spectral_slope)
        spectral_gain[0] = 0.0
        if "bandpass_ratio" in identity:
            band = _require_numeric_list(identity["bandpass_ratio"], "bandpass_ratio", min_length=2)
            if len(band) != 2 or not 0 < band[0] < band[1]:
                raise ValueError("bandpass_ratio must be [positive_low, greater_high]")
            mask = (frequencies >= reference * band[0]) & (frequencies <= reference * band[1])
            spectral_gain = spectral_gain * mask
        event = torch.fft.irfft(spectrum * spectral_gain, n=n)
        am_rate = _require_number(identity["am_rate_hz"], "am_rate_hz")
        am_depth = _require_number(identity["am_depth"], "am_depth")
        if not 0.0 <= am_depth <= 1.0:
            raise ValueError("noise_texture am_depth must lie in [0, 1]")
        event *= 1.0 + am_depth * torch.sin(2 * math.pi * am_rate * t + phase_rad)

    event = _normalize_audio(event)
    event *= _render_identity_envelope(n, identity["envelope"])
    return event


def _sample_onset(rule, duration_s, clip_duration_s, instance_id=None):
    if not isinstance(rule, dict):
        return _sample_rule(rule, "onset_s")
    distribution = str(rule.get("distribution", "uniform")).replace("-", "_")
    if distribution not in {"uniform_fit", "stratified_uniform_fit"}:
        return _sample_rule(rule, "onset_s")
    margin = _require_number(rule["edge_margin_s"], "onset_s.edge_margin_s")
    maximum_onset = clip_duration_s - duration_s - margin
    if maximum_onset < margin - 1e-9:
        raise ValueError("duration_s leaves no legal onset window")
    if distribution == "uniform_fit":
        return random.uniform(margin, maximum_onset), None

    num_strata = int(rule["num_strata"])
    if instance_id is None:
        stratum = random.randrange(num_strata)
    else:
        stratum = int(instance_id) % num_strata
    q = (stratum + random.random()) / num_strata
    return margin + q * (maximum_onset - margin), stratum


def _sample_instance_params(template, clip_duration_s=10.24, instance_id=None):
    rules = template["allowed_instance_variation"]
    phase_rule = rules.get("phase_rad", rules.get("global_phase"))
    duration_s = _sample_rule(rules["duration_s"], "duration_s")
    onset_result = _sample_onset(
        rules["onset_s"], duration_s, clip_duration_s, instance_id=instance_id
    )
    if isinstance(onset_result, tuple):
        onset_s, onset_stratum = onset_result
    else:
        onset_s, onset_stratum = onset_result, None
    params = {
        "anchor_hz": _sample_rule(rules["f0_hz"], "f0_hz"),
        "duration_s": duration_s,
        "gain_db": _sample_rule(rules["gain_db"], "gain_db"),
        "onset_s": onset_s,
        "phase_rad": _sample_rule(phase_rule, "phase_rad"),
    }
    if onset_stratum is not None:
        params["onset_stratum"] = onset_stratum
    return params


def _sample_matched_instance_params(
    template, bank_hash, split, instance_id, seed, clip_duration_s=10.24
):
    """Sample the frozen nuisance tuple shared by a cross-class instance ID."""
    nuisance_seed = _stable_seed(
        bank_hash,
        split,
        "matched_instance_nuisance",
        int(instance_id),
        int(seed),
    )
    with isolated_sample_rng(nuisance_seed):
        return _sample_instance_params(
            template, clip_duration_s=clip_duration_s, instance_id=instance_id
        )


def _render_formula_waveform(
    template,
    sample_rate,
    clip_duration_s,
    sample_seed,
    max_attempts,
    return_metadata=False,
    fixed_instance_params=None,
):
    constraints = template["validity_constraints"]
    min_active_rms = _require_number(constraints.get("min_active_rms", 1e-4), "min_active_rms")
    max_abs = _require_number(constraints.get("max_abs", 1.0), "max_abs")
    clip_samples = int(round(sample_rate * clip_duration_s))
    failures = []
    for attempt in range(max_attempts):
        attempt_seed = _stable_seed(sample_seed, "attempt", attempt)
        with isolated_sample_rng(attempt_seed):
            try:
                params = (
                    dict(fixed_instance_params)
                    if fixed_instance_params is not None
                    else _sample_instance_params(template, clip_duration_s=clip_duration_s)
                )
                duration_s = params["duration_s"]
                onset_s = params["onset_s"]
                if duration_s <= 0 or onset_s < 0 or onset_s + duration_s > clip_duration_s:
                    raise ValueError("event does not fit completely inside the clip")
                event_samples = int(round(duration_s * sample_rate))
                if event_samples < 16:
                    raise ValueError("event is shorter than 16 samples")
                event = _render_formula_event(
                    template, event_samples, sample_rate, params["anchor_hz"], params["phase_rad"]
                )
                event = _apply_db_gain(event, params["gain_db"])
                if not torch.isfinite(event).all():
                    raise ValueError("event contains NaN or Inf")
                active_rms = float(_rms(event).item())
                if active_rms < min_active_rms:
                    raise ValueError(f"active RMS {active_rms:.3e} is below {min_active_rms:.3e}")
                peak = float(event.abs().max().item())
                if peak > max_abs:
                    raise ValueError(f"peak {peak:.4f} exceeds max_abs {max_abs:.4f}")
                waveform = torch.zeros(clip_samples)
                start = int(round(onset_s * sample_rate))
                end = start + event_samples
                if end > clip_samples:
                    raise ValueError("rounded event boundary exceeds the clip")
                waveform[start:end] = event
                if return_metadata:
                    return waveform, attempt, {
                        **params,
                        "event_samples": event_samples,
                        "active_rms": active_rms,
                        "peak_abs": peak,
                        "accepted_attempt": attempt,
                    }
                return waveform, attempt
            except (KeyError, TypeError, ValueError) as error:
                failures.append(str(error))
    detail = failures[-1] if failures else "unknown validity failure"
    raise RuntimeError(
        f"Unable to render valid instance for {template['class_id']} after {max_attempts} attempts: {detail}"
    )


def _waveform_to_fbank(waveform, sample_rate=16000, target_length=1024):
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )
    frame_delta = target_length - fbank.shape[0]
    if frame_delta > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, frame_delta))
    elif frame_delta < 0:
        fbank = fbank[:target_length]
    fbank = (fbank - (-4.26)) / (4.56 * 2)
    return fbank.unsqueeze(0)


class FormulaExposureDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        bank,
        split,
        seed,
        instances_per_class,
        exposure_count=None,
        max_render_attempts=16,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split!r}")
        if instances_per_class <= 0:
            raise ValueError("instances_per_class must be positive")
        self.bank = bank
        self.split = split
        self.seed = int(seed)
        self.instances_per_class = int(instances_per_class)
        self.max_render_attempts = int(max_render_attempts)
        self._epoch = mp.Value("q", 0)
        self._permutation_cache = collections.OrderedDict()
        self._permutation_cache_limit = max(64, bank.num_classes * 4)
        if split == "train":
            if exposure_count is None or exposure_count <= 0:
                raise ValueError("train exposure_count must be positive")
            self.exposure_count = int(exposure_count)
        else:
            self.exposure_count = bank.num_classes * self.instances_per_class
        self.pool_hash = hashlib.sha256(
            _canonical_json({
                "bank_hash": bank.bank_hash,
                "subset": bank.subset,
                "split": split,
                "seed": self.seed,
                "instances_per_class": self.instances_per_class,
            }).encode("utf-8")
        ).hexdigest()

    def set_epoch(self, epoch):
        self._epoch.value = int(epoch)

    def __len__(self):
        return self.exposure_count

    def _permuted_instance_id(self, label, occurrence):
        cycle, position = divmod(occurrence, self.instances_per_class)
        key = (int(label), int(cycle))
        permutation = self._permutation_cache.get(key)
        if permutation is None:
            rng = np.random.RandomState(
                _stable_seed(self.bank.bank_hash, self.split, self.seed, label, cycle)
            )
            permutation = rng.permutation(self.instances_per_class)
            self._permutation_cache[key] = permutation
            if len(self._permutation_cache) > self._permutation_cache_limit:
                self._permutation_cache.popitem(last=False)
        else:
            self._permutation_cache.move_to_end(key)
        return int(permutation[position])

    def __getitem__(self, idx):
        idx = int(idx)
        if self.split == "train":
            global_exposure = self._epoch.value * self.exposure_count + idx
            label = global_exposure % self.bank.num_classes
            occurrence = global_exposure // self.bank.num_classes
            instance_id = self._permuted_instance_id(label, occurrence)
        else:
            label, instance_id = divmod(idx, self.instances_per_class)
        template = self.bank.templates[label]
        sample_seed = _stable_seed(
            self.bank.bank_hash,
            self.split,
            template["class_id"],
            instance_id,
            self.seed,
        )
        fixed_instance_params = _sample_matched_instance_params(
            template,
            self.bank.bank_hash,
            self.split,
            instance_id,
            self.seed,
            clip_duration_s=self.bank.clip_duration_s,
        )
        waveform, rejected_attempts = _render_formula_waveform(
            template,
            self.bank.sample_rate,
            self.bank.clip_duration_s,
            sample_seed,
            self.max_render_attempts,
            fixed_instance_params=fixed_instance_params,
        )
        fbank = _waveform_to_fbank(waveform, sample_rate=self.bank.sample_rate)
        return fbank, torch.tensor(label, dtype=torch.long), torch.tensor(rejected_attempts, dtype=torch.long)


class FormulaViTClassifier(nn.Module):
    def __init__(self, num_classes, variant="small", drop_path_rate=None):
        super().__init__()
        if variant not in FORMULA_MODEL_CONFIGS:
            raise ValueError(f"Unsupported model variant: {variant!r}")
        config = FORMULA_MODEL_CONFIGS[variant]
        embed_dim = config["embed_dim"]
        depth = config["depth"]
        num_heads = config["num_heads"]
        if drop_path_rate is None:
            drop_path_rate = config["drop_path_rate"]
        self.variant = variant
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed_new(
            img_size=(1024, 128), patch_size=16, stride=16, in_chans=1, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        drop_path_values = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                drop_path=drop_path_values[index],
            )
            for index in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        position = get_2d_sincos_pos_embed_flexible(
            self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(position).float().unsqueeze(0))
        torch.nn.init.xavier_uniform_(self.patch_embed.proj.weight.view(self.patch_embed.proj.weight.shape[0], -1))
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 1:, :].mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed(force_cpu=False):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        use_cuda = torch.cuda.is_available() and not force_cpu
        return 0, 1, torch.device("cuda" if use_cuda else "cpu")
    if force_cpu:
        raise RuntimeError("--cpu cannot be combined with distributed training")
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed supervised pre-training requires CUDA/NCCL")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_distributed(world_size):
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(world_size, device=None):
    if world_size > 1:
        device_ids = None
        if device is not None and device.type == "cuda" and device.index is not None:
            device_ids = [device.index]
        dist.barrier(device_ids=device_ids)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def resolve_batch_size(variant, args, world_size):
    global_batch_size = args.global_batch_size
    if global_batch_size <= 0:
        per_rank = DEFAULT_BATCH_SIZE[variant] if args.batch_size <= 0 else args.batch_size
        global_batch_size = per_rank * world_size
    if global_batch_size % world_size != 0:
        raise ValueError("global batch size must be divisible by WORLD_SIZE")
    return global_batch_size // world_size, global_batch_size


def _optimizer_groups(model, weight_decay):
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or name.endswith("cls_token"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _learning_rate_at_step(step, total_steps, warmup_steps, peak_lr, warmup_lr, min_lr):
    if warmup_steps > 0 and step < warmup_steps:
        progress = (step + 1) / warmup_steps
        return warmup_lr + progress * (peak_lr - warmup_lr)
    cosine_steps = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, (step - warmup_steps) / cosine_steps))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def _validate_formal_protocol(
    bank,
    variant,
    args,
    device,
    world_size,
    global_batch,
    steps_per_epoch,
    total_steps,
    total_exposures,
    peak_lr,
    warmup_lr,
    min_lr,
):
    if not bank.is_formal_subset:
        return "non-formal"
    if args.dev_run:
        return f"dev-{bank.subset}-I{args.instances_per_class}"
    config_ids = {
        ("C56A", 500): "CB-56A",
        ("C112", 250): "CB-112",
        ("C224", 125): "CB-224",
        ("C56A", 125): "SI-56",
        ("C224", 500): "SI-224",
        ("C56B", 500): "SB-56B",
        ("C56C", 500): "SB-56C",
        ("S14", 125): "CS-C14-I125",
        ("S28", 125): "CS-C28-I125",
        ("C112", 125): "CS-C112-I125",
        ("C224", 16): "IS-C224-I16",
        ("C224", 32): "IS-C224-I32",
        ("C224", 64): "IS-C224-I64",
        ("C224", 250): "IS-C224-I250",
        ("C224", 1000): "IS-C224-I1000",
    }
    config_id = config_ids.get((bank.subset, args.instances_per_class))
    if config_id is None:
        raise ValueError(
            f"Unsupported formal class/instance allocation: {bank.subset} x {args.instances_per_class}"
        )
    expected_exact = {
        "variant": (variant, "small"),
        "device": (device.type, "cuda"),
        "world_size": (world_size, 1),
        "epochs": (args.epochs, 500),
        "epoch_len": (args.epoch_len, 5000),
        "global_batch_size": (global_batch, 32),
        "steps_per_epoch": (steps_per_epoch, 156),
        "total_steps": (total_steps, 78000),
        "total_exposures": (total_exposures, 2496000),
        "val_instances_per_class": (args.val_instances_per_class, 50),
        "warmup_epochs": (args.warmup_epochs, 5),
        "seed": (args.seed, 2026),
        "amp": (args.amp, True),
    }
    mismatches = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in expected_exact.items()
        if actual != expected
    ]
    expected_float = {
        "peak_lr": (peak_lr, 1.25e-4),
        "warmup_lr": (warmup_lr, 1.25e-5),
        "min_lr": (min_lr, 1.25e-6),
        "weight_decay": (args.weight_decay, 0.05),
        "label_smoothing": (args.label_smoothing, 0.1),
        "drop_path": (args.drop_path, 0.1),
        "mixup_alpha": (args.mixup_alpha, 0.0),
        "cutmix_alpha": (args.cutmix_alpha, 0.0),
        "grad_clip": (args.grad_clip, 0.0),
    }
    mismatches.extend(
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in expected_float.items()
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
    )
    if args.run_name and args.run_name != config_id:
        mismatches.append(f"run_name={args.run_name!r} (expected {config_id!r} or empty)")
    if mismatches:
        raise ValueError("Formal protocol mismatch: " + "; ".join(mismatches))
    return config_id


def _topk_correct(logits, targets):
    max_k = min(5, logits.shape[1])
    predictions = logits.topk(max_k, dim=1).indices
    matches = predictions.eq(targets.view(-1, 1))
    top1 = int(matches[:, :1].any(dim=1).sum().item())
    top5 = int(matches.any(dim=1).sum().item())
    return top1, top5


def _reduce_totals(totals, device, world_size):
    tensor = torch.tensor(totals, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def _evaluate(model, dataloader, device, world_size, amp_enabled):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    totals = [0.0, 0.0, 0.0, 0.0]
    with torch.no_grad():
        for images, targets, rejected_attempts in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_cuda(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            top1, top5 = _topk_correct(logits, targets)
            batch_size = targets.shape[0]
            totals[0] += float(loss.item()) * batch_size
            totals[1] += top1
            totals[2] += top5
            totals[3] += batch_size
    loss_sum, top1_sum, top5_sum, sample_count = _reduce_totals(totals, device, world_size)
    model.train()
    return {
        "loss": loss_sum / max(1.0, sample_count),
        "top1": 100.0 * top1_sum / max(1.0, sample_count),
        "top5": 100.0 * top5_sum / max(1.0, sample_count),
        "samples": int(sample_count),
    }


def _save_final_checkpoint(path, model, variant, args, bank, train_dataset, runtime, train_metrics, val_metrics):
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    encoder_state = {key: value for key, value in state.items() if not key.startswith("head.")}
    head_state = {
        key[len("head."):]: value
        for key, value in state.items()
        if key.startswith("head.")
    }
    payload = {
        "variant": variant,
        "model_state": encoder_state,
        "formula_head_state": head_state,
        "epoch": args.epochs,
        "global_step": runtime["total_steps"],
        "seed": args.seed,
        "config": runtime,
        "bank_hash": bank.bank_hash,
        "instance_pool_hash": train_dataset.pool_hash,
        "label_mapping": bank.label_mapping,
        "final_train_metrics": train_metrics,
        "final_validation_metrics": val_metrics,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def train_one_variant(variant, args, device, rank=0, world_size=1):
    is_main_process = rank == 0
    output_dir = Path(args.output_dir) / f"vit_{variant}"
    final_path = output_dir / f"formula_sl_vit_{variant}_final.pth"
    log_path = output_dir / "train.log"
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        if final_path.exists() or log_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing supervised run: {output_dir}"
            )
    barrier(world_size, device)

    logger = logging.getLogger(f"pretrain_sl_{variant}_{args.bank_subset}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    if is_main_process:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        logger.addHandler(logging.NullHandler())

    try:
        bank = FormulaBank(args.formula_bank, args.bank_subset, args.subset_registry)
        per_rank_batch, global_batch = resolve_batch_size(variant, args, world_size)
        steps_per_epoch = args.epoch_len // global_batch
        if steps_per_epoch <= 0:
            raise ValueError("epoch_len must contain at least one complete global batch")
        exposures_per_epoch = steps_per_epoch * global_batch
        total_steps = args.epochs * steps_per_epoch
        total_exposures = total_steps * global_batch
        peak_lr = args.lr if args.lr > 0 else args.reference_lr * global_batch / args.reference_batch_size
        warmup_lr = args.warmup_lr * global_batch / args.reference_batch_size
        min_lr = args.min_lr * global_batch / args.reference_batch_size
        warmup_steps = min(total_steps, args.warmup_epochs * steps_per_epoch)
        config_id = _validate_formal_protocol(
            bank,
            variant,
            args,
            device,
            world_size,
            global_batch,
            steps_per_epoch,
            total_steps,
            total_exposures,
            peak_lr,
            warmup_lr,
            min_lr,
        )
        runtime = {
            "objective": "formula_supervised_cross_entropy",
            "config_id": config_id,
            "formal_protocol": bank.is_formal_subset and not args.dev_run,
            "development_run": args.dev_run,
            "img_size": (1024, 128),
            "patch_size": 16,
            "variant": variant,
            "num_classes": bank.num_classes,
            "bank_id": bank.bank_id,
            "bank_version": bank.bank_version,
            "bank_subset": bank.subset,
            "bank_hash": bank.bank_hash,
            "subset_registry": None if bank.subset_registry is None else {
                "path": bank.subset_registry["path"],
                "sha256": bank.subset_registry["sha256"],
                "scale_id": bank.subset_registry["scale_id"],
            },
            "generator_commit": bank.generator_commit,
            "generator_config_hash": bank.generator_config_hash,
            "admission_report": bank.admission_report,
            "instances_per_class": args.instances_per_class,
            "val_instances_per_class": args.val_instances_per_class,
            "unique_training_samples": bank.num_classes * args.instances_per_class,
            "mean_exposures_per_unique_training_sample": (
                total_exposures / (bank.num_classes * args.instances_per_class)
            ),
            "epochs": args.epochs,
            "epoch_len_nominal": args.epoch_len,
            "exposures_per_epoch": exposures_per_epoch,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "total_exposures": total_exposures,
            "per_rank_batch_size": per_rank_batch,
            "global_batch_size": global_batch,
            "world_size": world_size,
            "peak_lr": peak_lr,
            "warmup_lr": warmup_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "weight_decay": args.weight_decay,
            "betas": (0.9, 0.999),
            "label_smoothing": args.label_smoothing,
            "drop_path": args.drop_path,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "seed": args.seed,
            "run_name": args.run_name or config_id,
        }
        base_exposures, extra_exposures = divmod(total_exposures, bank.num_classes)
        class_exposure_histogram = {
            template["class_id"]: base_exposures + (label < extra_exposures)
            for label, template in enumerate(bank.templates)
        }
        runtime["class_exposure_histogram"] = class_exposure_histogram

        train_dataset = FormulaExposureDataset(
            bank=bank,
            split="train",
            seed=args.seed,
            instances_per_class=args.instances_per_class,
            exposure_count=exposures_per_epoch,
            max_render_attempts=args.max_render_attempts,
        )
        val_dataset = FormulaExposureDataset(
            bank=bank,
            split="val",
            seed=args.seed,
            instances_per_class=args.val_instances_per_class,
            max_render_attempts=args.max_render_attempts,
        )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=per_rank_batch,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
        if len(train_loader) != steps_per_epoch:
            raise RuntimeError(
                f"Expected {steps_per_epoch} steps/epoch, DataLoader produced {len(train_loader)}"
            )

        logger.info("Formula-driven supervised pre-training starts")
        logger.info("Host=%s device=%s torch=%s timm=%s", socket.gethostname(), device, torch.__version__, timm.__version__)
        if device.type == "cuda":
            logger.info("GPU=%s", torch.cuda.get_device_name(device))
        logger.info("Command=%s", " ".join(sys.argv))
        logger.info("Resolved args=%s", _canonical_json(vars(args)))
        logger.info("Runtime=%s", _canonical_json(runtime))
        logger.info(
            "Formula Bank=%s version=%s subset=%s classes=%d hash=%s generator_commit=%s generator_config_hash=%s admission=%s",
            bank.bank_id,
            bank.bank_version or "non-formal",
            bank.subset,
            bank.num_classes,
            bank.bank_hash,
            bank.generator_commit or "non-formal",
            bank.generator_config_hash or "non-formal",
            _canonical_json(bank.admission_report),
        )
        logger.info("Label mapping hash=%s", hashlib.sha256(_canonical_json(bank.label_mapping).encode("utf-8")).hexdigest())
        logger.info("Train instance pool hash=%s", train_dataset.pool_hash)
        logger.info("Full-run class exposure histogram=%s", _canonical_json(class_exposure_histogram))

        model = FormulaViTClassifier(
            num_classes=bank.num_classes,
            variant=variant,
            drop_path_rate=args.drop_path,
        ).to(device)
        logger.info("Trainable parameters=%.2fM", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
        if world_size > 1:
            model = DDP(model, device_ids=[device.index], output_device=device.index)
        optimizer = optim.AdamW(
            _optimizer_groups(model, args.weight_decay),
            lr=peak_lr,
            betas=(0.9, 0.999),
        )
        mixup_active = args.mixup_alpha > 0 or args.cutmix_alpha > 0
        mixup_fn = None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup_alpha,
                cutmix_alpha=args.cutmix_alpha,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=args.label_smoothing,
                num_classes=bank.num_classes,
            )
            train_criterion = SoftTargetCrossEntropy()
            logger.warning("Mixup/CutMix enabled: this is not the formal Atomic-Clean protocol")
        else:
            train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        amp_enabled = device.type == "cuda" and args.amp
        scaler = make_grad_scaler(enabled=amp_enabled)
        run_class_exposures = torch.zeros(bank.num_classes, dtype=torch.long, device=device)
        run_class_rejected_attempts = torch.zeros(bank.num_classes, dtype=torch.long, device=device)

        global_step = 0
        final_train_metrics = None
        training_started = time.time()
        for epoch in range(args.epochs):
            epoch_started = time.time()
            train_dataset.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            model.train()
            totals = [0.0, 0.0, 0.0, 0.0, 0.0]
            for step_in_epoch, (images, targets, rejected_attempts) in enumerate(train_loader):
                step_started = time.time()
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                hard_targets = targets
                rejected_attempts_device = rejected_attempts.to(device, non_blocking=True, dtype=torch.long)
                run_class_exposures.scatter_add_(
                    0, hard_targets, torch.ones_like(hard_targets, dtype=torch.long)
                )
                run_class_rejected_attempts.scatter_add_(
                    0, hard_targets, rejected_attempts_device
                )
                if mixup_fn is not None:
                    images, loss_targets = mixup_fn(images, targets)
                else:
                    loss_targets = targets
                learning_rate = _learning_rate_at_step(
                    global_step,
                    total_steps,
                    warmup_steps,
                    peak_lr,
                    warmup_lr,
                    min_lr,
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                with autocast_cuda(enabled=amp_enabled):
                    logits = model(images)
                    loss = train_criterion(logits, loss_targets)
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                top1, top5 = _topk_correct(logits.detach(), hard_targets)
                batch_size = hard_targets.shape[0]
                totals[0] += float(loss.detach().item()) * batch_size
                totals[1] += top1
                totals[2] += top5
                totals[3] += batch_size
                totals[4] += int(rejected_attempts_device.sum().item())
                global_step += 1
                if is_main_process and (global_step == 1 or global_step % args.log_interval == 0):
                    memory_gib = 0.0
                    if device.type == "cuda":
                        memory_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    logger.info(
                        "Step %d/%d epoch=%d/%d step_in_epoch=%d/%d loss=%.6f top1=%.2f top5=%.2f lr=%.6e throughput=%.2f/s max_mem=%.2fGiB rejected_attempts=%d",
                        global_step,
                        total_steps,
                        epoch + 1,
                        args.epochs,
                        step_in_epoch + 1,
                        steps_per_epoch,
                        float(loss.detach().item()),
                        100.0 * top1 / max(1, batch_size),
                        100.0 * top5 / max(1, batch_size),
                        learning_rate,
                        batch_size / max(1e-9, time.time() - step_started),
                        memory_gib,
                        int(rejected_attempts_device.sum().item()),
                    )

            loss_sum, top1_sum, top5_sum, sample_count, rejected_sum = _reduce_totals(totals, device, world_size)
            final_train_metrics = {
                "loss": loss_sum / max(1.0, sample_count),
                "top1": 100.0 * top1_sum / max(1.0, sample_count),
                "top5": 100.0 * top5_sum / max(1.0, sample_count),
                "samples": int(sample_count),
                "rejected_attempts": int(rejected_sum),
            }
            if is_main_process:
                logger.info(
                    "Epoch %d/%d complete global_step=%d loss=%.6f top1=%.3f top5=%.3f samples=%d rejected_attempts=%d elapsed=%.1fs",
                    epoch + 1,
                    args.epochs,
                    global_step,
                    final_train_metrics["loss"],
                    final_train_metrics["top1"],
                    final_train_metrics["top5"],
                    final_train_metrics["samples"],
                    final_train_metrics["rejected_attempts"],
                    time.time() - epoch_started,
                )

        if global_step != total_steps:
            raise RuntimeError(f"Training ended at {global_step} steps; expected {total_steps}")
        if world_size > 1:
            dist.all_reduce(run_class_exposures, op=dist.ReduceOp.SUM)
            dist.all_reduce(run_class_rejected_attempts, op=dist.ReduceOp.SUM)
        observed_exposures = run_class_exposures.cpu().tolist()
        observed_rejections = run_class_rejected_attempts.cpu().tolist()
        observed_exposure_histogram = {
            template["class_id"]: int(observed_exposures[label])
            for label, template in enumerate(bank.templates)
        }
        if observed_exposure_histogram != class_exposure_histogram:
            raise RuntimeError(
                "Observed class exposure histogram does not match the precomputed balanced schedule"
            )
        class_rejection_audit = {
            template["class_id"]: {
                "exposures": int(observed_exposures[label]),
                "rejected_attempts": int(observed_rejections[label]),
                "rejected_attempts_per_exposure": (
                    float(observed_rejections[label]) / max(1, int(observed_exposures[label]))
                ),
            }
            for label, template in enumerate(bank.templates)
        }
        runtime["observed_class_exposure_histogram"] = observed_exposure_histogram
        runtime["class_rejection_audit"] = class_rejection_audit
        if is_main_process:
            logger.info("Observed class exposure histogram=%s", _canonical_json(observed_exposure_histogram))
            logger.info("Per-class rejection audit=%s", _canonical_json(class_rejection_audit))
        val_metrics = _evaluate(model, val_loader, device, world_size, amp_enabled)
        if is_main_process:
            logger.info("Final validation=%s", _canonical_json(val_metrics))
            _save_final_checkpoint(
                final_path,
                unwrap_model(model),
                variant,
                args,
                bank,
                train_dataset,
                runtime,
                final_train_metrics,
                val_metrics,
            )
            checkpoint_hash = _sha256_file(final_path)
            logger.info("Final checkpoint=%s sha256=%s", final_path.resolve(), checkpoint_hash)
            logger.info("Training completed successfully in %.1fs", time.time() - training_started)
        barrier(world_size, device)
        del model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        logger.exception("Supervised pre-training failed")
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Formula-driven supervised pre-training for procedural single-event audio"
    )
    parser.add_argument("--formula-bank", type=str, required=True, help="Path to a Formula Bank JSON manifest")
    parser.add_argument("--bank-subset", type=str, required=True, help="Subset name recorded in the Formula Bank")
    parser.add_argument(
        "--subset-registry",
        type=str,
        default="",
        help="Optional frozen additive scale-subset registry bound to the Formula Bank hash",
    )
    parser.add_argument("--instances-per-class", type=int, required=True, help="Frozen unique training instances per class")
    parser.add_argument("--val-instances-per-class", type=int, default=50, help="Frozen validation instances per class")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["small"],
        choices=list(FORMULA_MODEL_CONFIGS),
        help="Formal protocol uses small; other sizes are development-only",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--epoch-len", type=int, default=5000, help="Nominal samples per epoch; incomplete global batch is omitted")
    parser.add_argument("--batch-size", type=int, default=0, help="Per-rank batch when global batch is unset")
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--reference-lr", type=float, default=1e-3, help="Reference peak LR at --reference-batch-size")
    parser.add_argument("--reference-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0, help="Explicit peak LR; 0 enables linear batch scaling")
    parser.add_argument("--warmup-lr", type=float, default=1e-4, help="Reference warmup LR before batch scaling")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Reference cosine floor before batch scaling")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--max-render-attempts", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/formula_class_instance_ablation/pretrain",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument(
        "--dev-run",
        action="store_true",
        help="Allow a non-formal sanity run on a formal Bank; permanently marks the run non-formal",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution for smoke/debug runs")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.epoch_len <= 0:
        parser.error("--epochs and --epoch-len must be positive")
    if args.instances_per_class <= 0 or args.val_instances_per_class <= 0:
        parser.error("instance counts must be positive")
    if args.num_workers < 0 or args.max_render_attempts <= 0:
        parser.error("--num-workers must be nonnegative and --max-render-attempts must be positive")
    if args.batch_size < 0 or args.global_batch_size < 0:
        parser.error("Batch-size values must be nonnegative")
    if args.log_interval <= 0 or args.reference_batch_size <= 0:
        parser.error("--log-interval and --reference-batch-size must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must lie in [0, 1)")
    if not 0.0 <= args.drop_path < 1.0:
        parser.error("--drop-path must lie in [0, 1)")
    if args.mixup_alpha < 0.0 or args.cutmix_alpha < 0.0 or args.grad_clip < 0.0:
        parser.error("Mixup, CutMix, and gradient-clipping values must be nonnegative")
    if args.lr < 0.0 or args.reference_lr <= 0.0 or args.warmup_lr < 0.0 or args.min_lr < 0.0:
        parser.error("Learning-rate values must be nonnegative and --reference-lr must be positive")
    if args.weight_decay < 0.0 or args.warmup_epochs < 0:
        parser.error("--weight-decay and --warmup-epochs must be nonnegative")
    return args


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed(force_cpu=args.cpu)
    try:
        set_seed(args.seed + rank)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        for variant in args.models:
            train_one_variant(variant, args, device, rank=rank, world_size=world_size)
    finally:
        cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
