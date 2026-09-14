import sys
import types
import collections
import collections.abc
import math
import random
import argparse
import os
import logging
import json
import hashlib
import multiprocessing as mp
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import torchaudio
import torchaudio.functional as F
from functools import partial
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

    @torch.no_grad()
    def forward_encoder_with_attention(self, x, layer=-1):
        """Run the unmasked encoder and return one block's self-attention.

        The training encoder intentionally shuffles patches in ``random_masking``.
        For visualization we must bypass that operation, otherwise the attention
        map would no longer be in the original time-frequency order.  The returned
        tensor has shape ``[B, heads, 1 + num_patches, 1 + num_patches]``.
        """
        if not (-len(self.blocks) <= layer < len(self.blocks)):
            raise ValueError(f"attention layer {layer} is out of range")
        target = self.blocks[layer]
        captured = {}

        def capture_qkv(_module, _inputs, output):
            # timm Attention computes q/k from this projection, including when
            # fused attention is enabled, so a hook here is version-agnostic.
            b, n, three_c = output.shape
            num_heads = target.attn.num_heads
            head_dim = getattr(target.attn, "head_dim", three_c // 3 // num_heads)
            qkv = output.reshape(b, n, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            q_norm = getattr(target.attn, "q_norm", None)
            k_norm = getattr(target.attn, "k_norm", None)
            if q_norm is not None:
                q = q_norm(q)
            if k_norm is not None:
                k = k_norm(k)
            scale = getattr(target.attn, "scale", head_dim ** -0.5)
            captured["attention"] = (q * scale @ k.transpose(-2, -1)).softmax(dim=-1)

        handle = target.attn.qkv.register_forward_hook(capture_qkv)
        try:
            tokens = self.patch_embed(x)
            tokens = tokens + self.pos_embed[:, 1:, :]
            cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
            for block in self.blocks:
                tokens = block(tokens)
        finally:
            handle.remove()
        if "attention" not in captured:
            raise RuntimeError("failed to capture ViT attention weights")
        return tokens, captured["attention"]

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
# The synthesizer changes low-level acoustics without introducing semantic labels.
# - Multiple sources, configurable event layouts, and diverse source families.
# - Transient shaping, spectral coloration, and lightweight channel effects.
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


def _load_model_weights(model, checkpoint_path, device):
    """Load a full MAE checkpoint or an encoder-only final checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    decoder_only_prefixes = (
        "mask_token",
        "decoder_pos_embed",
        "decoder_embed.",
        "decoder_blocks.",
        "decoder_norm.",
        "decoder_pred.",
    )
    invalid_missing = [
        key for key in missing
        if not key.startswith(decoder_only_prefixes)
    ]
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the encoder "
            f"(missing={invalid_missing}, unexpected={unexpected})"
        )
    return "encoder-only" if missing else "full-model"


def save_audio_attention_analysis(
    model, dataset, output_dir, num_samples=5, layer=-1, checkpoint_path=None
):
    """Save spectrogram, CLS-to-patch attention, and overlay for five samples.

    AudioMAE uses a 1024x128 time-frequency input and 16x16 patches, yielding a
    64x8 attention grid.  Attention is averaged over heads and taken from the
    selected encoder block, matching the paper's direct self-attention analysis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if num_samples != 5:
        raise ValueError("This analysis is intentionally limited to exactly 5 samples")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    patch_h, patch_w = model.patch_embed.patch_hw
    records = []

    for index in range(num_samples):
        spectrogram, _ = dataset[index]
        spectrogram = spectrogram.unsqueeze(0).to(device)
        _, attention = model.forward_encoder_with_attention(spectrogram, layer=layer)
        # CLS token attends to the audio patches; discard CLS-to-CLS entry.
        patch_attention = attention[:, :, 0, 1:].mean(dim=1)
        patch_attention = patch_attention.reshape(1, 1, patch_h, patch_w)
        patch_attention = nnF.interpolate(
            patch_attention,
            size=(spectrogram.shape[-2], spectrogram.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
        patch_attention -= patch_attention.min()
        patch_attention /= max(float(patch_attention.max()), 1e-8)
        spec = spectrogram[0, 0].cpu().numpy()

        stem = output_dir / f"sample_{index:02d}"
        for kind, attention_alpha in (("spectrogram", None), ("attention", None), ("overlay", 0.45)):
            fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
            ax.imshow(spec.T, origin="lower", aspect="auto", cmap="magma")
            if kind == "attention":
                ax.clear()
                ax.imshow(patch_attention.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
            elif kind == "overlay":
                ax.imshow(patch_attention.T, origin="lower", aspect="auto", cmap="jet", vmin=0, vmax=1, alpha=attention_alpha)
            ax.set_xlabel("Time frame")
            ax.set_ylabel("Mel bin")
            ax.set_title(f"AudioMAE {kind} (sample {index})")
            fig.tight_layout()
            fig.savefig(f"{stem}_{kind}.png")
            plt.close(fig)
        records.append({
            "sample_index": index,
            "attention_layer": layer,
            "patch_grid": [patch_h, patch_w],
            "input_shape": [int(spectrogram.shape[-2]), int(spectrogram.shape[-1])],
            "attention_definition": "mean_heads(CLS -> patch)",
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        })

    (output_dir / "metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

# =========================================================
# Pre-training entry point for Tiny, Small, Base, and Large variants.
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Initialize torchrun DDP when WORLD_SIZE is set; otherwise stay single-process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed pre-training requires CUDA/NCCL.")

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
    """Return per-rank and effective global batches without changing legacy defaults."""
    if world_size > 1 and args.global_batch_size <= 0:
        raise ValueError(
            "DDP pre-training requires --global-batch-size so the effective optimization "
            "batch cannot silently change with the number of GPUs."
        )
    if args.global_batch_size > 0:
        if args.global_batch_size % world_size != 0:
            raise ValueError(
                f"--global-batch-size ({args.global_batch_size}) must be divisible by "
                f"WORLD_SIZE ({world_size})."
            )
        return args.global_batch_size // world_size, args.global_batch_size

    per_rank_batch = DEFAULT_BATCH_SIZE[variant] if args.batch_size <= 0 else args.batch_size
    return per_rank_batch, per_rank_batch * world_size


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(variant):
    if variant not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model variant: {variant}. Choose from {list(MODEL_FACTORIES.keys())}")
    model = MODEL_FACTORIES[variant](
        norm_pix_loss=True,
        audio_exp=True,
        in_chans=1,
        img_size=(1024, 128),
    )
    return model


def latest_checkpoint(ckpt_dir, variant):
    ckpts = sorted(Path(ckpt_dir).glob(f"audiomae_vit_{variant}_epoch_*.pth"))
    if not ckpts:
        return None

    def parse_epoch(path):
        stem = path.stem
        # audiomae_vit_base_epoch_100 -> 100
        try:
            return int(stem.split("_epoch_")[-1])
        except Exception:
            return -1

    ckpts = sorted(ckpts, key=parse_epoch)
    return ckpts[-1]


def save_checkpoint(path, model, optimizer, scaler, epoch, variant, args, avg_loss, runtime=None):
    ckpt = {
        "epoch": epoch,
        "variant": variant,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "avg_loss": avg_loss,
        "config": {
            "img_size": (1024, 128),
            "patch_size": 16,
            "mask_ratio": args.mask_ratio,
            "norm_pix_loss": True,
            "audio_exp": True,
            "in_chans": 1,
            "epochs": args.epochs,
            "epoch_len": args.epoch_len,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "global_batch_size": args.global_batch_size,
            "deterministic_synth": args.deterministic_synth,
            "save_every": args.save_every,
            "synth_config_path": args.synth_config,
            "synth_config_hash": SYNTH_CONFIG_HASH,
            "synth_config": SYNTH_CONFIG,
            "runtime": runtime or {},
        },
    }
    torch.save(ckpt, path)


def save_final_weights(path, model, variant, args, runtime=None):
    # Save a weights-only copy for direct fine-tuning with load_state_dict.
    torch.save(
        {
            "variant": variant,
            "model_state": model.state_dict(),
            "config": {
                "img_size": (1024, 128),
                "patch_size": 16,
                "norm_pix_loss": True,
                "audio_exp": True,
                "in_chans": 1,
                "synth_config_path": args.synth_config,
                "synth_config_hash": SYNTH_CONFIG_HASH,
                "synth_config": SYNTH_CONFIG,
                "runtime": runtime or {},
            },
        },
        path,
    )


def train_one_variant(variant, args, device, rank=0, world_size=1):
    is_main_process = rank == 0
    if is_main_process:
        print("\n" + "=" * 88)
        print(f"🚀 Start pre-training AudioMAE ViT-{variant.upper()} for {args.epochs} epochs")
        print("=" * 88)

    # Only rank 0 owns files; all ranks wait until the directory exists.
    ckpt_dir = Path(args.output_dir) / f"vit_{variant}"
    if is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    barrier(world_size, device)

    logger = logging.getLogger(f"pretrain_{variant}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    if is_main_process:
        fh = logging.FileHandler(ckpt_dir / "train.log")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    else:
        logger.addHandler(logging.NullHandler())

    batch_size, global_batch_size = resolve_batch_size(variant, args, world_size)
    runtime = {
        "world_size": world_size,
        "per_rank_batch_size": batch_size,
        "global_batch_size": global_batch_size,
        "steps_per_epoch": args.epoch_len // global_batch_size,
        "deterministic_synth": args.deterministic_synth,
    }
    logger.info(
        "📦 Per-rank batch=%d; effective global batch=%d; steps/epoch=%d",
        batch_size,
        global_batch_size,
        runtime["steps_per_epoch"],
    )

    # With deterministic synthesis, sample (seed, epoch, index) is independent of rank and worker.
    dataset = PhysicsDataset(
        epoch_len=args.epoch_len,
        seed=args.seed if args.deterministic_synth else None,
    )
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"No complete global batch: epoch_len={args.epoch_len}, "
            f"global_batch_size={global_batch_size}."
        )

    logger.info(f"🏗️ Building AudioMAE ViT-{variant.upper()}...")
    model = build_model(variant).to(device)
    n_params = count_parameters(model)
    logger.info(f"🔢 Trainable parameters: {n_params / 1e6:.2f}M")
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    amp_enabled = (device.type == "cuda" and args.amp)
    scaler = make_grad_scaler(enabled=amp_enabled)

    start_epoch = 0
    if args.auto_resume:
        last = latest_checkpoint(ckpt_dir, variant)
        if last is not None:
            logger.info(f"🔁 Auto-resume from {last}")
            checkpoint = torch.load(last, map_location=device)
            unwrap_model(model).load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if checkpoint.get("scaler_state") is not None and scaler is not None:
                scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint.get("epoch", 0))
            logger.info(f"✅ Resumed at epoch {start_epoch}")

    if start_epoch >= args.epochs:
        logger.info(f"✅ ViT-{variant.upper()} already finished {start_epoch}/{args.epochs} epochs. Skip.")
        barrier(world_size, device)
        return

    model.train()
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        pbar = tqdm(
            dataloader,
            desc=f"ViT-{variant.upper()} Epoch {epoch + 1}/{args.epochs}",
            disable=not is_main_process,
        )
        for batch in pbar:
            images, _ = batch
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(enabled=amp_enabled):
                loss, _, _ = model(images, mask_ratio=args.mask_ratio)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().item())
            total_loss += loss_value
            if is_main_process:
                pbar.set_postfix({"loss": f"{loss_value:.4f}"})

        if world_size > 1:
            loss_sum = torch.tensor(total_loss, device=device)
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            total_loss = float(loss_sum.item())
        avg_loss = total_loss / max(1, len(dataloader) * world_size)
        logger.info(f"✅ ViT-{variant.upper()} Epoch {epoch + 1}/{args.epochs} Done. Avg Loss: {avg_loss:.4f}")

        if is_main_process and (((epoch + 1) % args.save_every == 0) or ((epoch + 1) == args.epochs)):
            ckpt_path = ckpt_dir / f"audiomae_vit_{variant}_epoch_{epoch + 1}.pth"
            save_checkpoint(
                ckpt_path,
                unwrap_model(model),
                optimizer,
                scaler,
                epoch + 1,
                variant,
                args,
                avg_loss,
                runtime,
            )
            logger.info(f"💾 Saved checkpoint: {ckpt_path}")
        barrier(world_size, device)

    if is_main_process:
        final_path = ckpt_dir / f"audiomae_vit_{variant}_final.pth"
        save_final_weights(final_path, unwrap_model(model), variant, args, runtime)
        logger.info(f"🏁 Finished ViT-{variant.upper()}. Final weights saved to: {final_path}")
    barrier(world_size, device)

    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-click AudioMAE pre-training on complex procedural synthesizer v3."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tiny", "small", "base", "large"],
        choices=["tiny", "small", "base", "large"],
        help="Model variants to train. Default: tiny small base large.",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Pre-training epochs for each model. Default: 500.")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N epochs. Default: 100.")
    parser.add_argument("--epoch-len", type=int, default=5000, help="Synthetic samples per epoch. Default: 5000.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size. 0 means auto per model: tiny=64, small=32, base=16, large=4.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=0,
        help=(
            "Effective batch across all DDP ranks. Required under torchrun; 0 keeps "
            "--batch-size behavior for a single process."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers. Default: 4.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate. Default: 1e-4.")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="AdamW weight decay. Default: 0.05.")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="MAE mask ratio. Default: 0.75.")
    parser.add_argument("--output-dir", type=str, default="checkpoints_audiomae_v3", help="Checkpoint output directory.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed. Default: 2026.")
    parser.add_argument("--synth-config", type=str, default="", help="Optional JSON synthesizer config.")
    parser.add_argument("--run-name", type=str, default="", help="Optional run name for logs/checkpoint metadata.")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping max norm. 0 disables it.")
    parser.add_argument(
        "--deterministic-synth",
        action="store_true",
        help="Derive every synthetic waveform from (seed, epoch, sample index) for DDP-stable data generation.",
    )
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable CUDA mixed precision.")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
        help="Disable automatic resume from latest checkpoint in each variant directory.",
    )
    parser.set_defaults(auto_resume=True)
    parser.add_argument(
        "--attention-analysis",
        action="store_true",
        help="Skip training and save attention visualizations for exactly five synthetic spectrograms.",
    )
    parser.add_argument(
        "--attention-checkpoint",
        type=str,
        default="",
        help="Checkpoint to analyze (defaults to the latest/final checkpoint for each --models variant).",
    )
    parser.add_argument(
        "--attention-output-dir",
        type=str,
        default="attention_analysis_audio",
        help="Directory for five spectrogram/attention/overlay PNG triplets.",
    )
    parser.add_argument(
        "--attention-layer",
        type=int,
        default=-1,
        help="Encoder block used for attention (default: final block).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()
    is_main_process = rank == 0
    try:
        synth_cfg, synth_hash = _load_synth_config(args.synth_config)
        set_synth_config(synth_cfg, synth_hash)
        # DDP broadcasts model weights from rank 0. Offset nonzero ranks only for local
        # stochastic operations; deterministic synthesis remains keyed by args.seed.
        set_seed(args.seed + rank)

        # CUDA performance settings; CPU behavior is unchanged.
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        if is_main_process:
            print(f"🔥 One-click AudioMAE pre-training on device: {device}")
            print(f"🌐 DDP world size: {world_size}")
            print(f"🎛️ Models: {args.models}")
            print(f"🧪 Synthesizer: {SYNTH_CONFIG.get('name', 'unnamed')} hash={SYNTH_CONFIG_HASH}")
            if args.synth_config:
                print(f"🧪 Synth config path: {args.synth_config}")
            if args.run_name:
                print(f"🏷️ Run name: {args.run_name}")
            print(f"📌 Epochs per model: {args.epochs}; checkpoint every {args.save_every} epochs")
            print(f"📁 Output dir: {args.output_dir}")
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        barrier(world_size, device)

        if args.attention_analysis:
            if world_size > 1:
                raise RuntimeError("--attention-analysis is intended for a single process (omit torchrun).")
            dataset = PhysicsDataset(epoch_len=5, seed=args.seed)
            for variant in args.models:
                model = build_model(variant).to(device)
                checkpoint_path = args.attention_checkpoint
                if not checkpoint_path:
                    ckpt_dir = Path(args.output_dir) / f"vit_{variant}"
                    final_path = ckpt_dir / f"audiomae_vit_{variant}_final.pth"
                    checkpoint_path = str(final_path if final_path.exists() else latest_checkpoint(ckpt_dir, variant) or "")
                if checkpoint_path:
                    checkpoint_kind = _load_model_weights(model, checkpoint_path, device)
                    print(f"🔍 Attention checkpoint ({checkpoint_kind}): {checkpoint_path}")
                else:
                    print(f"⚠️ No checkpoint found for {variant}; visualizing random initialization.")
                save_dir = Path(args.attention_output_dir) / f"vit_{variant}"
                save_audio_attention_analysis(
                    model,
                    dataset,
                    save_dir,
                    num_samples=5,
                    layer=args.attention_layer,
                    checkpoint_path=checkpoint_path or None,
                )
                print(f"🖼️ Saved five attention analyses to: {save_dir.resolve()}")
            return

        for variant in args.models:
            train_one_variant(variant, args, device, rank=rank, world_size=world_size)

        if is_main_process:
            print("\n🎉 All requested AudioMAE variants finished!")
            print(f"📁 Checkpoints are under: {Path(args.output_dir).resolve()}")
    finally:
        cleanup_distributed(world_size)


if __name__ == '__main__':
    main()
