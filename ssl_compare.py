#!/usr/bin/env python3
"""Distributed SSL pre-training for the AudioMAE comparison study.

The exported checkpoint always contains only the AudioMAE-compatible ViT
encoder keys.  ESC-50, FSD50K, and UrbanSound8K can therefore load every
method through their existing full fine-tuning scripts.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as torch_f
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

import pretrain as audiomae


ENCODER_EXCLUDE = ("decoder", "mask_token")
MOCO_METHODS = (
    "mocov3",
    "mocov3_templatequeue",
    "mocov3_templatequeue_nomixup",
    "mocov3_templatequeue_nomixup_noqueue",
    "mocov3_minimalpair",
    "mocov3_weaknoisepair",
)
TEMPLATE_PAIR_METHODS = (
    "mocov3_templatequeue",
    "mocov3_templatequeue_nomixup",
    "mocov3_templatequeue_nomixup_noqueue",
    "mocov3_minimalpair",
    "mocov3_weaknoisepair",
)
TEMPLATEQUEUE_METHODS = (
    "mocov3_templatequeue",
    "mocov3_templatequeue_nomixup",
    "mocov3_templatequeue_nomixup_noqueue",
)


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AudioMAE / MoCo-v3 / BYOL-ViT controlled SSL comparison"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=("audiomae", *MOCO_METHODS, "byol_vit"),
    )
    parser.add_argument("--model-size", choices=("base",), default="base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synth-config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--epoch-len", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0,
                        help="Explicit learning rate override; zero selects the method recipe.")
    parser.add_argument("--weight-decay", type=float, default=-1.0,
                        help="Explicit weight-decay override; negative selects the method recipe.")
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--moco-base-m", type=float, default=0.99)
    parser.add_argument("--moco-base-lr", type=float, default=1.5e-4,
                        help="MoCo base LR at global batch 256.")
    parser.add_argument("--byol-base-m", type=float, default=0.99)
    parser.add_argument("--byol-base-lr", type=float, default=1.5e-4,
                        help="BYOL-ViT base LR at global batch 256.")
    parser.add_argument("--warmup-epochs", type=int, default=40)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-hidden-dim", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--moco-queue-size", type=int, default=0,
                        help="Momentum-key queue size; zero keeps batch-only MoCo v3.")
    parser.add_argument("--view-mask-ratio", type=float, default=0.2)
    parser.add_argument("--view-log-gain", type=float, default=0.1)
    parser.add_argument("--view-noise-std", type=float, default=0.02)
    parser.add_argument(
        "--train-patch-embed",
        action="store_true",
        help=(
            "Train the log-Mel patch projection in contrastive/BYOL encoders. "
            "The historical comparison runs leave it frozen."
        ),
    )
    parser.add_argument("--max-steps-per-epoch", type=int, default=0,
                        help="Nonzero only for smoke tests; bounds steps in every epoch.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing final checkpoint.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def setup_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(rank, world_size, local_rank, device)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(ctx: DistributedContext) -> None:
    if ctx.world_size > 1:
        if ctx.device.type == "cuda":
            dist.barrier(device_ids=[ctx.local_rank])
        else:
            dist.barrier()


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@contextmanager
def temporary_rng_state(seed: int) -> Iterable[None]:
    """Render one procedural template without perturbing the worker RNG stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def waveform_to_fbank(waveform: torch.Tensor, target_length: int = 1024) -> torch.Tensor:
    fbank = audiomae.torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )
    if fbank.shape[0] < target_length:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_length - fbank.shape[0]))
    else:
        fbank = fbank[:target_length]
    return ((fbank - (-4.26)) / (4.56 * 2)).unsqueeze(0)


def sample_synth_f0() -> float:
    f0_min, f0_max = map(float, audiomae.SYNTH_CONFIG.get("f0_range_hz", [50.0, 2000.0]))
    distribution = str(audiomae.SYNTH_CONFIG.get("f0_distribution", "uniform")).replace("-", "_")
    if distribution == "uniform":
        return float(np.random.uniform(f0_min, f0_max))
    if distribution == "log_uniform":
        return float(np.exp(np.random.uniform(np.log(f0_min), np.log(f0_max))))
    raise ValueError(f"Unsupported f0_distribution: {distribution}")


def apply_template_nuisance(waveform: torch.Tensor, seed: int) -> torch.Tensor:
    """Independent post-render nuisance while retaining the event template."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    result = waveform.clone()
    result = result * (10.0 ** (rng.uniform(-3.0, 3.0) / 20.0))

    if rng.random() < 0.6:
        delay = int(rng.uniform(0.003, 0.05) * 16000)
        if 0 < delay < result.numel():
            result[delay:] += rng.uniform(0.03, 0.18) * result[:-delay]
    if rng.random() < 0.5:
        result = audiomae._safe_biquad(
            audiomae.F.lowpass_biquad, result, 16000, cutoff_freq=rng.uniform(3500.0, 7000.0)
        )
    if rng.random() < 0.4:
        result = audiomae._safe_biquad(
            audiomae.F.highpass_biquad, result, 16000, cutoff_freq=rng.uniform(20.0, 220.0)
        )

    noise = torch.randn(result.shape, generator=generator, dtype=result.dtype)
    snr_db = rng.uniform(22.0, 38.0)
    noise = noise * (audiomae._rms(result) / (audiomae._rms(noise) * (10.0 ** (snr_db / 20.0))))
    return audiomae._normalize_audio(result + noise)


def apply_weak_waveform_noise(waveform: torch.Tensor, seed: int) -> torch.Tensor:
    """Add only high-SNR recording noise without altering event structure."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
    snr_db = rng.uniform(35.0, 45.0)
    noise = noise * (
        audiomae._rms(waveform) /
        (audiomae._rms(noise) * (10.0 ** (snr_db / 20.0)))
    )
    return audiomae._normalize_audio(waveform + noise)


class TemplatePairDataset(Dataset):
    """One event template per item, rendered into two nuisance-distinct views."""

    def __init__(self, epoch_len: int, seed: int, profile: str = "templatequeue") -> None:
        self.epoch_len = epoch_len
        self.seed = seed
        self.epoch = 0
        if profile not in ("templatequeue", "minimal", "weak_noise"):
            raise ValueError(f"Unsupported template pair profile: {profile}")
        self.profile = profile

    def __len__(self) -> int:
        return self.epoch_len

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        template_seed = (
            self.seed * 1_000_003 + self.epoch * 10_000_019 + int(index)
        ) % (2**31 - 1)
        with temporary_rng_state(template_seed):
            template = audiomae.advanced_physics_synth(sample_synth_f0())
        if self.profile == "templatequeue":
            first = apply_template_nuisance(template, template_seed + 1)
            second = apply_template_nuisance(template, template_seed + 2)
        elif self.profile == "weak_noise":
            first = apply_weak_waveform_noise(template, template_seed + 1)
            second = apply_weak_waveform_noise(template, template_seed + 2)
        else:
            first = template
            second = template
        return waveform_to_fbank(first), waveform_to_fbank(second), torch.tensor(0)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def is_encoder_key(key: str) -> bool:
    return not any(token in key for token in ENCODER_EXCLUDE)


def encoder_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return bare encoder keys accepted by the existing refer scripts."""
    state = model.state_dict()
    exported = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("backbone."):
            key = key[9:]
        if is_encoder_key(key):
            exported[key] = value.detach().cpu().clone()
    return exported


class ViTFeatureEncoder(nn.Module):
    """AudioMAE encoder with the downstream-compatible pooling path."""

    def __init__(self, model_size: str, freeze_patch_embed: bool = True) -> None:
        super().__init__()
        self.backbone = audiomae.build_model(model_size)
        self.output_dim = int(self.backbone.cls_token.shape[-1])
        if freeze_patch_embed:
            for parameter in self.backbone.patch_embed.parameters():
                parameter.requires_grad = False
        # Contrastive/BYOL paths never reconstruct patches. Removing the MAE
        # decoder avoids unused DDP parameters and reduces model memory.
        for name in (
            "decoder_embed",
            "mask_token",
            "decoder_pos_embed",
            "decoder_blocks",
            "decoder_norm",
            "decoder_pred",
        ):
            delattr(self.backbone, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        tokens = backbone.patch_embed(x)
        base_height, base_width = backbone.patch_embed.patch_hw
        projection = backbone.patch_embed.proj
        patch_height = (x.shape[2] - projection.kernel_size[0]) // projection.stride[0] + 1
        patch_width = (x.shape[3] - projection.kernel_size[1]) // projection.stride[1] + 1
        if patch_height > base_height or patch_width > base_width:
            raise ValueError(
                f"Input patch grid {(patch_height, patch_width)} exceeds "
                f"base grid {(base_height, base_width)}"
            )
        patch_positions = backbone.pos_embed[:, 1:, :].reshape(
            1, base_height, base_width, -1
        )[:, :patch_height, :patch_width, :].reshape(1, -1, self.output_dim)
        cls_token = backbone.cls_token + backbone.pos_embed[:, :1, :]
        tokens = tokens + patch_positions
        tokens = torch.cat((cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        for block in backbone.blocks:
            tokens = block(tokens)
        tokens = backbone.norm(tokens)
        return tokens[:, 1:, :].mean(dim=1)

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        return encoder_state_dict(self.backbone)


def make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    last_bn: bool,
    bias: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for layer_idx in range(num_layers):
        dim_in = input_dim if layer_idx == 0 else hidden_dim
        dim_out = output_dim if layer_idx == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(dim_in, dim_out, bias=bias))
        if layer_idx < num_layers - 1:
            layers.append(nn.BatchNorm1d(dim_out))
            layers.append(nn.ReLU(inplace=True))
        elif last_bn:
            layers.append(nn.BatchNorm1d(dim_out, affine=False))
    return nn.Sequential(*layers)


@torch.no_grad()
def update_ema(target: nn.Module, online: nn.Module, momentum: float) -> None:
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.mul_(momentum).add_(online_param, alpha=1.0 - momentum)


def all_gather_no_grad(tensor: torch.Tensor, ctx: DistributedContext) -> torch.Tensor:
    if ctx.world_size == 1:
        return tensor
    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


@torch.no_grad()
def representation_diagnostics(
    features: torch.Tensor,
    projections: torch.Tensor,
    positive_targets: torch.Tensor,
    ctx: DistributedContext,
) -> Dict[str, torch.Tensor]:
    global_features = all_gather_no_grad(features.detach(), ctx)
    global_projections = all_gather_no_grad(projections.detach(), ctx)
    global_targets = all_gather_no_grad(positive_targets.detach(), ctx)
    normalized = torch_f.normalize(global_projections.float(), dim=1)
    targets = torch_f.normalize(global_targets.float(), dim=1)
    similarity = normalized @ normalized.T
    count = similarity.shape[0]
    off_diagonal = (
        (similarity.sum() - similarity.diagonal().sum()) / max(1, count * (count - 1))
    )
    return {
        "feature_std": global_features.float().std(dim=0).mean(),
        "projection_std": normalized.std(dim=0).mean(),
        "positive_cosine": (normalized * targets).sum(dim=1).mean(),
        "offdiag_cosine": off_diagonal,
    }


class MoCoV3Audio(nn.Module):
    """MoCo v3 objective, adapted only at the audio input/ViT boundary."""

    def __init__(
        self,
        model_size: str,
        projection_dim: int,
        hidden_dim: int,
        temperature: float,
        queue_size: int = 0,
        train_patch_embed: bool = False,
    ) -> None:
        super().__init__()
        self.online = ViTFeatureEncoder(
            model_size, freeze_patch_embed=not train_patch_embed
        )
        self.target = copy.deepcopy(self.online)
        self.online_projector = make_mlp(
            self.online.output_dim, hidden_dim, projection_dim, 3, last_bn=True
        )
        self.target_projector = copy.deepcopy(self.online_projector)
        self.predictor = make_mlp(projection_dim, hidden_dim, projection_dim, 2, last_bn=True)
        self.temperature = temperature
        self.queue_size = queue_size
        if queue_size > 0:
            self.register_buffer("queue", torch_f.normalize(torch.randn(queue_size, projection_dim), dim=1))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        for module in (self.target, self.target_projector):
            for parameter in module.parameters():
                parameter.requires_grad = False

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        update_ema(self.target, self.online, momentum)
        update_ema(self.target_projector, self.online_projector, momentum)

    def contrastive_loss(
        self, query: torch.Tensor, key: torch.Tensor, ctx: DistributedContext
    ) -> torch.Tensor:
        query = torch_f.normalize(query, dim=1)
        key = torch_f.normalize(key, dim=1)
        keys = all_gather_no_grad(key, ctx)
        candidates = keys if self.queue_size == 0 else torch.cat((keys, self.queue.detach()), dim=0)
        logits = torch.einsum("nc,mc->nm", query, candidates) / self.temperature
        labels = torch.arange(query.shape[0], device=query.device)
        labels += query.shape[0] * ctx.rank
        return torch_f.cross_entropy(logits, labels) * (2.0 * self.temperature)

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor, ctx: DistributedContext) -> None:
        if self.queue_size == 0:
            return
        keys = all_gather_no_grad(torch_f.normalize(keys, dim=1), ctx)
        if self.queue_size % keys.shape[0] != 0:
            raise ValueError(
                f"moco queue size {self.queue_size} must divide global key batch {keys.shape[0]}"
            )
        pointer = int(self.queue_ptr.item())
        self.queue[pointer:pointer + keys.shape[0]] = keys
        self.queue_ptr[0] = (pointer + keys.shape[0]) % self.queue_size

    def forward(
        self, view_one: torch.Tensor, view_two: torch.Tensor, momentum: float, ctx: DistributedContext
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        online_one = self.online(view_one)
        online_two = self.online(view_two)
        query_one = self.predictor(self.online_projector(online_one))
        query_two = self.predictor(self.online_projector(online_two))
        with torch.no_grad():
            self.update_target(momentum)
            key_one = self.target_projector(self.target(view_one))
            key_two = self.target_projector(self.target(view_two))
        loss = self.contrastive_loss(query_one, key_two, ctx)
        loss = loss + self.contrastive_loss(query_two, key_one, ctx)
        stats = representation_diagnostics(online_one, query_one, key_two, ctx)
        self.enqueue(torch.cat((key_one, key_two), dim=0), ctx)
        return loss, stats


class BYOLAudioViT(nn.Module):
    """BYOL objective with the shared AudioMAE ViT-base encoder."""

    def __init__(self, model_size: str, projection_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.online = ViTFeatureEncoder(model_size)
        self.target = copy.deepcopy(self.online)
        self.online_projector = make_mlp(
            self.online.output_dim,
            hidden_dim,
            projection_dim,
            2,
            last_bn=False,
            bias=True,
        )
        self.target_projector = copy.deepcopy(self.online_projector)
        self.predictor = make_mlp(
            projection_dim, hidden_dim, projection_dim, 2, last_bn=False, bias=True
        )
        for module in (self.target, self.target_projector):
            for parameter in module.parameters():
                parameter.requires_grad = False

    @staticmethod
    def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = torch_f.normalize(prediction, dim=-1)
        target = torch_f.normalize(target, dim=-1)
        return (2.0 - 2.0 * (prediction * target).sum(dim=-1)).mean()

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        update_ema(self.target, self.online, momentum)
        update_ema(self.target_projector, self.online_projector, momentum)

    def forward(
        self, view_one: torch.Tensor, view_two: torch.Tensor, ctx: DistributedContext
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        online_one = self.online(view_one)
        online_two = self.online(view_two)
        prediction_one = self.predictor(self.online_projector(online_one))
        prediction_two = self.predictor(self.online_projector(online_two))
        with torch.no_grad():
            target_one = self.target_projector(self.target(view_one))
            target_two = self.target_projector(self.target(view_two))
        loss = 0.5 * (
            self.loss_fn(prediction_one, target_two)
            + self.loss_fn(prediction_two, target_one)
        )
        stats = representation_diagnostics(online_one, prediction_one, target_two, ctx)
        return loss, stats


class EventPreservingViews(nn.Module):
    """Two full-length views that preserve source and event composition."""

    def __init__(
        self,
        mask_ratio: float = 0.2,
        log_gain: float = 0.1,
        noise_std: float = 0.02,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("view mask ratio must be in [0, 1)")
        if log_gain < 0.0 or noise_std < 0.0:
            raise ValueError("view log gain and noise std must be non-negative")
        self.mask_ratio = mask_ratio
        self.log_gain = log_gain
        self.noise_std = noise_std
        self.patch_size = patch_size

    def _patch_mask(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, time_steps, mel_bins = x.shape
        if time_steps % self.patch_size or mel_bins % self.patch_size:
            raise ValueError(
                f"Input {(time_steps, mel_bins)} is not divisible by patch size "
                f"{self.patch_size}"
            )
        grid_time = time_steps // self.patch_size
        grid_mel = mel_bins // self.patch_size
        mask = torch.rand(
            batch, 1, grid_time, grid_mel, device=x.device
        ) < self.mask_ratio
        mask = mask.repeat_interleave(self.patch_size, dim=2)
        mask = mask.repeat_interleave(self.patch_size, dim=3)
        return x.masked_fill(mask, 0.0)

    def _view(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        gain = (
            torch.rand(batch, 1, 1, 1, device=x.device) * 2.0 - 1.0
        ) * self.log_gain
        noise_scale = (
            torch.rand(batch, 1, 1, 1, device=x.device) * self.noise_std
        )
        augmented = x + gain + torch.randn_like(x) * noise_scale
        return self._patch_mask(augmented)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._view(x), self._view(x)


class TemplateQueueViews(EventPreservingViews):
    """Stronger full-clip audio views without independently dropping events."""

    def __init__(self, use_log_mixup: bool = True) -> None:
        super().__init__(mask_ratio=0.10, log_gain=0.35, noise_std=0.03)
        self.use_log_mixup = use_log_mixup

    @staticmethod
    def _resize_crop(x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        output = []
        for item in x:
            canvas_height = int(height * 1.20)
            canvas_width = int(width * 1.20)
            canvas = item.new_zeros(channels, canvas_height, canvas_width)
            top = (canvas_height - height) // 2
            left = (canvas_width - width) // 2
            canvas[:, top:top + height, left:left + width] = item
            scale_h = float(torch.empty((), device=x.device).uniform_(0.85, 1.15))
            scale_w = float(torch.empty((), device=x.device).uniform_(0.85, 1.15))
            crop_h = max(1, min(canvas_height, int(height * scale_h)))
            crop_w = max(1, min(canvas_width, int(width * scale_w)))
            start_h = int(torch.randint(canvas_height - crop_h + 1, (), device=x.device))
            start_w = int(torch.randint(canvas_width - crop_w + 1, (), device=x.device))
            crop = canvas[:, start_h:start_h + crop_h, start_w:start_w + crop_w]
            output.append(
                torch_f.interpolate(
                    crop.unsqueeze(0), size=(height, width), mode="bicubic", align_corners=True
                ).squeeze(0)
            )
        return torch.stack(output)

    @staticmethod
    def _log_mixup(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 2:
            return x
        active = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < 0.20
        alpha = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 0.20
        partner = x[torch.randperm(x.shape[0], device=x.device)]
        mixed = torch.logaddexp(x + torch.log1p(-alpha), partner + torch.log(alpha.clamp_min(1e-6)))
        return torch.where(active, mixed, x)

    def _view(self, x: torch.Tensor) -> torch.Tensor:
        augmented = self._resize_crop(x)
        batch = augmented.shape[0]
        head = torch.rand(batch, 1, 1, 1, device=x.device) * 0.50 - 0.25
        tail = torch.rand(batch, 1, 1, 1, device=x.device) * 0.50 - 0.25
        ramp = torch.linspace(0.0, 1.0, augmented.shape[2], device=x.device).view(1, 1, -1, 1)
        augmented = augmented + head * (1.0 - ramp) + tail * ramp
        if self.use_log_mixup:
            augmented = self._log_mixup(augmented)
        return super()._view(augmented)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._view(first), self._view(second)


class WeakTemplateViews(nn.Module):
    """AudioMAE-aligned views that preserve the complete event spectrogram."""

    def __init__(self, max_gain_db: float) -> None:
        super().__init__()
        self.max_gain = (max_gain_db * math.log(10.0) / 10.0) / (4.56 * 2.0)

    def _view(self, x: torch.Tensor) -> torch.Tensor:
        gain = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(
            -self.max_gain, self.max_gain
        )
        return x + gain

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._view(first), self._view(second)


def build_ssl_model(args: argparse.Namespace) -> nn.Module:
    if args.method == "audiomae":
        return audiomae.build_model(args.model_size)
    if args.method in MOCO_METHODS:
        return MoCoV3Audio(
            args.model_size,
            args.projection_dim,
            args.projection_hidden_dim,
            args.temperature,
            args.moco_queue_size,
            args.train_patch_embed,
        )
    if args.method == "byol_vit":
        return BYOLAudioViT(
            args.model_size, args.projection_dim, args.projection_hidden_dim
        )
    raise ValueError(f"Unsupported method: {args.method}")


def current_encoder(model: nn.Module, method: str) -> nn.Module:
    model = unwrap(model)
    if method in (*MOCO_METHODS, "byol_vit"):
        return model.online.backbone
    return model


def make_logger(output_dir: Path, is_main: bool) -> logging.Logger:
    logger = logging.getLogger(f"ssl_compare.{output_dir.name}")
    logger.setLevel(logging.INFO if is_main else logging.WARNING)
    logger.handlers.clear()
    if is_main:
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        file_handler = logging.FileHandler(output_dir / "train.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def reduce_mean(value: torch.Tensor, ctx: DistributedContext) -> float:
    reduced = value.detach().float().clone()
    if ctx.world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced.div_(ctx.world_size)
    return float(reduced.item())


def reduce_max(value: torch.Tensor, ctx: DistributedContext) -> float:
    reduced = value.detach().float().clone()
    if ctx.world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return float(reduced.item())


def method_optimizer_config(
    args: argparse.Namespace, ctx: DistributedContext
) -> Tuple[float, float, Tuple[float, float]]:
    if args.method in MOCO_METHODS:
        global_batch = args.batch_size * ctx.world_size
        default_lr = args.moco_base_lr * global_batch / 256.0
        return (
            args.lr if args.lr > 0 else default_lr,
            args.weight_decay if args.weight_decay >= 0 else 0.1,
            (0.9, 0.999),
        )
    if args.method == "byol_vit":
        global_batch = args.batch_size * ctx.world_size
        default_lr = args.byol_base_lr * global_batch / 256.0
        return (
            args.lr if args.lr > 0 else default_lr,
            args.weight_decay if args.weight_decay >= 0 else 0.1,
            (0.9, 0.95),
        )
    return (
        args.lr if args.lr > 0 else 1e-4,
        args.weight_decay if args.weight_decay >= 0 else 0.05,
        (0.9, 0.95),
    )


def ssl_parameter_groups(
    model: nn.Module, weight_decay: float
) -> list[Dict[str, Any]]:
    regularized = []
    unregularized = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            unregularized.append(parameter)
        else:
            regularized.append(parameter)
    return [
        {
            "params": regularized,
            "weight_decay": weight_decay,
        },
        {
            "params": unregularized,
            "weight_decay": 0.0,
        },
    ]


def adjust_moco_learning_rate(
    optimizer: optim.Optimizer,
    step: int,
    steps_per_epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    peak_lr: float,
) -> float:
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)
    total_steps = max(1, total_epochs * steps_per_epoch)
    if warmup_steps > 0 and step < warmup_steps:
        learning_rate = peak_lr * (step + 1) / warmup_steps
    else:
        cosine_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / cosine_steps))
        learning_rate = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    return learning_rate


def local_rng_state(generator: torch.Generator, device: torch.device) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "loader_generator": generator.get_state(),
    }


def gather_rng_states(
    generator: torch.Generator, ctx: DistributedContext
) -> list[Dict[str, Any]]:
    state = local_rng_state(generator, ctx.device)
    if ctx.world_size == 1:
        return [state]
    states: list[Dict[str, Any] | None] = [None] * ctx.world_size
    dist.all_gather_object(states, state)
    return [item for item in states if item is not None]


def restore_rng_state(
    states: list[Dict[str, Any]], generator: torch.Generator, ctx: DistributedContext
) -> None:
    if ctx.rank >= len(states):
        raise ValueError(f"Checkpoint has {len(states)} RNG states for rank {ctx.rank}")
    state = states[ctx.rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if ctx.device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device=ctx.device)
    generator.set_state(state["loader_generator"])


def save_checkpoint(
    path: Path,
    encoder: nn.Module,
    method: str,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    synth_config: Dict[str, object],
    synth_hash: str,
    optimizer: optim.Optimizer | None = None,
    scaler: object | None = None,
    metrics: Dict[str, float] | None = None,
    training_model: nn.Module | None = None,
    rng_states: list[Dict[str, Any]] | None = None,
) -> None:
    checkpoint = {
        "format": "ssl_compare_encoder_v2",
        "method": method,
        "variant": args.model_size,
        "model_state": encoder_state_dict(encoder),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics or {},
        "config": {
            "encoder_arch": f"audiomae_vit_{args.model_size}_patch16",
            "img_size": (1024, 128),
            "patch_size": 16,
            "embed_dim": int(encoder.cls_token.shape[-1]),
            "depth": len(encoder.blocks),
            "num_heads": int(encoder.blocks[0].attn.num_heads),
            "synth_config_path": args.synth_config,
            "synth_config_hash": synth_hash,
            "synth_config": synth_config,
            "args": vars(args),
        },
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        checkpoint["scaler_state"] = scaler.state_dict()
    if training_model is not None:
        checkpoint["training_model_state"] = {
            key.removeprefix("module."): value.detach().cpu().clone()
            for key, value in unwrap(training_model).state_dict().items()
        }
    if rng_states is not None:
        checkpoint["rng_states"] = rng_states
    torch.save(checkpoint, path)


def find_resume_checkpoint(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("checkpoint_epoch_*.pth"))
    return candidates[-1] if candidates else None


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: object,
    method: str,
    generator: torch.Generator,
    ctx: DistributedContext,
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "training_model_state" not in checkpoint:
        raise ValueError(f"Checkpoint cannot resume full {method} training: {path}")
    unwrap(model).load_state_dict(checkpoint["training_model_state"], strict=True)
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if checkpoint.get("scaler_state") is not None and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    if checkpoint.get("rng_states") is not None:
        restore_rng_state(checkpoint["rng_states"], generator, ctx)
    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def configure_synthesizer(path: str) -> Tuple[Dict[str, object], str]:
    config, config_hash = audiomae._load_synth_config(path)
    audiomae.set_synth_config(config, config_hash)
    return config, config_hash


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()
    seed_everything(args.seed, ctx.rank)
    output_dir = Path(args.output_dir)
    final_path = output_dir / "encoder_final.pth"

    try:
        if ctx.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        distributed_barrier(ctx)
        if final_path.exists() and not (args.resume or args.overwrite):
            raise FileExistsError(f"Refusing to overwrite completed run: {final_path}")

        logger = make_logger(output_dir, ctx.is_main)
        synth_config, synth_hash = configure_synthesizer(args.synth_config)
        if ctx.is_main:
            logger.info("method=%s seed=%s world_size=%s", args.method, args.seed, ctx.world_size)
            logger.info("synth=%s hash=%s", synth_config.get("name"), synth_hash)

        model = build_ssl_model(args).to(ctx.device)
        if ctx.world_size > 1:
            if ctx.device.type == "cuda" and args.method in (*MOCO_METHODS, "byol_vit"):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None)

        if args.method in TEMPLATEQUEUE_METHODS:
            dataset: Dataset = TemplatePairDataset(args.epoch_len, args.seed, "templatequeue")
        elif args.method == "mocov3_minimalpair":
            dataset = TemplatePairDataset(args.epoch_len, args.seed, "minimal")
        elif args.method == "mocov3_weaknoisepair":
            dataset = TemplatePairDataset(args.epoch_len, args.seed, "weak_noise")
        else:
            dataset = audiomae.PhysicsDataset(epoch_len=args.epoch_len)
        sampler = DistributedSampler(
            dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True, seed=args.seed, drop_last=True
        ) if ctx.world_size > 1 else None
        generator = torch.Generator()
        generator.manual_seed(args.seed + ctx.rank)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=ctx.device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=generator,
            # Recreate workers each epoch so a saved loader RNG state is enough
            # to reproduce the next epoch after resume.
            persistent_workers=False,
        )
        learning_rate, weight_decay, betas = method_optimizer_config(args, ctx)
        args.effective_lr = learning_rate
        args.effective_weight_decay = weight_decay
        args.effective_betas = betas
        if args.method in (*MOCO_METHODS, "byol_vit"):
            optimizer = optim.AdamW(
                ssl_parameter_groups(model, weight_decay),
                lr=learning_rate,
                betas=betas,
            )
        else:
            optimizer = optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay,
            )
        if ctx.is_main:
            logger.info(
                "optimizer=AdamW lr=%.8g weight_decay=%.4g betas=%s global_batch=%s",
                learning_rate, weight_decay, betas, args.batch_size * ctx.world_size,
            )
        amp_enabled = args.amp and ctx.device.type == "cuda"
        scaler = audiomae.make_grad_scaler(enabled=amp_enabled)
        if args.method in TEMPLATEQUEUE_METHODS:
            views: nn.Module | None = TemplateQueueViews(
                use_log_mixup=args.method == "mocov3_templatequeue"
            ).to(ctx.device)
        elif args.method == "mocov3_minimalpair":
            views = WeakTemplateViews(max_gain_db=1.5).to(ctx.device)
        elif args.method == "mocov3_weaknoisepair":
            views = WeakTemplateViews(max_gain_db=3.0).to(ctx.device)
        elif args.method in ("mocov3", "byol_vit"):
            views: nn.Module | None = EventPreservingViews(
                mask_ratio=args.view_mask_ratio,
                log_gain=args.view_log_gain,
                noise_std=args.view_noise_std,
            ).to(ctx.device)
        else:
            views = None

        start_epoch, global_step = 0, 0
        if args.resume:
            resume_path = find_resume_checkpoint(output_dir)
            if resume_path is not None:
                start_epoch, global_step = load_resume_checkpoint(
                    resume_path, model, optimizer, scaler, args.method, generator, ctx
                )
                if ctx.is_main:
                    logger.info("resumed from %s at epoch=%s step=%s", resume_path, start_epoch, global_step)

        total_steps = max(1, args.epochs * len(loader))
        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            if isinstance(dataset, TemplatePairDataset):
                dataset.set_epoch(epoch)
            if ctx.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(ctx.device)
            model.train()
            loss_total = 0.0
            diagnostic_totals: Dict[str, float] = {}
            step_count = 0
            current_lr = learning_rate
            current_momentum = 0.0
            iterator: Iterable = tqdm(loader, disable=not ctx.is_main, desc=f"{args.method} {epoch + 1}/{args.epochs}")
            for batch in iterator:
                if args.method in TEMPLATE_PAIR_METHODS:
                    image_one, image_two, _ = batch
                    image_one = image_one.to(ctx.device, non_blocking=True)
                    image_two = image_two.to(ctx.device, non_blocking=True)
                    images = None
                else:
                    images, _ = batch
                    images = images.to(ctx.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                if args.method in (*MOCO_METHODS, "byol_vit"):
                    current_lr = adjust_moco_learning_rate(
                        optimizer, global_step, len(loader), args.epochs,
                        args.warmup_epochs, learning_rate,
                    )
                if views is not None:
                    with torch.no_grad():
                        if args.method in TEMPLATE_PAIR_METHODS:
                            view_one, view_two = views(image_one.float(), image_two.float())
                        else:
                            if images is None:
                                raise RuntimeError("missing images for SSL views")
                            view_one, view_two = views(images.float())
                with audiomae.autocast_cuda(enabled=amp_enabled):
                    if args.method == "audiomae":
                        loss, _, _ = model(images, mask_ratio=args.mask_ratio)
                        stats = {}
                    else:
                        raw_model = unwrap(model)
                        if args.method in MOCO_METHODS:
                            progress = global_step / max(1, total_steps - 1)
                            current_momentum = 1.0 - (
                                1.0 - args.moco_base_m
                            ) * (math.cos(math.pi * progress) + 1.0) / 2.0
                            loss, stats = model(
                                view_one, view_two, current_momentum, ctx
                            )
                        else:
                            loss, stats = model(view_one, view_two, ctx)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if args.method == "byol_vit":
                    progress = global_step / max(1, total_steps - 1)
                    current_momentum = 1.0 - (
                        1.0 - args.byol_base_m
                    ) * (math.cos(math.pi * progress) + 1.0) / 2.0
                    raw_model.update_target(current_momentum)

                loss_value = reduce_mean(loss, ctx)
                loss_total += loss_value
                for name, value in stats.items():
                    diagnostic_totals[name] = (
                        diagnostic_totals.get(name, 0.0) + reduce_mean(value, ctx)
                    )
                step_count += 1
                global_step += 1
                if ctx.is_main:
                    iterator.set_postfix(loss=f"{loss_value:.4f}")
                if args.max_steps_per_epoch and step_count >= args.max_steps_per_epoch:
                    break

            metrics = {
                "loss": loss_total / max(1, step_count),
                "steps_in_epoch": float(step_count),
                "learning_rate": current_lr,
            }
            if args.method in (*MOCO_METHODS, "byol_vit"):
                metrics["teacher_momentum"] = current_momentum
                metrics["weight_decay"] = weight_decay
            if ctx.device.type == "cuda":
                peak_memory_gib = torch.tensor(
                    torch.cuda.max_memory_allocated(ctx.device) / 1024**3,
                    device=ctx.device,
                )
                metrics["peak_memory_gib"] = reduce_max(peak_memory_gib, ctx)
            metrics.update({
                name: total / max(1, step_count)
                for name, total in diagnostic_totals.items()
            })
            should_save = (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs
            rng_states = gather_rng_states(generator, ctx) if should_save else None
            if ctx.is_main:
                logger.info("epoch=%s/%s metrics=%s", epoch + 1, args.epochs, json.dumps(metrics, sort_keys=True))
                if should_save:
                    save_checkpoint(
                        output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth",
                        current_encoder(model, args.method), args.method, args, epoch + 1, global_step,
                        synth_config, synth_hash, optimizer, scaler, metrics,
                        training_model=model, rng_states=rng_states,
                    )

        if ctx.is_main:
            save_checkpoint(
                final_path, current_encoder(model, args.method), args.method, args, args.epochs, global_step,
                synth_config, synth_hash, metrics=metrics,
            )
            with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
                json.dump({"args": vars(args), "metrics": metrics, "world_size": ctx.world_size}, handle, indent=2)
            logger.info("completed checkpoint=%s", final_path)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
