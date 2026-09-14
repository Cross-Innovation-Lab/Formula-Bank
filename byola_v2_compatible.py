#!/usr/bin/env python3
"""DDP BYOL-A v2 pre-training on the shared synthetic audio distribution.

The AudioNTT2022 encoder and all view transforms come from the checked-out
BYOL-A v2 implementation. The data budget is controlled to match the ViT
comparison: fresh synthetic waveforms are generated online and cropped to the
official 0.95-second BYOL-A training unit.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as torch_f
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

import pretrain as audiomae


REPO_ROOT = Path(__file__).resolve().parent
BYOLA_V2_ROOT = REPO_ROOT / "external" / "byol-a" / "v2"


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
    parser = argparse.ArgumentParser(description="BYOL-A v2 AudioNTT2022 pre-training")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synth-config", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--epoch-len", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--feature-dim", type=int, default=3072)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--projection-hidden-dim", type=int, default=4096)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--norm-samples", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
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
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            init_method="env://",
        )
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


def require_dependencies():
    try:
        import nnAudio.features  # type: ignore
    except ImportError as error:
        raise RuntimeError("BYOL-A v2 requires nnAudio in the active environment.") from error
    if not BYOLA_V2_ROOT.is_dir():
        raise FileNotFoundError(f"Missing official BYOL-A v2 checkout: {BYOLA_V2_ROOT}")
    sys.path.insert(0, str(BYOLA_V2_ROOT))
    from byol_a2.models import AudioNTT2022  # type: ignore
    return nnAudio.features, AudioNTT2022


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False


def seed_worker(_: int) -> None:
    value = torch.initial_seed() % 2**32
    random.seed(value)
    np.random.seed(value)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class SyntheticWaveCropDataset(Dataset):
    """Generate the same base waveform stream as PhysicsDataset without PCM I/O."""

    def __init__(self, epoch_len: int, unit_samples: int, crop_seed: int) -> None:
        self.epoch_len = epoch_len
        self.unit_samples = unit_samples
        self.crop_seed = crop_seed

    def __len__(self) -> int:
        return self.epoch_len

    def __getitem__(self, index: int) -> torch.Tensor:
        f0_min, f0_max = map(
            float, audiomae.SYNTH_CONFIG.get("f0_range_hz", [50.0, 2000.0])
        )
        distribution = str(
            audiomae.SYNTH_CONFIG.get("f0_distribution", "uniform")
        ).replace("-", "_")
        if distribution == "uniform":
            f0 = np.random.uniform(f0_min, f0_max)
        elif distribution == "log_uniform":
            f0 = np.exp(np.random.uniform(np.log(f0_min), np.log(f0_max)))
        else:
            raise ValueError(f"Unsupported f0_distribution: {distribution}")
        waveform = audiomae.advanced_physics_synth(float(f0))
        if waveform.numel() < self.unit_samples:
            repeats = math.ceil(self.unit_samples / waveform.numel())
            waveform = waveform.repeat(repeats)
        # A local RNG prevents crop selection from changing the next synthesized
        # waveform in the worker's shared Python/NumPy/Torch RNG stream.
        crop_rng = random.Random(self.crop_seed * 1_000_003 + int(index))
        start = crop_rng.randint(0, waveform.numel() - self.unit_samples)
        return waveform[start:start + self.unit_samples].float()


class OriginalMixupBYOLA(nn.Module):
    """Dependency-light equivalent of external BYOL-A v2 MixupBYOLA."""

    def __init__(self, ratio: float = 0.2, n_memory: int = 2048) -> None:
        super().__init__()
        self.ratio = ratio
        self.n = n_memory
        self.memory_bank: list[torch.Tensor] = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        alpha = self.ratio * np.random.random()
        if self.memory_bank:
            background = self.memory_bank[np.random.randint(len(self.memory_bank))]
            mixed = torch.log(
                (1.0 - alpha) * image.exp()
                + alpha * background.exp()
                + torch.finfo(image.dtype).eps
            )
        else:
            mixed = image
        # The official implementation stores the original input, not `mixed`.
        self.memory_bank = (self.memory_bank + [image])[-self.n:]
        return mixed.float()


class OriginalRandomResizeCrop(nn.Module):
    """Exact v2 virtual-canvas resize/crop parameters."""

    def __init__(
        self,
        virtual_crop_scale: Tuple[float, float] = (1.0, 1.5),
        freq_scale: Tuple[float, float] = (0.6, 1.5),
        time_scale: Tuple[float, float] = (0.6, 1.5),
    ) -> None:
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        source_frequency, source_time = image.shape[-2:]
        canvas_frequency = int(source_frequency * self.virtual_crop_scale[0])
        canvas_time = int(source_time * self.virtual_crop_scale[1])
        crop_frequency = int(
            np.clip(
                int(np.random.uniform(*self.freq_scale) * source_frequency),
                1,
                canvas_frequency,
            )
        )
        crop_time = int(
            np.clip(
                int(np.random.uniform(*self.time_scale) * source_time),
                1,
                canvas_time,
            )
        )
        canvas = image.new_zeros((image.shape[0], canvas_frequency, canvas_time)).float()
        frequency_offset = (canvas_frequency - source_frequency) // 2
        time_offset = (canvas_time - source_time) // 2
        canvas[
            :,
            frequency_offset:frequency_offset + source_frequency,
            time_offset:time_offset + source_time,
        ] = image
        crop_frequency_offset = (
            random.randint(0, canvas_frequency - crop_frequency)
            if canvas_frequency > crop_frequency
            else 0
        )
        crop_time_offset = (
            random.randint(0, canvas_time - crop_time)
            if canvas_time > crop_time
            else 0
        )
        crop = canvas[
            :,
            crop_frequency_offset:crop_frequency_offset + crop_frequency,
            crop_time_offset:crop_time_offset + crop_time,
        ]
        return torch_f.interpolate(
            crop.unsqueeze(0),
            size=image.shape[-2:],
            mode="bicubic",
            align_corners=True,
        ).squeeze(0).float()


class OriginalRandomLinearFader(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        head, tail = 2.0 * np.random.rand(2) - 1.0
        slope = torch.linspace(
            head, tail, image.shape[-1], dtype=image.dtype, device=image.device
        ).reshape(1, 1, -1)
        return image + slope


class OriginalNormalizeBatch(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=(0, 2, 3), keepdim=True)
        std = inputs.std(dim=(0, 2, 3), keepdim=True).clamp_min(
            torch.finfo(inputs.dtype).eps
        )
        return (inputs - mean) / std


class BYOLAViewGenerator:
    """Exact BYOL-A v2 transform order and shared FIFO memory bank."""

    def __init__(self) -> None:
        self.transform = nn.Sequential(
            OriginalMixupBYOLA(ratio=0.2),
            OriginalRandomResizeCrop(
                virtual_crop_scale=(1.0, 1.5),
                freq_scale=(0.6, 1.5),
                time_scale=(0.6, 1.5),
            ),
            OriginalRandomLinearFader(),
        )
        self.post_normalize = OriginalNormalizeBatch()

    @property
    def mixup(self) -> nn.Module:
        return self.transform[0]

    @torch.no_grad()
    def __call__(self, log_mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        first, second = [], []
        for image in log_mel.float():
            first_view = self.transform(image)
            second_view = self.transform(image)
            first.append(first_view)
            second.append(second_view)
        batch_size = len(first)
        paired = self.post_normalize(torch.cat((torch.stack(first), torch.stack(second))))
        return paired[:batch_size], paired[batch_size:]

    def memory_state(self) -> list[torch.Tensor]:
        return [item.detach().cpu().clone() for item in self.mixup.memory_bank]

    def load_memory_state(self, state: list[torch.Tensor], device: torch.device) -> None:
        self.mixup.memory_bank = [item.to(device) for item in state]


class AudioNTTBYOL(nn.Module):
    def __init__(
        self, encoder: nn.Module, feature_dim: int, projection_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.online_projector = self._mlp(feature_dim, hidden_dim, projection_dim)
        self.target_projector = copy.deepcopy(self.online_projector)
        self.predictor = self._mlp(projection_dim, hidden_dim, projection_dim)
        for module in (self.target_encoder, self.target_projector):
            for parameter in module.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
        # Matches the pinned byol-pytorch MLP used by BYOL-A v2.
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = torch_f.normalize(prediction, dim=-1)
        target = torch_f.normalize(target, dim=-1)
        return (2.0 - 2.0 * (prediction * target).sum(dim=-1)).mean()

    def forward(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        online_first = self.online_encoder(first)
        online_second = self.online_encoder(second)
        projection_first = self.online_projector(online_first)
        projection_second = self.online_projector(online_second)
        prediction_first = self.predictor(projection_first)
        prediction_second = self.predictor(projection_second)
        with torch.no_grad():
            target_first = self.target_projector(self.target_encoder(first))
            target_second = self.target_projector(self.target_encoder(second))
        loss = self._loss(prediction_first, target_second)
        loss = loss + self._loss(prediction_second, target_first)
        return loss, online_first, prediction_first

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        pairs = (
            (self.target_encoder, self.online_encoder),
            (self.target_projector, self.online_projector),
        )
        for target, online in pairs:
            for target_param, online_param in zip(target.parameters(), online.parameters()):
                target_param.mul_(momentum).add_(online_param, alpha=1.0 - momentum)


def make_mel(nn_audio_features: Any, device: torch.device) -> nn.Module:
    return nn_audio_features.MelSpectrogram(
        sr=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=160,
        n_mels=64,
        fmin=60,
        fmax=7800,
        center=True,
        power=2,
        verbose=False,
    ).to(device)


def make_logger(output_dir: Path, is_main: bool) -> logging.Logger:
    logger = logging.getLogger(f"byola_v2.{output_dir.name}")
    logger.setLevel(logging.INFO if is_main else logging.WARNING)
    logger.handlers.clear()
    if is_main:
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        for handler in (
            logging.FileHandler(output_dir / "train.log"),
            logging.StreamHandler(),
        ):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


@torch.no_grad()
def compute_norm_stats(
    loader: DataLoader,
    mel: nn.Module,
    device: torch.device,
    global_sample_limit: int,
    ctx: DistributedContext,
) -> Tuple[float, float]:
    local_limit = math.ceil(global_sample_limit / ctx.world_size)
    total = torch.zeros(3, dtype=torch.float64, device=device)
    observed = 0
    for waveforms in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        values = (mel(waveforms.float()) + torch.finfo(torch.float32).eps).log()
        take = min(values.shape[0], local_limit - observed)
        values = values[:take].double()
        total[0] += values.sum()
        total[1] += values.square().sum()
        total[2] += values.numel()
        observed += take
        if observed >= local_limit:
            break
    if ctx.world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    count = max(1.0, float(total[2].item()))
    mean = float(total[0].item()) / count
    variance = max(float(total[1].item()) / count - mean * mean, 1e-12)
    return mean, variance**0.5


@torch.no_grad()
def global_feature_metrics(
    features: torch.Tensor, projections: torch.Tensor, ctx: DistributedContext
) -> Dict[str, float]:
    if ctx.world_size > 1:
        feature_parts = [torch.empty_like(features) for _ in range(ctx.world_size)]
        projection_parts = [torch.empty_like(projections) for _ in range(ctx.world_size)]
        dist.all_gather(feature_parts, features.detach())
        dist.all_gather(projection_parts, projections.detach())
        features = torch.cat(feature_parts)
        projections = torch.cat(projection_parts)
    normalized = torch_f.normalize(projections.float(), dim=1)
    similarity = normalized @ normalized.T
    count = similarity.shape[0]
    offdiag = (
        (similarity.sum() - similarity.diagonal().sum()) / max(1, count * (count - 1))
    )
    return {
        "feature_std": float(features.float().std(dim=0).mean().item()),
        "projection_std": float(normalized.std(dim=0).mean().item()),
        "offdiag_cosine": float(offdiag.item()),
    }


def configure_synthesizer(path: str) -> Tuple[Dict[str, object], str]:
    config, config_hash = audiomae._load_synth_config(path)
    audiomae.set_synth_config(config, config_hash)
    return config, config_hash


def local_state(
    generator: torch.Generator,
    views: BYOLAViewGenerator,
    ctx: DistributedContext,
) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(ctx.device) if ctx.device.type == "cuda" else None,
        "loader_generator": generator.get_state(),
        "mixup_memory": views.memory_state(),
    }


def restore_local_state(
    state: Dict[str, Any],
    generator: torch.Generator,
    views: BYOLAViewGenerator,
    ctx: DistributedContext,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if ctx.device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device=ctx.device)
    generator.set_state(state["loader_generator"])
    views.load_memory_state(state.get("mixup_memory", []), ctx.device)


def checkpoint_path(output_dir: Path, epoch: int) -> Path:
    return output_dir / f"checkpoint_epoch_{epoch:03d}.pth"


def rank_state_path(output_dir: Path, epoch: int, rank: int) -> Path:
    return output_dir / f"checkpoint_epoch_{epoch:03d}_rank_{rank:03d}.pth"


def find_resume_checkpoint(output_dir: Path) -> Path | None:
    paths = [
        path for path in output_dir.glob("checkpoint_epoch_*.pth")
        if "_rank_" not in path.name
    ]
    return sorted(paths)[-1] if paths else None


def save_training_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    global_step: int,
    norm_stats: Tuple[float, float],
    synth_config: Dict[str, object],
    synth_hash: str,
    args: argparse.Namespace,
    metrics: Dict[str, float],
    generator: torch.Generator,
    views: BYOLAViewGenerator,
    ctx: DistributedContext,
) -> None:
    torch.save(
        local_state(generator, views, ctx),
        rank_state_path(output_dir, epoch, ctx.rank),
    )
    distributed_barrier(ctx)
    if ctx.is_main:
        raw_model = unwrap(model)
        torch.save(
            {
                "format": "byola_v2_training_v2",
                "method": "byola_v2",
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "normalization_stats": list(norm_stats),
                "synth_config": synth_config,
                "synth_config_hash": synth_hash,
                "config": vars(args),
                "epoch": epoch,
                "global_step": global_step,
                "metrics": metrics,
            },
            checkpoint_path(output_dir, epoch),
        )
    distributed_barrier(ctx)


def main() -> None:
    args = parse_args()
    ctx = setup_distributed()
    output_dir = Path(args.output_dir)
    final_path = output_dir / "encoder_final.pth"
    try:
        seed_everything(args.seed, ctx.rank)
        if ctx.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        distributed_barrier(ctx)
        if final_path.exists() and not (args.resume or args.overwrite):
            raise FileExistsError(f"Refusing to overwrite completed run: {final_path}")

        logger = make_logger(output_dir, ctx.is_main)
        synth_config, synth_hash = configure_synthesizer(args.synth_config)
        nn_audio_features, audio_ntt_cls = require_dependencies()
        dataset = SyntheticWaveCropDataset(
            args.epoch_len, int(16000 * 0.95), args.seed
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
            if ctx.world_size > 1
            else None
        )
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
            persistent_workers=False,
        )
        norm_generator = torch.Generator()
        norm_generator.manual_seed(args.seed + 100_000 + ctx.rank)
        norm_sampler = (
            DistributedSampler(
                dataset,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=True,
                seed=args.seed + 100_000,
                drop_last=False,
            )
            if ctx.world_size > 1
            else None
        )
        norm_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=norm_sampler,
            shuffle=norm_sampler is None,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=ctx.device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=norm_generator,
            persistent_workers=False,
        )
        mel = make_mel(nn_audio_features, ctx.device)
        resume_path = find_resume_checkpoint(output_dir) if args.resume else None
        resume_checkpoint = (
            torch.load(resume_path, map_location="cpu", weights_only=False)
            if resume_path is not None
            else None
        )
        if resume_checkpoint is not None:
            norm_stats = tuple(map(float, resume_checkpoint["normalization_stats"]))
        else:
            norm_stats = compute_norm_stats(
                norm_loader, mel, ctx.device, args.norm_samples, ctx
            )
        if ctx.is_main:
            logger.info(
                "method=byola_v2 seed=%s world_size=%s global_batch=%s samples=%s",
                args.seed, ctx.world_size, args.batch_size * ctx.world_size,
                args.epoch_len * args.epochs,
            )
            logger.info(
                "synth=%s hash=%s normalization=(%.8f, %.8f)",
                synth_config.get("name"), synth_hash, norm_stats[0], norm_stats[1],
            )

        model = AudioNTTBYOL(
            audio_ntt_cls(n_mels=64, d=args.feature_dim),
            args.feature_dim,
            args.projection_dim,
            args.projection_hidden_dim,
        ).to(ctx.device)
        if ctx.world_size > 1:
            model = DDP(
                model,
                device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
            )
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.lr,
        )
        amp_enabled = args.amp and ctx.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        views = BYOLAViewGenerator()
        start_epoch, global_step = 0, 0
        if resume_checkpoint is not None:
            unwrap(model).load_state_dict(resume_checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            scaler.load_state_dict(resume_checkpoint["scaler_state"])
            start_epoch = int(resume_checkpoint["epoch"])
            global_step = int(resume_checkpoint.get("global_step", 0))
            local_path = rank_state_path(output_dir, start_epoch, ctx.rank)
            restore_local_state(
                torch.load(local_path, map_location="cpu", weights_only=False),
                generator,
                views,
                ctx,
            )
            if ctx.is_main:
                logger.info("resumed checkpoint=%s epoch=%s", resume_path, start_epoch)

        metrics: Dict[str, float] = {}
        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            totals: Dict[str, float] = {
                "loss": 0.0,
                "feature_std": 0.0,
                "projection_std": 0.0,
                "offdiag_cosine": 0.0,
            }
            steps = 0
            iterator = tqdm(
                loader,
                disable=not ctx.is_main,
                desc=f"byola_v2 {epoch + 1}/{args.epochs}",
            )
            for waveforms in iterator:
                waveforms = waveforms.to(ctx.device, non_blocking=True)
                with torch.no_grad():
                    log_mel = (
                        mel(waveforms.float()) + torch.finfo(torch.float32).eps
                    ).log().unsqueeze(1)
                    log_mel = (log_mel - norm_stats[0]) / norm_stats[1]
                    first, second = views(log_mel)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss, features, projections = model(first, second)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                unwrap(model).update_target(args.ema_decay)

                diagnostic = global_feature_metrics(features, projections, ctx)
                loss_tensor = loss.detach().float()
                if ctx.world_size > 1:
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    loss_tensor.div_(ctx.world_size)
                totals["loss"] += float(loss_tensor.item())
                for name, value in diagnostic.items():
                    totals[name] += value
                steps += 1
                global_step += 1
                if ctx.is_main:
                    iterator.set_postfix(loss=f"{float(loss_tensor.item()):.4f}")
                if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                    break

            metrics = {
                name: total / max(1, steps) for name, total in totals.items()
            }
            metrics["steps_in_epoch"] = float(steps)
            should_save = (
                (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs
            )
            if ctx.is_main:
                logger.info(
                    "epoch=%s/%s metrics=%s",
                    epoch + 1,
                    args.epochs,
                    json.dumps(metrics, sort_keys=True),
                )
            if should_save:
                save_training_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    scaler,
                    epoch + 1,
                    global_step,
                    norm_stats,
                    synth_config,
                    synth_hash,
                    args,
                    metrics,
                    generator,
                    views,
                    ctx,
                )

        if ctx.is_main:
            raw_model = unwrap(model)
            torch.save(
                {
                    "format": "byola_v2_encoder_v2",
                    "method": "byola_v2",
                    "model_state": {
                        key: value.detach().cpu()
                        for key, value in raw_model.online_encoder.state_dict().items()
                    },
                    "normalization_stats": list(norm_stats),
                    "synth_config": synth_config,
                    "synth_config_hash": synth_hash,
                    "config": vars(args),
                    "epoch": args.epochs,
                    "global_step": global_step,
                    "metrics": metrics,
                },
                final_path,
            )
            (output_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "args": vars(args),
                        "metrics": metrics,
                        "world_size": ctx.world_size,
                        "normalization_stats": norm_stats,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            logger.info("completed checkpoint=%s", final_path)
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
