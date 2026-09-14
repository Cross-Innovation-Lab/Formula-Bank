#!/usr/bin/env python3
"""Waveform-native full fine-tuning for BYOL-A v2 AudioNTT2022."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import average_precision_score
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from audioset20k_dataset import AudioSet20KTarDataset


SAMPLE_RATE = 16000
TARGET_SAMPLES = int(SAMPLE_RATE * 10.24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full fine-tuning for BYOL-A v2")
    parser.add_argument("--task", required=True, choices=("esc50", "fsd50k", "us8k", "scv2", "audioset20k"))
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data-path", default="")
    parser.add_argument("--norm-samples", type=int, default=10000)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--max-train-steps", type=int, default=0,
                        help="Nonzero only for smoke tests.")
    parser.add_argument("--max-val-steps", type=int, default=0,
                        help="Nonzero only for smoke tests.")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_: int) -> None:
    value = torch.initial_seed() % 2**32
    random.seed(value)
    np.random.seed(value)


def make_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"byola_v2_finetune.{output_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    for handler in (
        logging.FileHandler(output_dir / "train.log"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class WaveformAugment:
    def __init__(self, task: str) -> None:
        self.task = task

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.task == "esc50":
            # Match the ESC formal waveform augmentation exactly.
            if random.random() < 0.5:
                waveform = waveform * random.uniform(0.8, 1.2)
            if random.random() < 0.5:
                waveform = waveform + torch.randn_like(waveform) * random.uniform(0.001, 0.015)
            if random.random() < 0.5:
                waveform = torch.roll(
                    waveform, shifts=random.randint(0, waveform.shape[-1]), dims=-1
                )
            return waveform
        if random.random() < 0.5:
            waveform = waveform * random.uniform(0.7, 1.3)
        if random.random() < 0.5:
            waveform = waveform + torch.randn_like(waveform) * random.uniform(0.0005, 0.005)
        if random.random() < 0.5:
            waveform = torch.roll(
                waveform, shifts=random.randint(0, waveform.shape[-1] - 1), dims=-1
            )
        if random.random() < 0.5:
            waveform = -waveform
        return waveform


class WaveformTaskDataset(Dataset):
    """Expose repository task splits as fixed-length 16 kHz waveforms."""

    def __init__(
        self,
        task: str,
        root: str,
        fold: int,
        train: bool,
        augment: bool,
    ) -> None:
        self.task = task
        self.train = train
        self.augment = augment
        self.wave_augment = WaveformAugment(task)
        self.records: list[Tuple[str, torch.Tensor | int]] = []
        self.audioset = None

        if task == "esc50":
            from esc_refer import ESC50Dataset

            base = ESC50Dataset(root, fold=fold, train=train)
            for _, row in base.df.iterrows():
                self.records.append(
                    (
                        os.path.join(base.audio_dir, row["filename"]),
                        int(base.cat2idx[row["category"]]),
                    )
                )
            self.num_classes = 50
            self.multi_label = False
            self.crop_mode = "start"
        elif task == "us8k":
            from us8k_refer import UrbanSoundDataset

            base = UrbanSoundDataset(root, test_fold=fold, train=train)
            for filename, item_fold, label in zip(base.files, base.folds, base.labels):
                self.records.append(
                    (
                        os.path.join(root, "audio", f"fold{item_fold}", filename),
                        int(label),
                    )
                )
            self.num_classes = 10
            self.multi_label = False
            self.crop_mode = "random" if train else "start"
        elif task == "scv2":
            root_path = Path(root)
            classes = sorted(p.name for p in root_path.iterdir() if p.is_dir() and not p.name.startswith("_"))
            class_id = {name: i for i, name in enumerate(classes)}
            val = {line.strip() for line in (root_path / "validation_list.txt").read_text().splitlines() if line.strip()}
            test = {line.strip() for line in (root_path / "testing_list.txt").read_text().splitlines() if line.strip()}
            for cls in classes:
                for path in sorted((root_path / cls).glob("*.wav")):
                    relative = f"{cls}/{path.name}"
                    if train and relative not in val | test:
                        self.records.append((str(path), class_id[cls]))
                    if not train and relative in test:
                        self.records.append((str(path), class_id[cls]))
            self.num_classes, self.multi_label, self.crop_mode = 35, False, "start"
        elif task == "audioset20k":
            self.audioset = AudioSet20KTarDataset(root, train=train, sample_rate=SAMPLE_RATE, duration=10.24, augment=train)
            self.num_classes, self.multi_label, self.crop_mode = 527, True, "random" if train else "center"
        elif task == "fsd50k":
            from fsd50k_refer import FSD50KDataset

            base = FSD50KDataset(root, split="train" if train else "val")
            for filename, raw_labels in zip(base.fnames, base.raw_labels):
                target = torch.zeros(len(base.label2id), dtype=torch.float32)
                for label in (item.strip() for item in raw_labels.split(",")):
                    if label in base.label2id:
                        target[base.label2id[label]] = 1.0
                self.records.append(
                    (os.path.join(base.audio_dir, f"{filename}.wav"), target)
                )
            self.num_classes = len(base.label2id)
            self.multi_label = True
            self.crop_mode = "random" if train else "center"
        else:
            raise ValueError(task)

    def __len__(self) -> int:
        if self.audioset is not None:
            return len(self.audioset)
        return len(self.records)

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.shape[-1]
        if length == 0:
            return torch.zeros(1, TARGET_SAMPLES)
        if length < TARGET_SAMPLES:
            waveform = waveform.repeat(1, TARGET_SAMPLES // length + 1)
            length = waveform.shape[-1]
        maximum = length - TARGET_SAMPLES
        if self.crop_mode == "random" and maximum > 0:
            start = random.randint(0, maximum)
        elif self.crop_mode == "center" and maximum > 0:
            start = maximum // 2
        else:
            start = 0
        return waveform[:, start:start + TARGET_SAMPLES]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor | int]:
        if self.audioset is not None:
            waveform, target = self.audioset[index]
            waveform = waveform.unsqueeze(0)
            if self.train and self.augment:
                waveform = self.wave_augment(waveform)
            return waveform.squeeze(0).float(), target
        path, target = self.records[index]
        try:
            waveform, sample_rate = torchaudio.load(path)
        except Exception:
            waveform = torch.zeros(1, TARGET_SAMPLES)
            sample_rate = SAMPLE_RATE
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, SAMPLE_RATE
            )
        waveform = waveform.mean(dim=0, keepdim=True)
        waveform = self._fix_length(waveform)
        if self.train and self.augment:
            waveform = self.wave_augment(waveform)
        return waveform.squeeze(0).float(), target


def task_root(task: str, path: str) -> str:
    if path:
        return path
    return {
        "esc50": "datasets/ESC-50/ESC-50-master",
        "fsd50k": "datasets/FSD50K",
        "us8k": "datasets/UrbanSound8K",
        "scv2": "datasets/SpeechCommands/speech_commands_v0.02",
        "audioset20k": "datasets/AudioSetMirror",
    }[task]


def get_dependencies():
    from byola_v2_compatible import make_mel, require_dependencies
    from fsd50k_refer import AsymmetricLoss, MixupCutmix

    dependencies = require_dependencies()
    return dependencies[0], dependencies[1], make_mel, AsymmetricLoss, MixupCutmix


@torch.no_grad()
def compute_norm_stats(
    loader: DataLoader,
    mel: nn.Module,
    device: torch.device,
    sample_limit: int,
) -> Tuple[float, float]:
    total = torch.zeros(3, dtype=torch.float64, device=device)
    observed = 0
    for waveforms, _ in tqdm(loader, desc="normalization", leave=False):
        waveforms = waveforms.to(device, non_blocking=True)
        values = (mel(waveforms.float()) + torch.finfo(torch.float32).eps).log()
        take = min(values.shape[0], sample_limit - observed)
        values = values[:take].double()
        total[0] += values.sum()
        total[1] += values.square().sum()
        total[2] += values.numel()
        observed += take
        if observed >= sample_limit:
            break
    count = max(1.0, float(total[2].item()))
    mean = float(total[0].item()) / count
    variance = max(float(total[1].item()) / count - mean * mean, 1e-12)
    return mean, variance**0.5


class AudioNTTClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(spectrogram))


@torch.no_grad()
def to_log_mel(
    waveforms: torch.Tensor,
    mel: nn.Module,
    norm_stats: Tuple[float, float],
) -> torch.Tensor:
    values = (mel(waveforms.float()) + torch.finfo(torch.float32).eps).log()
    return ((values - norm_stats[0]) / norm_stats[1]).unsqueeze(1)


def build_mix_and_loss(
    task: str,
    num_classes: int,
    asymmetric_loss_cls: type[nn.Module],
    mixup_cutmix_cls: Any,
) -> Tuple[Any, nn.Module]:
    if task in {"fsd50k", "audioset20k"}:
        return (
            mixup_cutmix_cls(
                mixup_alpha=0.8,
                cutmix_alpha=1.0,
                mix_prob=0.5,
                cutmix_prob=0.5,
            ),
            asymmetric_loss_cls(gamma_neg=4, gamma_pos=1, clip=0.05),
        )
    return (
        Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.1,
            num_classes=num_classes,
        ),
        SoftTargetCrossEntropy(),
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    mel: nn.Module,
    norm_stats: Tuple[float, float],
    device: torch.device,
    multi_label: bool,
    amp_enabled: bool,
    max_steps: int = 0,
) -> float:
    model.eval()
    predictions, targets = [], []
    for step, (waveforms, target) in enumerate(
        tqdm(loader, desc="validate", leave=False)
    ):
        waveforms = waveforms.to(device, non_blocking=True)
        spectrogram = to_log_mel(waveforms, mel, norm_stats)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(spectrogram)
        predictions.append(logits.float().cpu())
        targets.append(target.cpu())
        if max_steps and step + 1 >= max_steps:
            break
    logits = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    if multi_label:
        return float(
            average_precision_score(
                target_tensor.numpy(),
                torch.sigmoid(logits).numpy(),
                average="macro",
            )
        )
    return float(
        (logits.argmax(dim=1) == target_tensor).float().mean().item() * 100.0
    )


def main() -> None:
    args = parse_args()
    if args.task == "esc50":
        args.batch_size = args.batch_size or 16
        args.lr = args.lr if args.lr is not None else 2e-4
        args.ema_decay = args.ema_decay if args.ema_decay is not None else 0.9998
    else:
        args.batch_size = args.batch_size or 32
        args.lr = args.lr if args.lr is not None else 1e-4
        args.ema_decay = args.ema_decay if args.ema_decay is not None else 0.999
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    logger = make_logger(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"

    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    feature_dim = int(config.get("feature_dim", 3072))
    root = task_root(args.task, args.data_path)
    train_dataset = WaveformTaskDataset(
        args.task, root, args.fold, train=True, augment=True
    )
    stats_dataset = WaveformTaskDataset(
        args.task, root, args.fold, train=True, augment=False
    )
    val_dataset = WaveformTaskDataset(
        args.task, root, args.fold, train=False, augment=False
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    stats_loader = DataLoader(
        stats_dataset, shuffle=False, drop_last=False, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_kwargs
    )

    nn_audio_features, audio_ntt_cls, make_mel, asymmetric_loss_cls, mixup_cutmix_cls = (
        get_dependencies()
    )
    mel = make_mel(nn_audio_features, device)
    norm_stats = compute_norm_stats(
        stats_loader,
        mel,
        device,
        min(args.norm_samples, len(stats_dataset)),
    )
    model = AudioNTTClassifier(
        audio_ntt_cls(n_mels=64, d=feature_dim),
        feature_dim,
        train_dataset.num_classes,
    )
    model.encoder.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model_ema = ModelEmaV2(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01 if args.task in {"fsd50k", "audioset20k"} else 0.05,
    )
    if args.task == "esc50":
        scheduler: Any = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
        )
        scheduler_per_step = True
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        scheduler_per_step = False
    mix_fn, criterion = build_mix_and_loss(
        args.task,
        train_dataset.num_classes,
        asymmetric_loss_cls,
        mixup_cutmix_cls,
    )
    freq_mask = torchaudio.transforms.FrequencyMasking(
        freq_mask_param=24 if args.task in {"fsd50k", "audioset20k"} else 30
    ).to(device)
    time_mask = torchaudio.transforms.TimeMasking(
        time_mask_param=96 if args.task in {"fsd50k", "audioset20k"} else 80
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_metric = float("-inf")
    metric_name = "mAP" if train_dataset.multi_label else "accuracy"

    logger.info(
        "task=%s train=%s val=%s classes=%s epochs=%s frontend=nnAudio-64mel",
        args.task,
        len(train_dataset),
        len(val_dataset),
        train_dataset.num_classes,
        args.epochs,
    )
    logger.info(
        "normalization_source=downstream_train mean=%.8f std=%.8f",
        norm_stats[0],
        norm_stats[1],
    )
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        train_steps = 0
        iterator = tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}")
        for step, (waveforms, target) in enumerate(iterator):
            waveforms = waveforms.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            spectrogram = to_log_mel(waveforms, mel, norm_stats)
            spectrogram = freq_mask(spectrogram)
            spectrogram = time_mask(spectrogram)
            if args.task in {"fsd50k", "audioset20k"}:
                spectrogram = freq_mask(spectrogram)
                spectrogram = time_mask(spectrogram)
            spectrogram, target = mix_fn(spectrogram, target)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(spectrogram)
                loss = criterion(
                    logits,
                    target.float() if train_dataset.multi_label else target,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model_ema.update(model)
            if scheduler_per_step:
                scheduler.step()
            total_loss += float(loss.detach().item())
            train_steps += 1
            iterator.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            if args.max_train_steps and step + 1 >= args.max_train_steps:
                break
        if not scheduler_per_step:
            scheduler.step()

        metric = validate(
            model_ema.module,
            val_loader,
            mel,
            norm_stats,
            device,
            train_dataset.multi_label,
            amp_enabled,
            args.max_val_steps,
        )
        logger.info(
            "epoch=%s/%s train_loss=%.6f %s=%.6f",
            epoch + 1,
            args.epochs,
            total_loss / max(1, train_steps),
            metric_name,
            metric,
        )
        if metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "model_state": model_ema.module.state_dict(),
                    "normalization_stats": list(norm_stats),
                    "task": args.task,
                    "metric": metric_name,
                    "best": best_metric,
                    "epoch": epoch + 1,
                },
                output_dir / "best.pth",
            )

    summary = {
        "task": args.task,
        "metric": metric_name,
        "best": best_metric,
        "epochs": args.epochs,
        "pretrained": args.pretrained,
        "frontend": "nnAudio-64mel-waveform-native",
        "normalization_stats": list(norm_stats),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("completed %s", json.dumps(summary, sort_keys=True))
    del model, model_ema, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
