"""Full downstream fine-tuning for official SSAST and PaSST baselines."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from official_ssl.runtime import make_logger, seed_everything, write_json
from official_ssl.passt_compat import prepare_passt_import
from official_ssl.synthetic import waveform_to_ssast_fbank


ROOT = Path(__file__).resolve().parents[1]


TASK_DEFAULTS = {
    "esc50": {"epochs": 200, "classes": 50, "ema": 0.9998, "lr": 2e-4},
    "fsd50k": {"epochs": 100, "classes": None, "ema": 0.999, "lr": 1e-4},
    "us8k": {"epochs": 150, "classes": 10, "ema": 0.999, "lr": 2e-4},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ssast", "passt"], required=True)
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA autocast and GradScaler.")
    parser.add_argument(
        "--fast-audio-loading",
        action="store_true",
        help="Read only the selected source-audio window before resampling.",
    )
    parser.add_argument(
        "--ssast-worker-fbank",
        action="store_true",
        help="Compute SSAST Kaldi fbank features inside DataLoader workers.",
    )
    return parser.parse_args()


def waveform_augment(waveform, task):
    if random.random() < 0.5:
        low, high = (0.7, 1.3) if task == "fsd50k" else (0.8, 1.2)
        waveform = waveform * random.uniform(low, high)
    if random.random() < 0.5:
        low, high = (0.0005, 0.005) if task == "fsd50k" else ((0.001, 0.01) if task == "us8k" else (0.001, 0.015))
        waveform = waveform + torch.randn_like(waveform) * random.uniform(low, high)
    if random.random() < 0.5:
        waveform = torch.roll(waveform, random.randint(0, waveform.shape[-1]), dims=-1)
    if task == "fsd50k" and random.random() < 0.5:
        waveform = -waveform
    return waveform


def _load_window_before_resample(path, target_sr, target_samples, train, crop_random):
    """Load a target-rate crop with enough context for resampling.

    The read offset is aligned to the resampling grid, so cropping the locally
    resampled waveform matches cropping a fully resampled waveform apart from
    floating-point noise at the context boundary.
    """
    info = torchaudio.info(path)
    source_sr = int(info.sample_rate)
    source_frames = int(info.num_frames)
    if source_sr <= 0 or source_frames <= 0:
        return None

    output_frames = math.ceil(source_frames * target_sr / source_sr)
    if output_frames <= target_samples:
        return None
    if train:
        crop_start = random.randint(0, output_frames - target_samples)
    else:
        crop_start = (output_frames - target_samples) // 2 if crop_random else 0

    context_frames = 64
    read_start_at_target_rate = max(0, crop_start - context_frames)
    source_offset = math.floor(read_start_at_target_rate * source_sr / target_sr)
    alignment = source_sr // math.gcd(source_sr, target_sr)
    source_offset -= source_offset % alignment
    output_offset = crop_start - source_offset * target_sr // source_sr

    read_end_at_target_rate = crop_start + target_samples + context_frames
    source_end = math.ceil(read_end_at_target_rate * source_sr / target_sr) + alignment
    source_end = min(source_end, source_frames)
    waveform, loaded_sr = torchaudio.load(
        path,
        frame_offset=source_offset,
        num_frames=source_end - source_offset,
    )
    if loaded_sr != source_sr:
        raise RuntimeError(f"Audio metadata changed while reading {path}: {source_sr} -> {loaded_sr}")
    waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
    if source_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, source_sr, target_sr)
    waveform = waveform[:, output_offset:output_offset + target_samples]
    return waveform if waveform.shape[-1] == target_samples else None


def load_mono_waveform(path, target_sr, target_samples, train, crop_random, fast_audio_loading=False):
    if fast_audio_loading:
        waveform = _load_window_before_resample(path, target_sr, target_samples, train, crop_random)
        if waveform is not None:
            return waveform

    waveform, source_sr = torchaudio.load(path)
    if source_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, source_sr, target_sr)
    waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
    if waveform.shape[-1] < target_samples:
        waveform = waveform.repeat(1, (target_samples + waveform.shape[-1] - 1) // waveform.shape[-1])
    if waveform.shape[-1] > target_samples:
        if train and crop_random:
            start = random.randint(0, waveform.shape[-1] - target_samples)
        elif train:
            start = random.randint(0, waveform.shape[-1] - target_samples)
        else:
            start = (waveform.shape[-1] - target_samples) // 2 if crop_random else 0
        waveform = waveform[:, start:start + target_samples]
    return waveform


class RawBenchmarkDataset(Dataset):
    """Same task partitions and waveform augmentation as the existing scripts."""

    def __init__(
        self,
        task,
        root,
        train,
        target_sr,
        duration,
        fold=None,
        fast_audio_loading=False,
        as_ssast_fbank=False,
    ):
        self.task, self.root, self.train = task, Path(root), train
        self.target_sr = target_sr
        self.target_samples = int(target_sr * duration)
        self.fold = fold or (1 if task == "esc50" else 10)
        self.fast_audio_loading = bool(fast_audio_loading)
        self.as_ssast_fbank = bool(as_ssast_fbank)
        self.entries, self.num_classes = self._build_entries()

    def _build_entries(self):
        if self.task == "esc50":
            df = pd.read_csv(self.root / "meta" / "esc50.csv")
            selected = df[df["fold"] != self.fold] if self.train else df[df["fold"] == self.fold]
            categories = sorted(df["category"].unique())
            class_id = {category: index for index, category in enumerate(categories)}
            entries = [(self.root / "audio" / row.filename, class_id[row.category]) for row in selected.itertuples()]
            return entries, len(categories)
        if self.task == "us8k":
            meta = self.root / "metadata" / "UrbanSound8K.csv"
            df = pd.read_csv(meta if meta.exists() else self.root / "UrbanSound8K.csv")
            selected = df[df["fold"] != self.fold] if self.train else df[df["fold"] == self.fold]
            entries = [
                (self.root / "audio" / f"fold{row.fold}" / row.slice_file_name, int(row.classID))
                for row in selected.itertuples()
            ]
            return entries, 10

        gt_dir = self.root / "FSD50K.ground_truth"
        if not gt_dir.is_dir():
            gt_dir = self.root / "labels"
        vocabulary = pd.read_csv(gt_dir / "vocabulary.csv", header=None, dtype=str)
        labels = []
        for _, row in vocabulary.iterrows():
            values = [str(value).strip() for value in row.values if pd.notna(value)]
            if any(value.lower() == "label" for value in values):
                continue
            candidates = [value for value in values if not value.startswith("/m/") and not value.isdigit()]
            if candidates:
                labels.append(candidates[0])
        label_id = {label: index for index, label in enumerate(labels)}
        df = pd.read_csv(gt_dir / "dev.csv", dtype=str)
        split = "train" if self.train else "val"
        selected = df[df["split"].str.strip() == split]
        audio_dir = self.root / "FSD50K.dev_audio"
        if not audio_dir.is_dir():
            audio_dir = self.root / "clips" / "dev"
        entries = []
        for row in selected.itertuples():
            target = torch.zeros(len(label_id), dtype=torch.float32)
            for label in str(row.labels).split(","):
                if label.strip() in label_id:
                    target[label_id[label.strip()]] = 1.0
            entries.append((audio_dir / f"{row.fname}.wav", target))
        return entries, len(label_id)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, target = self.entries[index]
        crop_random = self.task != "esc50"
        try:
            waveform = load_mono_waveform(
                path,
                self.target_sr,
                self.target_samples,
                self.train,
                crop_random,
                fast_audio_loading=self.fast_audio_loading,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to load {path}") from exc
        if self.train:
            waveform = waveform_augment(waveform, self.task)
        waveform = waveform.squeeze(0)
        if self.as_ssast_fbank:
            waveform = waveform_to_ssast_fbank(waveform)
        return waveform, target


class AsymmetricLoss(nn.Module):
    """Unchanged ASL parameters from fsd50k_refer.py."""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, target):
        positive = torch.sigmoid(logits)
        negative = 1.0 - positive
        if self.clip:
            negative = (negative + self.clip).clamp(max=1)
        loss = target * torch.log(positive.clamp(min=self.eps))
        loss = loss + (1.0 - target) * torch.log(negative.clamp(min=self.eps))
        with torch.no_grad():
            pt = positive * target + negative * (1.0 - target)
            gamma = self.gamma_pos * target + self.gamma_neg * (1.0 - target)
            weight = (1.0 - pt).pow(gamma)
        return -(loss * weight).mean()


class EmaModel:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_parameter, parameter in zip(self.model.parameters(), model.parameters()):
            ema_parameter.mul_(self.decay).add_(parameter, alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def build_ssast(checkpoint, classes, device):
    sys.path.insert(0, str(ROOT / "external" / "ssast" / "src"))
    from models.ast_models import ASTModel
    return ASTModel(
        label_dim=classes, fshape=16, tshape=16, fstride=10, tstride=10,
        input_fdim=128, input_tdim=1024, model_size="base", pretrain_stage=False,
        load_pretrained_mdl_path=str(checkpoint),
    ).to(device)


def build_passt(checkpoint, classes, device):
    prepare_passt_import(ROOT)
    from models.passt import get_model
    model = get_model(
        arch="passt_deit_bd_p16_384", pretrained=False, n_classes=classes,
        fstride=10, tstride=10, input_fdim=128, input_tdim=998,
        s_patchout_t=40, s_patchout_f=4,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    # AveragedModel stores its wrapped PaSST under ``module``.  Normalize both
    # SWA and ordinary checkpoints before excluding task-specific classifiers.
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    state = {key: value for key, value in state.items() if not key.startswith(("head.", "head_dist."))}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if not key.startswith(("head.", "head_dist."))]
    if unexpected or any(not key.startswith(("head.", "head_dist.")) for key in missing):
        raise RuntimeError(f"Unexpected PaSST checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device)


class Frontend:
    def __init__(self, method, task, device, precomputed_ssast_fbank=False):
        self.method, self.task, self.device = method, task, device
        self.precomputed_ssast_fbank = bool(precomputed_ssast_fbank)
        if method == "passt":
            prepare_passt_import(ROOT)
            from models.preprocess import AugmentMelSTFT
            self.mel = AugmentMelSTFT(
                n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024,
                freqm=48, timem=192, htk=False, fmin=0.0, fmax=None,
                norm=1, fmin_aug_range=10, fmax_aug_range=2000,
            ).to(device)
        else:
            self.mel = None
            masks = {"esc50": (30, 80, 1), "us8k": (30, 80, 1), "fsd50k": (24, 96, 2)}
            frequency, time, repetitions = masks[task]
            self.frequency_mask = T.FrequencyMasking(frequency)
            self.time_mask = T.TimeMasking(time)
            self.mask_repetitions = repetitions

    def train(self):
        if self.mel:
            self.mel.train()

    def eval(self):
        if self.mel:
            self.mel.eval()

    def __call__(self, waveform, training):
        if self.method == "passt":
            features = self.mel(waveform)
            features = features[..., :998] if features.shape[-1] >= 998 else F.pad(features, (0, 998 - features.shape[-1]))
            return features.unsqueeze(1)
        if self.precomputed_ssast_fbank:
            fbank = waveform
        else:
            fbank = torch.stack([waveform_to_ssast_fbank(item.cpu()).to(waveform.device) for item in waveform])
        if training:
            fbank = fbank.transpose(1, 2).unsqueeze(1)
            for _ in range(self.mask_repetitions):
                fbank = self.frequency_mask(fbank)
                fbank = self.time_mask(fbank)
            fbank = fbank.squeeze(1).transpose(1, 2)
        return fbank


def apply_mixup_cutmix(features, target, task, classes):
    if task in {"esc50", "us8k"}:
        target = F.one_hot(target.long(), classes).float()
    if random.random() > 1.0:
        return features, target
    indices = torch.randperm(features.shape[0], device=features.device)
    if random.random() < 0.5:
        lam = float(np.random.beta(1.0, 1.0))
        height, width = features.shape[-2:]
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
        center_h, center_w = random.randrange(height), random.randrange(width)
        h1, h2 = max(0, center_h - cut_h // 2), min(height, center_h + cut_h // 2)
        w1, w2 = max(0, center_w - cut_w // 2), min(width, center_w + cut_w // 2)
        if features.ndim == 4:
            features[:, :, h1:h2, w1:w2] = features[indices, :, h1:h2, w1:w2]
        else:
            features[:, h1:h2, w1:w2] = features[indices, h1:h2, w1:w2]
        lam = 1.0 - (h2 - h1) * (w2 - w1) / max(height * width, 1)
    else:
        lam = float(np.random.beta(0.8, 0.8))
        features = lam * features + (1.0 - lam) * features[indices]
    return features, lam * target + (1.0 - lam) * target[indices]


def forward_model(model, features, method):
    if method == "passt":
        return model(features)[0]
    return model(features, "ft_avgtok")


@torch.no_grad()
def evaluate(model, frontend, loader, method, task, device, max_batches, amp_enabled):
    model.eval()
    frontend.eval()
    predictions, targets = [], []
    for index, (waveform, target) in enumerate(loader, start=1):
        waveform = waveform.to(device, non_blocking=True)
        features = frontend(waveform, training=False)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = forward_model(model, features, method)
        predictions.append(logits.detach().cpu())
        targets.append(target.cpu())
        if max_batches and index >= max_batches:
            break
    predictions, targets = torch.cat(predictions), torch.cat(targets)
    if task == "fsd50k":
        return float(average_precision_score(targets.numpy(), predictions.sigmoid().numpy(), average="macro"))
    return float((predictions.argmax(dim=1) == targets.long()).float().mean().item() * 100.0)


def main():
    args = parse_args()
    if args.task == "esc50":
        raise RuntimeError(
            "ESC-50 must use official_ssl.finetune_esc_formal so its training "
            "policy stays aligned with esc_formal_version.py."
        )
    defaults = TASK_DEFAULTS[args.task]
    args.epochs = args.epochs or defaults["epochs"]
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoints").mkdir()
    logger = make_logger(output_dir / "logs" / "finetune.log", f"{args.method}_{args.task}_{output_dir.name}")
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    sample_rate, duration = (16000, 10.24) if args.method == "ssast" else (32000, 10.0)
    worker_fbank = bool(args.ssast_worker_fbank and args.method == "ssast")
    train_set = RawBenchmarkDataset(
        args.task,
        args.data_path,
        True,
        sample_rate,
        duration,
        args.fold,
        fast_audio_loading=args.fast_audio_loading,
        as_ssast_fbank=worker_fbank,
    )
    val_set = RawBenchmarkDataset(
        args.task,
        args.data_path,
        False,
        sample_rate,
        duration,
        args.fold,
        fast_audio_loading=args.fast_audio_loading,
        as_ssast_fbank=worker_fbank,
    )
    classes = train_set.num_classes
    if defaults["classes"] and classes != defaults["classes"]:
        raise ValueError(f"Expected {defaults['classes']} classes for {args.task}, found {classes}")
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
    model = build_ssast(args.checkpoint, classes, device) if args.method == "ssast" else build_passt(args.checkpoint, classes, device)
    frontend = Frontend(args.method, args.task, device, precomputed_ssast_fbank=worker_fbank)
    ema = EmaModel(model, defaults["ema"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=defaults["lr"], weight_decay=0.01 if args.task == "fsd50k" else 0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=defaults["lr"], epochs=args.epochs,
                                                      steps_per_epoch=len(train_loader), pct_start=0.1) if args.task == "esc50" else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = AsymmetricLoss() if args.task == "fsd50k" else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    metric_name = "mAP" if args.task == "fsd50k" else "accuracy"
    best_metric = float("-inf")
    write_json(output_dir / "run.json", {
        "method": args.method, "task": args.task, "checkpoint": str(Path(args.checkpoint).resolve()),
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "fold": args.fold or (1 if args.task == "esc50" else 10), "full_finetune": True,
        "task_augmentation": "existing waveform + spec Mixup/CutMix; PaSST retains official native Mel masks",
        "amp": amp_enabled,
        "fast_audio_loading": bool(args.fast_audio_loading),
        "ssast_worker_fbank": worker_fbank,
    })
    for epoch in range(1, args.epochs + 1):
        model.train()
        frontend.train()
        total_loss, batch_count = 0.0, 0
        for index, (waveform, target) in enumerate(train_loader, start=1):
            waveform, target = waveform.to(device, non_blocking=True), target.to(device, non_blocking=True)
            features = frontend(waveform, training=True)
            features, mixed_target = apply_mixup_cutmix(features, target, args.task, classes)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = forward_model(model, features, args.method)
            logits_for_loss = logits.float()
            target_for_loss = mixed_target.float()
            loss = criterion(logits_for_loss, target_for_loss) if criterion else -(target_for_loss * F.log_softmax(logits_for_loss, dim=-1)).sum(dim=-1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            if args.task == "esc50":
                scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            if args.max_train_batches and index >= args.max_train_batches:
                break
        if args.task != "esc50":
            scheduler.step()
        metric = evaluate(
            ema.model,
            frontend,
            val_loader,
            args.method,
            args.task,
            device,
            args.max_val_batches,
            amp_enabled,
        )
        logger.info("epoch=%d/%d train_loss=%.5f ema_%s=%.4f lr=%.2e", epoch, args.epochs,
                    total_loss / max(batch_count, 1), metric_name, metric, optimizer.param_groups[0]["lr"])
        if metric > best_metric:
            best_metric = metric
            torch.save({"model_state": ema.model.state_dict(), "epoch": epoch, "metric": metric}, output_dir / "checkpoints" / "best.pth")
    write_json(output_dir / "summary.json", {"metric": metric_name, "best": best_metric, "epochs": args.epochs})
    logger.info("complete: best_%s=%.4f", metric_name, best_metric)


if __name__ == "__main__":
    main()
