"""Formal ESC-50 full fine-tuning adapter for official SSAST and PaSST.

This keeps the train-time policy of ``esc_formal_version.py`` while retaining
the architecture-specific frontend required by each external checkpoint.
"""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader, Dataset

from official_ssl.finetune import build_passt, build_ssast
from official_ssl.passt_compat import prepare_passt_import
from official_ssl.runtime import make_logger, seed_everything, write_json


ROOT = Path(__file__).resolve().parents[1]
ESC50_CLASSES = 50
TARGET_FRAMES = 1024


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["ssast", "passt"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class WaveformAugment:
    """The ESC formal waveform policy, unchanged from esc_formal_version.py."""

    def __call__(self, waveform):
        if random.random() < 0.5:
            waveform = waveform * random.uniform(0.8, 1.2)
        if random.random() < 0.5:
            waveform = waveform + torch.randn_like(waveform) * random.uniform(0.001, 0.015)
        if random.random() < 0.5:
            waveform = torch.roll(waveform, random.randint(0, waveform.shape[1]), dims=1)
        return waveform


def waveform_to_formal_fbank(waveform):
    """The AudioMAE formal 16 kHz Kaldi fbank and normalization."""

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )
    padding = TARGET_FRAMES - fbank.shape[0]
    if padding > 0:
        fbank = F.pad(fbank, (0, 0, 0, padding))
    else:
        fbank = fbank[:TARGET_FRAMES]
    return (fbank - (-4.26)) / (4.56 * 2)


class ESC50FormalDataset(Dataset):
    """Formal ESC data policy with optional SSAST-compatible fbank output."""

    def __init__(self, root, fold, train, sample_rate, duration, as_fbank):
        self.root = Path(root)
        self.train = train
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * duration)
        self.as_fbank = as_fbank
        metadata = pd.read_csv(self.root / "meta" / "esc50.csv")
        self.rows = metadata[metadata["fold"] != fold] if train else metadata[metadata["fold"] == fold]
        categories = sorted(metadata["category"].unique())
        self.category_to_index = {category: index for index, category in enumerate(categories)}
        if len(self.category_to_index) != ESC50_CLASSES:
            raise ValueError(f"Expected {ESC50_CLASSES} ESC-50 classes, found {len(self.category_to_index)}")
        if train:
            self.wave_augment = WaveformAugment()
            if as_fbank:
                self.frequency_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=30)
                self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=80)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        waveform, source_rate = torchaudio.load(self.root / "audio" / row.filename)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] < self.target_samples:
            waveform = waveform.repeat(1, self.target_samples // waveform.shape[1] + 1)
        waveform = waveform[:, :self.target_samples]
        if self.train:
            waveform = self.wave_augment(waveform)

        target = self.category_to_index[row.category]
        if not self.as_fbank:
            return waveform.squeeze(0), target

        fbank = waveform_to_formal_fbank(waveform)
        if self.train:
            fbank = self.frequency_mask(fbank.transpose(0, 1).unsqueeze(0))
            fbank = self.time_mask(fbank).squeeze(0).transpose(0, 1)
        return fbank.unsqueeze(0), target


class PaSSTFrontend:
    """The PaSST repository's mandatory 32 kHz frontend and native masks."""

    def __init__(self, device):
        prepare_passt_import(ROOT)
        from models.preprocess import AugmentMelSTFT

        self.mel = AugmentMelSTFT(
            n_mels=128,
            sr=32000,
            win_length=800,
            hopsize=320,
            n_fft=1024,
            freqm=48,
            timem=192,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000,
        ).to(device)

    def train(self):
        self.mel.train()

    def eval(self):
        self.mel.eval()

    def __call__(self, waveform):
        features = self.mel(waveform)
        if features.shape[-1] < 998:
            features = F.pad(features, (0, 998 - features.shape[-1]))
        return features[..., :998].unsqueeze(1)


def infer_transformer_depth(model):
    ids = []
    for name, _ in model.named_parameters():
        match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", name)
        if match:
            ids.append(int(match.group(1)))
    if not ids:
        raise RuntimeError("Could not infer transformer block depth for LLRD.")
    return max(ids) + 1


def parameter_groups_with_llrd(model, weight_decay, lr, layer_decay):
    """Formal AudioMAE LLRD policy generalized to AST and PaSST parameter names."""

    depth = infer_transformer_depth(model)
    groups = {}
    input_tokens = ("patch_embed", "cls_token", "dist_token", "pos_embed", "new_pos_embed", "freq_new_pos_embed", "time_new_pos_embed")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", name)
        if match:
            layer_id = int(match.group(1)) + 1
        elif any(token in name for token in input_tokens):
            layer_id = 0
        else:
            layer_id = depth + 1
        decay = 0.0 if parameter.ndim == 1 or name.endswith(".bias") or name.endswith(("cls_token", "dist_token", "pos_embed")) else weight_decay
        key = (layer_id, decay)
        if key not in groups:
            scale = layer_decay ** (depth + 1 - layer_id)
            groups[key] = {"params": [], "weight_decay": decay, "lr": lr * scale}
        groups[key]["params"].append(parameter)
    return list(groups.values())


def evaluate(model, frontend, loader, method, device):
    model.eval()
    if frontend:
        frontend.eval()
    correct = total = 0
    with torch.no_grad():
        for features, target in loader:
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                if method == "passt":
                    features = frontend(features.to(device, non_blocking=True))
                    logits = model(features)[0]
                else:
                    logits = model(features.to(device, non_blocking=True).squeeze(1), "ft_avgtok")
            correct += (logits.argmax(dim=1) == target).sum().item()
            total += target.shape[0]
    return 100.0 * correct / total


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoints").mkdir()
    logger = make_logger(output_dir / "logs" / "finetune.log", f"{args.method}_esc50_{output_dir.name}")
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_ssast = args.method == "ssast"
    sample_rate, duration = (16000, 10.24) if is_ssast else (32000, 10.0)
    train_set = ESC50FormalDataset(args.data_path, args.fold, True, sample_rate, duration, as_fbank=is_ssast)
    validation_set = ESC50FormalDataset(args.data_path, args.fold, False, sample_rate, duration, as_fbank=is_ssast)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                   pin_memory=True, persistent_workers=args.workers > 0)
    model = build_ssast(args.checkpoint, ESC50_CLASSES, device) if is_ssast else build_passt(args.checkpoint, ESC50_CLASSES, device)
    frontend = None if is_ssast else PaSSTFrontend(device)
    model_ema = ModelEmaV2(model, decay=0.9998)
    optimizer = torch.optim.AdamW(parameter_groups_with_llrd(model, args.weight_decay, args.lr, args.layer_decay))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in optimizer.param_groups],
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode="batch", label_smoothing=0.1, num_classes=ESC50_CLASSES)
    criterion = SoftTargetCrossEntropy()
    write_json(output_dir / "run.json", {
        "method": args.method,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "fold": args.fold,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "optimizer": "AdamW + LLRD (layer_decay=0.75)",
        "mixup": "timm Mixup(alpha=0.8) / CutMix(alpha=1.0), label_smoothing=0.1",
        "precision": "torch.amp.autocast + torch.amp.GradScaler",
        "frontend": "formal 16 kHz fbank" if is_ssast else "official PaSST 32 kHz AugmentMelSTFT (architecture-required)",
        "smoke": args.smoke,
    })
    best_accuracy = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        if frontend:
            frontend.train()
        total_loss = 0.0
        for features, target in train_loader:
            target = target.to(device, non_blocking=True)
            if is_ssast:
                features = features.to(device, non_blocking=True)
            else:
                features = frontend(features.to(device, non_blocking=True))
            features, target = mixup(features, target)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(features.squeeze(1), "ft_avgtok") if is_ssast else model(features)[0]
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model_ema.update(model)
            total_loss += loss.item()
        accuracy = evaluate(model_ema.module, frontend, validation_loader, args.method, device)
        logger.info("epoch=%d/%d train_loss=%.5f ema_accuracy=%.4f lr=%.2e", epoch, args.epochs,
                    total_loss / max(len(train_loader), 1), accuracy, optimizer.param_groups[-1]["lr"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if not args.smoke:
                torch.save({"model_state": model_ema.module.state_dict(), "epoch": epoch, "metric": accuracy}, output_dir / "checkpoints" / "best.pth")
    write_json(output_dir / "summary.json", {"metric": "accuracy", "best": best_accuracy, "epochs": args.epochs, "smoke": args.smoke})
    logger.info("complete: best_accuracy=%.4f", best_accuracy)


if __name__ == "__main__":
    main()
