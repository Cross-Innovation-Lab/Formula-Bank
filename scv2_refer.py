#!/usr/bin/env python
"""Speech Commands v2 full fine-tuning for the AudioMAE encoder.

The split follows the official ``validation_list.txt`` and
``testing_list.txt`` files.  Commands are repeated deterministically to the
10.24-s AudioMAE input duration; training additionally uses waveform and
log-mel augmentation.  Validation/test never use stochastic transforms.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2

from esc_refer import AudioMAEClassifier, get_audiomae_config, get_parameter_groups_with_llrd


class SCv2Dataset(Dataset):
    """Official SCv2 split with deterministic 10.24-s waveform construction."""

    def __init__(self, root, split, train=False, specaug=True):
        self.root = Path(root)
        self.train = train
        self.specaug = specaug and train
        self.sr = 16000
        self.target_samples = int(10.24 * self.sr)
        self.target_frames = 1024
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=12)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=48)
        self.noise_files = sorted((self.root / "_background_noise_").glob("*.wav"))

        val_list = self._read_list("validation_list.txt")
        test_list = self._read_list("testing_list.txt")
        if split == "val": chosen = val_list
        elif split == "test": chosen = test_list
        elif split == "train":
            chosen = None
        else: raise ValueError("split must be train, val, or test")

        classes = sorted(p.name for p in self.root.iterdir()
                         if p.is_dir() and not p.name.startswith("_"))
        self.classes = classes
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        all_items = []
        listed_val = set(val_list)
        listed_test = set(test_list)
        for cls in classes:
            for path in sorted((self.root / cls).glob("*.wav")):
                rel = f"{cls}/{path.name}"
                if split == "train" and (rel in listed_val or rel in listed_test):
                    continue
                if chosen is not None and rel not in set(chosen):
                    continue
                all_items.append((path, self.class_to_idx[cls]))
        self.items = all_items

    def _read_list(self, name):
        p = self.root / name
        if not p.exists():
            raise FileNotFoundError(f"Missing official split file: {p}")
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]

    def __len__(self): return len(self.items)

    @staticmethod
    def _zero_shift(x, shift):
        if shift == 0: return x
        y = torch.zeros_like(x)
        if shift > 0: y[..., shift:] = x[..., :-shift]
        else: y[..., :shift] = x[..., -shift:]
        return y

    def _augment_waveform(self, x):
        if random.random() < 0.8:
            x = self._zero_shift(x, random.randint(-1600, 1600))
        if random.random() < 0.8:
            x = x * random.uniform(0.7, 1.3)
        if random.random() < 0.5:
            x = -x
        if self.noise_files and random.random() < 0.5:
            n, nsr = torchaudio.load(random.choice(self.noise_files))
            if nsr != self.sr: n = torchaudio.functional.resample(n, nsr, self.sr)
            n = n.mean(0, keepdim=True)
            if n.shape[1] < x.shape[1]:
                n = n.repeat(1, x.shape[1] // n.shape[1] + 1)
            start = random.randint(0, max(0, n.shape[1] - x.shape[1]))
            n = n[:, start:start + x.shape[1]]
            snr = random.uniform(8.0, 25.0)
            scale = x.std().clamp_min(1e-5) / n.std().clamp_min(1e-5) / (10 ** (snr / 20))
            x = x + scale * n
        if random.random() < 0.3:
            x = x + torch.randn_like(x) * random.uniform(0.0005, 0.003)
        return x.clamp(-1, 1)

    def __getitem__(self, index):
        path, label = self.items[index]
        x, sr = torchaudio.load(path)
        if sr != self.sr: x = torchaudio.functional.resample(x, sr, self.sr)
        x = x.mean(0, keepdim=True)
        if x.shape[1] < self.sr: x = F.pad(x, (0, self.sr - x.shape[1]))
        else: x = x[:, :self.sr]
        # Match the 10.24-s AudioMAE frontend without positional interpolation.
        x = x.repeat(1, 11)[:, :self.target_samples]
        # Apply stochastic waveform transforms to the complete AudioMAE
        # window.  Applying them before repetition would replicate the same
        # noise/shift ten times and introduce an artificial periodic cue.
        if self.train: x = self._augment_waveform(x)
        # Use the same Kaldi-compatible frontend and normalization as the
        # existing ESC/US8K/FSD AudioMAE protocols.
        mel = torchaudio.compliance.kaldi.fbank(
            x, htk_compat=True, sample_frequency=self.sr, use_energy=False,
            window_type="hanning", num_mel_bins=128, dither=0.0,
            frame_shift=10)
        if mel.shape[0] < self.target_frames: mel = F.pad(mel, (0, 0, 0, self.target_frames - mel.shape[0]))
        else: mel = mel[:self.target_frames]
        if self.specaug:
            m = mel.transpose(0, 1).unsqueeze(0)
            if random.random() < 0.7: m = self.freq_mask(m)
            if random.random() < 0.7: m = self.time_mask(m)
            mel = m.squeeze(0).transpose(0, 1)
        mel = (mel - (-4.26)) / (4.56 * 2)
        return mel.unsqueeze(0).float(), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="datasets/SpeechCommands/speech_commands_v0.02")
    ap.add_argument("--pretrained", default="",
                    help="MAE checkpoint used for full fine-tuning")
    ap.add_argument("--eval-only", action="store_true",
                    help="Evaluate a saved classifier checkpoint on the official SCv2 test split")
    ap.add_argument("--eval-checkpoint", default="",
                    help="Classifier state_dict produced by this script; required with --eval-only")
    ap.add_argument("--model-size", default="base", choices=["tiny", "small", "base", "large"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--llrd-decay", type=float, default=0.95)
    ap.add_argument("--ema-decay", type=float, default=0.9998)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output-dir", default="ft_checkpoints/scv2")
    ap.add_argument("--no-specaug", action="store_true")
    args = ap.parse_args()
    if args.eval_only and not args.eval_checkpoint:
        ap.error("--eval-checkpoint is required with --eval-only")
    if not args.eval_only and not args.pretrained:
        ap.error("--pretrained is required unless --eval-only is used")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    file_handler = logging.FileHandler(os.path.join(args.output_dir, "train.log"), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)
    log = logging.getLogger("scv2")
    test_ds = SCv2Dataset(args.data_path, "test", False, False)
    if args.eval_only:
        log.info("SCv2 test-only: classes=%d test=%d", len(test_ds.classes), len(test_ds))
        test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        model = AudioMAEClassifier(num_classes=len(test_ds.classes), stride=16,
                                   **get_audiomae_config(args.model_size))
        model.load_state_dict(torch.load(args.eval_checkpoint, map_location=device, weights_only=True))
        model.to(device).eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="SCv2 test"):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    p = model(x.to(device, non_blocking=True)).argmax(1)
                correct += (p.cpu() == y).sum().item()
                total += y.numel()
        test_acc = 100 * correct / total
        summary = {
            "task": "scv2",
            "metric": "test_accuracy",
            "test_acc": test_acc,
            "checkpoint": args.eval_checkpoint,
            "num_test_examples": total,
        }
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        log.info("finished test_acc=%.2f%% summary=%s", test_acc, summary_path)
        return

    train_ds = SCv2Dataset(args.data_path, "train", True, not args.no_specaug)
    val_ds = SCv2Dataset(args.data_path, "val", False, False)
    log.info("SCv2 classes=%d train=%d val=%d test=%d", len(train_ds.classes), len(train_ds), len(val_ds), len(test_ds))
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = AudioMAEClassifier(num_classes=len(train_ds.classes), stride=16,
                               **get_audiomae_config(args.model_size))
    model.load_pretrained_mae(args.pretrained, log)
    model.to(device)
    ema = ModelEmaV2(model, decay=args.ema_decay)
    groups = get_parameter_groups_with_llrd(model, weight_decay=0.05, lr=args.lr, layer_decay=args.llrd_decay)
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=.1)
    mixup = Mixup(mixup_alpha=.4, cutmix_alpha=.2, prob=.5, switch_prob=.5,
                  mode="batch", label_smoothing=.1, num_classes=len(train_ds.classes))
    criterion = SoftTargetCrossEntropy(); scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = 0.0
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y = mixup(x, y)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            running += loss.item()
        ema.module.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"): p = ema.module(x.to(device)).argmax(1)
                correct += (p.cpu() == y).sum().item(); total += y.numel()
        acc = 100 * correct / total
        log.info("epoch=%d train_loss=%.4f val_acc=%.2f%%", epoch + 1, running / len(train_loader), acc)
        if acc > best:
            best = acc; torch.save(ema.module.state_dict(), os.path.join(args.output_dir, "best.pth"))
    best_path = os.path.join(args.output_dir, "best.pth")
    if os.path.exists(best_path):
        ema.module.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    ema.module.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"): p = ema.module(x.to(device)).argmax(1)
            correct += (p.cpu() == y).sum().item(); total += y.numel()
    test_acc = 100 * correct / total
    summary = {
        "task": "scv2",
        "metric": "test_accuracy",
        "best_val_acc": best,
        "test_acc": test_acc,
        "epochs": args.epochs,
        "pretrained": args.pretrained,
        "num_test_examples": total,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    log.info("finished best_val=%.2f%% test_acc=%.2f%% summary=%s", best, test_acc, summary_path)


if __name__ == "__main__": main()
