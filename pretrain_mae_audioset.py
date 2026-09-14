#!/usr/bin/env python
"""AudioSet-manifest AudioMAE pre-training matched to the FormulaBank baseline.

The manifest, rather than AudioSet labels, defines the unsupervised corpus.  A
single-GPU epoch visits every listed clip exactly once in a deterministic
permutation.  WAV members remain inside their WebDataset tar shards; each
DataLoader worker keeps a bounded archive cache so that the archive index is
not rebuilt for every sample.
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import random
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from pretrain import autocast_cuda, build_model, count_parameters, make_grad_scaler


PROTOCOL_NAME = "audioset28k_matched_audiomae_v1"
TARGET_SAMPLES = int(16_000 * 10.24)
TARGET_FRAMES = 1024


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_rows(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"row_index", "source", "tar_path", "wav_member", "json_member", "rank_sha256"}
    if not rows or set(rows[0]) != expected:
        raise ValueError(f"Unexpected manifest columns in {path}")
    for expected_index, row in enumerate(rows):
        if int(row["row_index"]) != expected_index:
            raise ValueError(f"Non-contiguous row_index in {path}: {row['row_index']}")
    return rows


def waveform_to_fbank(waveform, sample_rate):
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[1] < TARGET_SAMPLES:
        waveform = nnf.pad(waveform, (0, TARGET_SAMPLES - waveform.shape[1]))
    else:
        waveform = waveform[:, :TARGET_SAMPLES]
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=16_000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )
    if fbank.shape[0] < TARGET_FRAMES:
        fbank = nnf.pad(fbank, (0, 0, 0, TARGET_FRAMES - fbank.shape[0]))
    else:
        fbank = fbank[:TARGET_FRAMES]
    return ((fbank - (-4.26)) / (4.56 * 2)).unsqueeze(0)


class AudioSetTarDataset(Dataset):
    """Read a locked CSV manifest without ever exposing AudioSet labels."""

    def __init__(self, root, manifest):
        self.root = Path(root)
        self.rows = read_manifest(manifest)
        self.manifest_hash = sha256_rows(self.rows)
        self._archives = {}

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archives"] = {}
        return state

    def _archive(self, relative_path):
        archive = self._archives.get(relative_path)
        if archive is None:
            archive = tarfile.open(self.root / relative_path, "r")
            self._archives[relative_path] = archive
        return archive

    def __getitem__(self, index):
        row = self.rows[int(index)]
        archive = self._archive(row["tar_path"])
        member = archive.extractfile(row["wav_member"])
        if member is None:
            raise FileNotFoundError(f"Missing {row['tar_path']}:{row['wav_member']}")
        try:
            payload = member.read()
        finally:
            member.close()
        waveform, sample_rate = torchaudio.load(io.BytesIO(payload))
        return waveform_to_fbank(waveform, sample_rate), torch.tensor(0, dtype=torch.long)


class EpochPermutationSampler(Sampler):
    """One deterministic, without-replacement permutation per epoch."""

    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1_000_003)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self):
        return len(self.data_source)


def reconstruction_validation(model, loader, device, amp_enabled, mask_ratio):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, _labels in tqdm(loader, desc="AudioSet-MAE held-out", leave=False):
            images = images.to(device, non_blocking=True)
            with autocast_cuda(enabled=amp_enabled):
                loss, _predictions, _mask = model(images, mask_ratio=mask_ratio)
            total_loss += float(loss.detach().item()) * int(images.shape[0])
            total_samples += int(images.shape[0])
    model.train()
    return {"loss": total_loss / max(1, total_samples), "samples": total_samples}


def build_logger(path):
    logger = logging.getLogger("pretrain_mae_audioset")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, mode="x", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audioset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, required=True)
    parser.add_argument("--eval-mask-ratios", nargs="*", type=float, default=[])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help="Development-only cap on optimizer steps; writes no formal claim.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive and num-workers nonnegative")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        parser.error("invalid optimizer argument")
    if not 0 < args.mask_ratio < 1 or any(not 0 < value < 1 for value in args.eval_mask_ratios):
        parser.error("mask ratios must lie in (0, 1)")
    if args.smoke_steps < 0:
        parser.error("smoke-steps must be nonnegative")
    return args


def run(args):
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu" and not args.cpu:
        raise RuntimeError("CUDA is unavailable; use --cpu only for a development smoke test")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")

    train_dataset = AudioSetTarDataset(args.audioset_root, args.train_manifest)
    audit_dataset = AudioSetTarDataset(args.audioset_root, args.audit_manifest)
    overlap = {row["rank_sha256"] for row in train_dataset.rows} & {row["rank_sha256"] for row in audit_dataset.rows}
    if overlap:
        raise RuntimeError(f"Train/audit manifests overlap: {len(overlap)} clips")
    if args.smoke_steps == 0 and len(train_dataset) != 28_000:
        raise RuntimeError(f"Formal matched run requires exactly 28,000 clips, found {len(train_dataset)}")

    variant_dir = args.output_dir / "vit_small"
    variant_dir.mkdir(parents=True, exist_ok=False)
    logger = build_logger(variant_dir / "train.log")
    train_sampler = EpochPermutationSampler(train_dataset, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    audit_loader = DataLoader(
        audit_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    steps_per_epoch = len(train_loader)
    runtime = {
        "protocol": PROTOCOL_NAME,
        "objective": "audiomae_masked_reconstruction",
        "audioset_labels_used_in_loss": False,
        "development_run": bool(args.smoke_steps),
        "source_dataset": "confit/audioset-16khz-wds",
        "source_protocol": str(args.train_manifest.parent / "protocol.json"),
        "train_manifest": str(args.train_manifest),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "train_pool_hash": train_dataset.manifest_hash,
        "audit_manifest": str(args.audit_manifest),
        "audit_manifest_sha256": sha256_file(args.audit_manifest),
        "audit_pool_hash": audit_dataset.manifest_hash,
        "unique_training_samples_per_epoch": len(train_dataset),
        "audit_samples": len(audit_dataset),
        "epochs": args.epochs,
        "per_instance_repetitions": args.epochs if not args.smoke_steps else None,
        "total_exposures": args.epochs * len(train_dataset) if not args.smoke_steps else None,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": args.epochs * steps_per_epoch if not args.smoke_steps else args.smoke_steps,
        "sampler": "single_gpu_deterministic_without_replacement_per_epoch",
        "batch_size": args.batch_size,
        "model": "small",
        "img_size": [1024, 128],
        "patch_size": 16,
        "mask_ratio": args.mask_ratio,
        "norm_pix_loss": True,
        "optimizer": "AdamW",
        "lr": args.lr,
        "betas": [0.9, 0.95],
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "amp": args.amp,
        "frontend": "mono16kHz_rightpad_to_10.24s_kaldi_fbank_1024x128_normalized",
    }
    logger.info("AudioSet-MAE training starts: %s", canonical_json(runtime))

    model = build_model("small").to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = make_grad_scaler(enabled=device.type == "cuda" and args.amp)
    started_at = time.time()
    global_step = 0
    try:
        logger.info("Model=AudioMAE ViT-SMALL trainable_parameters=%.2fM", count_parameters(model) / 1e6)
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            total_samples = 0
            for images, _labels in tqdm(train_loader, desc=f"AudioSet-MAE {epoch + 1}/{args.epochs}"):
                images = images.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_cuda(enabled=device.type == "cuda" and args.amp):
                    loss, _predictions, _mask = model(images, mask_ratio=args.mask_ratio)
                scaler.scale(loss).backward()
                if args.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach().item()) * int(images.shape[0])
                total_samples += int(images.shape[0])
                global_step += 1
                if args.smoke_steps and global_step >= args.smoke_steps:
                    break
            logger.info(
                "Epoch %d/%d step=%d/%d loss=%.6f samples=%d elapsed=%.1fs",
                epoch + 1, args.epochs, global_step, runtime["total_steps"],
                total_loss / max(1, total_samples), total_samples, time.time() - started_at,
            )
            if args.smoke_steps and global_step >= args.smoke_steps:
                break

        if args.smoke_steps:
            logger.info("Smoke completed after %d steps; no checkpoint or result is retained.", global_step)
            return
        torch.manual_seed(args.seed + 999_983)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 999_983)
        final_validation = reconstruction_validation(model, audit_loader, device, args.amp, args.mask_ratio)
        comparisons = {}
        for ratio in args.eval_mask_ratios:
            torch.manual_seed(args.seed + 1_000_000 + int(round(ratio * 10_000)))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + 1_000_000 + int(round(ratio * 10_000)))
            comparisons[f"mask_ratio_{ratio:.2f}"] = reconstruction_validation(model, audit_loader, device, args.amp, ratio)
        runtime["final_reconstruction_validation"] = final_validation
        runtime["comparison_reconstruction_validations"] = comparisons
        runtime["resolved_config_hash"] = hashlib.sha256(canonical_json(runtime).encode("utf-8")).hexdigest()
        with (variant_dir / "run_metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(runtime, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        checkpoint = {
            "variant": "small",
            "model_state": model.state_dict(),
            "config": {
                "img_size": (1024, 128), "patch_size": 16, "norm_pix_loss": True,
                "audio_exp": True, "in_chans": 1, "mask_ratio": args.mask_ratio,
                "objective": "audiomae_masked_reconstruction", "runtime": runtime,
            },
            "final_reconstruction_validation": final_validation,
            "comparison_reconstruction_validations": comparisons,
        }
        final_path = variant_dir / "audioset_mae_vit_small_final.pth"
        torch.save(checkpoint, final_path)
        logger.info("Final reconstruction validation=%s", canonical_json(final_validation))
        logger.info("Final checkpoint=%s sha256=%s elapsed=%.1fs", final_path.resolve(), sha256_file(final_path), time.time() - started_at)
    except Exception:
        logger.exception("AudioSet-MAE pre-training failed")
        raise
    finally:
        del model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(parse_args())


if __name__ == "__main__":
    main()
