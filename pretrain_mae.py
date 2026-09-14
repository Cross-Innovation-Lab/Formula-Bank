#!/usr/bin/env python
"""AudioMAE reconstruction pre-training on a frozen Formula Bank.

This is a deliberately separate entry point for the Formula Class × Instance
control experiment.  It preserves the AudioMAE model, frontend, random masking,
normalized-pixel reconstruction loss, and AdamW settings from ``pretrain.py``.
It replaces only the free-running ``PhysicsDataset`` with the manifest-driven,
finite-instance Formula Bank dataset used by ``pretrain_sl.py``.

Formula template IDs are returned by the dataset solely for balanced sampling
and audit logs.  They never enter the model or the reconstruction loss.
"""

import argparse
import collections
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pretrain import autocast_cuda, build_model, count_parameters, make_grad_scaler
from pretrain_sl import FormulaBank, FormulaExposureDataset
try:
    from formula_property_dataset import PropertyExposureDataset
except ImportError:
    PropertyExposureDataset = None


def _canonical_json(payload):
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _configure_logger(log_path, variant):
    logger = logging.getLogger(f"formula_mae_{variant}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="x")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _resolved_runtime(args, bank, exposure_count, steps_per_epoch):
    total_steps = args.epochs * steps_per_epoch
    total_exposures = args.epochs * exposure_count
    return {
        "objective": "audiomae_masked_reconstruction",
        "formula_labels_used_in_loss": False,
        "formal_protocol": not args.dev_run,
        "bank_id": bank.bank_id,
        "bank_version": bank.payload.get("bank_version"),
        "bank_hash": bank.bank_hash,
        "bank_subset": bank.subset,
        "num_templates": bank.num_classes,
        "formula_bank_path": str(Path(args.formula_bank).resolve()),
        "formula_bank_admission": bank.payload.get("admission_report"),
        "instances_per_template": args.instances_per_class,
        "unique_training_samples": bank.num_classes * args.instances_per_class,
        "train_pool_hash": None,
        "val_instances_per_template": args.val_instances_per_class,
        "sample_rate": bank.sample_rate,
        "clip_duration_s": bank.clip_duration_s,
        "img_size": [1024, 128],
        "patch_size": 16,
        "variant": args.models[0],
        "mask_ratio": args.mask_ratio,
        "norm_pix_loss": True,
        "optimizer": "AdamW",
        "betas": [0.9, 0.95],
        "weight_decay": args.weight_decay,
        "lr": args.lr,
        "global_batch_size": args.global_batch_size,
        "epochs": args.epochs,
        "epoch_len_nominal": args.epoch_len,
        "exposures_per_epoch": exposure_count,
        "steps_per_epoch": steps_per_epoch,
        "total_exposures": total_exposures,
        "total_steps": total_steps,
        "mean_exposures_per_unique_training_sample": (
            total_exposures / (bank.num_classes * args.instances_per_class)
        ),
        "seed": args.seed,
        "amp": args.amp,
        "max_render_attempts": args.max_render_attempts,
        "run_name": args.run_name,
    }


def _save_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_output_dir(output_dir, variant):
    variant_dir = Path(output_dir) / f"vit_{variant}"
    final_path = variant_dir / f"formula_mae_vit_{variant}_final.pth"
    log_path = variant_dir / "train.log"
    if final_path.exists() or log_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Formula-MAE output: {variant_dir}. "
            "Use a new --output-dir."
        )
    variant_dir.mkdir(parents=True, exist_ok=False)
    return variant_dir, final_path, log_path


def _run_reconstruction_validation(model, dataloader, device, amp_enabled, mask_ratio):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_rejected_attempts = 0
    with torch.no_grad():
        for images, _labels, rejected_attempts in tqdm(
            dataloader, desc="Formula-MAE validation", leave=False
        ):
            images = images.to(device, non_blocking=True)
            with autocast_cuda(enabled=amp_enabled):
                loss, _pred, _mask = model(images, mask_ratio=mask_ratio)
            batch_size = images.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            total_rejected_attempts += int(rejected_attempts.sum().item())
    model.train()
    return {
        "loss": total_loss / max(1, total_samples),
        "samples": total_samples,
        "rejected_attempts": total_rejected_attempts,
    }


def train_formula_mae(args):
    if len(args.models) != 1:
        raise ValueError("Formula-MAE runs exactly one variant per invocation")
    variant = args.models[0]
    if variant != "small":
        raise ValueError("The matched Formula-MAE control is frozen to ViT-Small")
    if args.epoch_len < args.global_batch_size:
        raise ValueError("--epoch-len must contain at least one complete global batch")

    _set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu" and not args.cpu:
        raise RuntimeError("CUDA is unavailable; use --cpu only for a development smoke test")

    bank = FormulaBank(args.formula_bank, args.bank_subset, args.subset_registry)
    exposure_count = (args.epoch_len // args.global_batch_size) * args.global_batch_size
    steps_per_epoch = exposure_count // args.global_batch_size
    variant_dir, final_path, log_path = _validate_output_dir(args.output_dir, variant)
    logger = _configure_logger(log_path, variant)

    if args.property_source_pool:
        if PropertyExposureDataset is None:
            raise RuntimeError(
                "Property-ablation inputs require the private property-ablation "
                "utilities, which are not part of this public release."
            )
        train_dataset = PropertyExposureDataset(
            bank=bank, split="train", seed=args.seed,
            instances_per_class=args.instances_per_class,
            source_pool_path=args.property_source_pool,
            variant=args.property_variant, audit_path=args.property_audit,
            exposure_count=exposure_count,
        )
        val_dataset = PropertyExposureDataset(
            bank=bank, split="val", seed=args.seed,
            instances_per_class=args.val_instances_per_class,
            source_pool_path=args.property_validation_source_pool,
            variant=args.property_variant, audit_path=args.property_audit,
        )
    else:
        train_dataset = FormulaExposureDataset(
            bank=bank, split="train", seed=args.seed,
            instances_per_class=args.instances_per_class,
            exposure_count=exposure_count,
            max_render_attempts=args.max_render_attempts,
        )
        val_dataset = FormulaExposureDataset(
            bank=bank,
            split="val",
            seed=args.seed,
            instances_per_class=args.val_instances_per_class,
            max_render_attempts=args.max_render_attempts,
        )
    runtime = _resolved_runtime(args, bank, exposure_count, steps_per_epoch)
    runtime["train_pool_hash"] = train_dataset.pool_hash
    runtime["val_pool_hash"] = val_dataset.pool_hash
    runtime["formula_renderer_sha256"] = _sha256_file(Path(__file__).with_name("pretrain_sl.py"))
    runtime["resolved_config_hash"] = hashlib.sha256(
        _canonical_json(runtime).encode("utf-8")
    ).hexdigest()
    _save_json(variant_dir / "run_metadata.json", runtime)

    logger.info("Formula-MAE run started")
    logger.info("Command=%s", " ".join(["pretrain_mae.py", *sys.argv[1:]]))
    logger.info("Runtime=%s", _canonical_json(runtime))
    logger.info(
        "Formula labels are used only for the balanced sampler and audit; "
        "the MAE reconstruction loss receives only Log-Mel inputs."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.global_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.global_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(variant).to(device)
    logger.info("Model=AudioMAE ViT-%s trainable_parameters=%.2fM", variant.upper(), count_parameters(model) / 1e6)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    amp_enabled = device.type == "cuda" and args.amp
    scaler = make_grad_scaler(enabled=amp_enabled)

    exposure_histogram = collections.Counter()
    rejection_histogram = collections.Counter()
    global_step = 0
    started_at = time.time()
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        total_loss = 0.0
        total_rejected_attempts = 0
        progress = tqdm(train_loader, desc=f"Formula-MAE {epoch + 1}/{args.epochs}")
        for images, labels, rejected_attempts in progress:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(enabled=amp_enabled):
                loss, _pred, _mask = model(images, mask_ratio=args.mask_ratio)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().item())
            total_loss += loss_value
            total_rejected_attempts += int(rejected_attempts.sum().item())
            for label, rejected in zip(labels.tolist(), rejected_attempts.tolist()):
                class_id = bank.templates[int(label)]["class_id"]
                exposure_histogram[class_id] += 1
                rejection_histogram[class_id] += int(rejected)
            global_step += 1
            progress.set_postfix(loss=f"{loss_value:.4f}")

        elapsed = time.time() - started_at
        logger.info(
            "Epoch %d/%d complete global_step=%d loss=%.6f samples=%d "
            "rejected_attempts=%d elapsed=%.1fs",
            epoch + 1,
            args.epochs,
            global_step,
            total_loss / max(1, len(train_loader)),
            exposure_count,
            total_rejected_attempts,
            elapsed,
        )

    torch.manual_seed(args.seed + 999_983)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + 999_983)
    final_validation = _run_reconstruction_validation(
        model, val_loader, device, amp_enabled, args.mask_ratio
    )
    logger.info("Observed template exposure histogram=%s", _canonical_json(dict(sorted(exposure_histogram.items()))))
    logger.info("Per-template rejection audit=%s", _canonical_json({
        class_id: {
            "exposures": exposure_histogram[class_id],
            "rejected_attempts": rejection_histogram[class_id],
            "rejected_attempts_per_exposure": rejection_histogram[class_id] / max(1, exposure_histogram[class_id]),
        }
        for class_id in sorted(exposure_histogram)
    }))
    logger.info("Final reconstruction validation=%s", _canonical_json(final_validation))

    checkpoint = {
        "variant": variant,
        "model_state": model.state_dict(),
        "config": {
            "img_size": (1024, 128),
            "patch_size": 16,
            "norm_pix_loss": True,
            "audio_exp": True,
            "in_chans": 1,
            "mask_ratio": args.mask_ratio,
            "objective": "audiomae_masked_reconstruction",
            "runtime": runtime,
        },
        "final_reconstruction_validation": final_validation,
    }
    torch.save(checkpoint, final_path)
    checkpoint_hash = _sha256_file(final_path)
    logger.info("Final checkpoint=%s sha256=%s", final_path.resolve(), checkpoint_hash)
    logger.info("Training completed successfully in %.1fs", time.time() - started_at)

    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="AudioMAE reconstruction pre-training on a frozen Formula Bank"
    )
    parser.add_argument("--formula-bank", type=str, required=True)
    parser.add_argument(
        "--subset-registry",
        type=str,
        default="",
        help="Optional frozen scale-subset registry used to resolve S14/S28 memberships",
    )
    parser.add_argument("--bank-subset", type=str, required=True)
    parser.add_argument("--instances-per-class", type=int, required=True)
    parser.add_argument("--val-instances-per-class", type=int, default=50)
    parser.add_argument("--models", nargs="+", choices=["small"], default=["small"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--epoch-len", type=int, default=5000)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-render-attempts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--property-source-pool", type=str, default="")
    parser.add_argument("--property-variant", type=str, default="")
    parser.add_argument("--property-audit", type=str, default="")
    parser.add_argument("--property-validation-source-pool", type=str, default="")
    parser.add_argument("--dev-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.epoch_len <= 0 or args.global_batch_size <= 0:
        parser.error("--epochs, --epoch-len, and --global-batch-size must be positive")
    if args.instances_per_class <= 0 or args.val_instances_per_class <= 0:
        parser.error("instance counts must be positive")
    property_args = (args.property_source_pool, args.property_variant, args.property_audit, args.property_validation_source_pool)
    if any(property_args) and not all(property_args):
        parser.error("property source pool, variant, and audit must be supplied together")
    if args.num_workers < 0 or args.max_render_attempts <= 0:
        parser.error("--num-workers must be nonnegative and --max-render-attempts must be positive")
    if not 0.0 < args.mask_ratio < 1.0:
        parser.error("--mask-ratio must lie in (0, 1)")
    return args


def main():
    train_formula_mae(parse_args())


if __name__ == "__main__":
    main()
