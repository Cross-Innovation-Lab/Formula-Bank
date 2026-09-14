#!/usr/bin/env python
"""FDSL-style AudioMAE pre-training on a frozen Formula Bank.

Each epoch visits every materialized (template, instance) waveform exactly
once.  Formula IDs are used solely to construct and audit that schedule; the
AudioMAE reconstruction loss receives only Log-Mel inputs.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import pretrain_sl as core
from pretrain import autocast_cuda, build_model, count_parameters, make_grad_scaler


PROTOCOL_NAME = "audiopg_epoch_scaled_audiomae_v1"


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_logger(path):
    logger = logging.getLogger("pretrain_mae_fdsl")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, mode="x", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class FDSLFormulaExposureDataset(core.FormulaExposureDataset):
    """Expose every frozen instance once per epoch and return its ID for audit."""

    def instance_metadata(self, idx):
        global_exposure = self._epoch.value * self.exposure_count + int(idx)
        label = global_exposure % self.bank.num_classes
        occurrence = global_exposure // self.bank.num_classes
        return int(label), self._permuted_instance_id(label, occurrence)

    def __getitem__(self, idx):
        features, label, rejected_attempts = super().__getitem__(idx)
        expected_label, instance_id = self.instance_metadata(idx)
        if int(label.item()) != expected_label:
            raise RuntimeError("Formula dataset label disagrees with FDSL schedule")
        return features, label, rejected_attempts, torch.tensor(instance_id, dtype=torch.long)


def reconstruction_validation(model, loader, device, amp_enabled, mask_ratio):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_rejected_attempts = 0
    with torch.no_grad():
        for images, _labels, rejected_attempts in tqdm(loader, desc="Formula-MAE validation", leave=False):
            images = images.to(device, non_blocking=True)
            with autocast_cuda(enabled=amp_enabled):
                loss, _predictions, _mask = model(images, mask_ratio=mask_ratio)
            batch_size = int(images.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            total_rejected_attempts += int(rejected_attempts.sum().item())
    model.train()
    return {
        "loss": total_loss / max(1, total_samples),
        "samples": total_samples,
        "rejected_attempts": total_rejected_attempts,
    }


def output_paths(output_dir):
    variant_dir = Path(output_dir) / "vit_small"
    final_path = variant_dir / "formula_mae_fdsl_vit_small_final.pth"
    log_path = variant_dir / "train.log"
    if variant_dir.exists() or final_path.exists() or log_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing FDSL-MAE artifact: {variant_dir}")
    variant_dir.mkdir(parents=True, exist_ok=False)
    return variant_dir, final_path, log_path


def runtime_metadata(args, bank, unique_per_epoch, steps_per_epoch, final_batch_size):
    complete_batches, tail_size = divmod(unique_per_epoch, args.batch_size)
    if tail_size == 0:
        tail_size = args.batch_size
        complete_batches -= 1
    return {
        "protocol": PROTOCOL_NAME,
        "formal_protocol": False,
        "development_run": args.dev_run,
        "objective": "audiomae_masked_reconstruction",
        "formula_labels_used_in_loss": False,
        "formula_bank_sha256": bank.bank_hash,
        "bank_subset": bank.subset,
        "num_templates": bank.num_classes,
        "instances_per_template": args.instances_per_class,
        "unique_training_samples_per_epoch": unique_per_epoch,
        "epochs": args.epochs,
        "per_instance_repetitions": args.epochs,
        "total_exposures": args.epochs * unique_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": args.epochs * steps_per_epoch,
        "sampler": "single_gpu_each_frozen_instance_once_per_epoch_no_padding_or_drop",
        "batch_size": args.batch_size,
        "complete_batches_before_final": complete_batches,
        "final_batch_size": final_batch_size,
        "final_batch_policy": "retain_incomplete_batch_with_nominal_adamw_lr",
        "model": "small",
        "img_size": [1024, 128],
        "patch_size": 16,
        "mask_ratio": args.mask_ratio,
        "norm_pix_loss": True,
        "optimizer": "AdamW",
        "lr": args.lr,
        "betas": [0.9, 0.95],
        "weight_decay": args.weight_decay,
        "pretraining_llrd": None,
        "seed": args.seed,
        "pool_seed": args.pool_seed,
        "amp": args.amp,
        "max_render_attempts": args.max_render_attempts,
        "renderer_source_sha256": sha256_file(Path(core.__file__).resolve()),
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
    }


def run(args):
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu" and not args.cpu:
        raise RuntimeError("CUDA is unavailable; use --cpu only for a development smoke test")

    bank = core.FormulaBank(args.formula_bank, args.bank_subset, args.subset_registry)
    unique_per_epoch = bank.num_classes * args.instances_per_class
    steps_per_epoch = (unique_per_epoch + args.batch_size - 1) // args.batch_size
    final_batch_size = unique_per_epoch % args.batch_size or args.batch_size
    variant_dir, final_path, log_path = output_paths(args.output_dir)
    logger = build_logger(log_path)

    try:
        train_dataset = FDSLFormulaExposureDataset(
            bank=bank,
            split="train",
            seed=args.pool_seed,
            instances_per_class=args.instances_per_class,
            exposure_count=unique_per_epoch,
            max_render_attempts=args.max_render_attempts,
        )
        val_dataset = core.FormulaExposureDataset(
            bank=bank,
            split="val",
            seed=args.pool_seed,
            instances_per_class=args.val_instances_per_class,
            max_render_attempts=args.max_render_attempts,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
        if len(train_loader) != steps_per_epoch:
            raise RuntimeError("DataLoader violates the exact FDSL once-per-epoch schedule")

        runtime = runtime_metadata(args, bank, unique_per_epoch, steps_per_epoch, final_batch_size)
        runtime["train_pool_hash"] = train_dataset.pool_hash
        runtime["val_pool_hash"] = val_dataset.pool_hash
        logger.info("FDSL-AudioMAE training starts: %s", canonical_json(runtime))
        logger.info(
            "Formula labels are used only for the frozen sampler and exposure audit; "
            "the reconstruction loss receives only Log-Mel inputs."
        )

        model = build_model("small").to(device)
        logger.info("Model=AudioMAE ViT-SMALL trainable_parameters=%.2fM", count_parameters(model) / 1e6)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        amp_enabled = device.type == "cuda" and args.amp
        scaler = make_grad_scaler(enabled=amp_enabled)
        instance_exposures = torch.zeros((bank.num_classes, args.instances_per_class), dtype=torch.long)
        rejection_exposures = torch.zeros(bank.num_classes, dtype=torch.long)
        global_step = 0
        started_at = time.time()

        for epoch in range(args.epochs):
            train_dataset.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            total_samples = 0
            for images, labels, rejected_attempts, instance_ids in tqdm(
                train_loader, desc=f"FDSL-MAE {epoch + 1}/{args.epochs}"
            ):
                flat_indices = labels.to(torch.long) * args.instances_per_class + instance_ids.to(torch.long)
                instance_exposures.view(-1).index_add_(
                    0, flat_indices, torch.ones_like(flat_indices, dtype=torch.long)
                )
                rejection_exposures.index_add_(0, labels.to(torch.long), rejected_attempts.to(torch.long))
                images = images.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_cuda(enabled=amp_enabled):
                    loss, _predictions, _mask = model(images, mask_ratio=args.mask_ratio)
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                batch_size = int(images.shape[0])
                total_loss += float(loss.detach().item()) * batch_size
                total_samples += batch_size
                global_step += 1
            logger.info(
                "Epoch %d/%d step=%d/%d loss=%.6f samples=%d elapsed=%.1fs",
                epoch + 1,
                args.epochs,
                global_step,
                runtime["total_steps"],
                total_loss / max(1, total_samples),
                total_samples,
                time.time() - started_at,
            )

        expected = torch.full_like(instance_exposures, args.epochs)
        if not torch.equal(instance_exposures, expected):
            raise RuntimeError("Instance exposure audit failed: not every frozen instance appeared once per epoch")
        torch.manual_seed(args.pool_seed + 999_983)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.pool_seed + 999_983)
        final_validation = reconstruction_validation(model, val_loader, device, amp_enabled, args.mask_ratio)
        comparison_validations = {}
        for evaluation_mask_ratio in args.eval_mask_ratios:
            evaluation_seed = args.pool_seed + 1_000_000 + int(round(evaluation_mask_ratio * 10_000))
            torch.manual_seed(evaluation_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(evaluation_seed)
            comparison_validations[f"mask_ratio_{evaluation_mask_ratio:.2f}"] = reconstruction_validation(
                model,
                val_loader,
                device,
                amp_enabled,
                evaluation_mask_ratio,
            )
        runtime["instance_exposure_audit"] = {
            template["class_id"]: {
                "min_instance_exposures": int(instance_exposures[index].min().item()),
                "max_instance_exposures": int(instance_exposures[index].max().item()),
                "total_exposures": int(instance_exposures[index].sum().item()),
                "rejected_attempts": int(rejection_exposures[index].item()),
            }
            for index, template in enumerate(bank.templates)
        }
        runtime["final_reconstruction_validation"] = final_validation
        runtime["comparison_reconstruction_validations"] = comparison_validations
        runtime["resolved_config_hash"] = hashlib.sha256(canonical_json(runtime).encode("utf-8")).hexdigest()
        with (variant_dir / "run_metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(runtime, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        checkpoint = {
            "variant": "small",
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
            "comparison_reconstruction_validations": comparison_validations,
        }
        torch.save(checkpoint, final_path)
        logger.info("Final reconstruction validation=%s", canonical_json(final_validation))
        if comparison_validations:
            logger.info("Comparison reconstruction validations=%s", canonical_json(comparison_validations))
        logger.info("Final checkpoint=%s sha256=%s elapsed=%.1fs", final_path.resolve(), sha256_file(final_path), time.time() - started_at)
    except Exception:
        logger.exception("FDSL-AudioMAE pre-training failed")
        raise
    finally:
        if "model" in locals():
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="FDSL-style AudioMAE pre-training on a frozen Formula Bank")
    parser.add_argument("--formula-bank", required=True)
    parser.add_argument("--subset-registry", default="")
    parser.add_argument("--bank-subset", required=True)
    parser.add_argument("--instances-per-class", required=True, type=int)
    parser.add_argument("--val-instances-per-class", default=50, type=int)
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--mask-ratio", default=0.75, type=float)
    parser.add_argument(
        "--eval-mask-ratios",
        default=[],
        nargs="*",
        type=float,
        help="Additional fixed mask ratios for final reconstruction validation.",
    )
    parser.add_argument("--grad-clip", default=0.0, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--max-render-attempts", default=16, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument(
        "--pool-seed",
        default=None,
        type=int,
        help="Frozen FormulaBank pool seed. Defaults to --seed for backward compatibility.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    args = parser.parse_args()
    if args.pool_seed is None:
        args.pool_seed = args.seed
    if args.instances_per_class <= 0 or args.val_instances_per_class <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error("instance counts, epochs, and batch size must be positive")
    if args.num_workers < 0 or args.max_render_attempts <= 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("invalid runtime or optimizer arguments")
    if not 0 < args.mask_ratio < 1 or args.grad_clip < 0:
        parser.error("invalid mask ratio or gradient clip")
    if any(not 0 < ratio < 1 for ratio in args.eval_mask_ratios):
        parser.error("evaluation mask ratios must lie strictly between zero and one")
    return args


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(parse_args())


if __name__ == "__main__":
    main()
