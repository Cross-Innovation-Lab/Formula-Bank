#!/usr/bin/env python
"""FDSL-style epoch-scaled supervised pre-training for AudioPG.

Unlike pretrain_sl.py's fixed-compute protocol, each epoch visits every
materialised category x instance waveform exactly once.  Thus a larger bank
has proportionally more steps at a fixed epoch count, as in FDSL.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pretrain_sl as core


PROTOCOL_NAME = "audiopg_epoch_scaled_supervised_v2"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def setup_distributed(force_cpu):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This FDSL protocol is intentionally single-GPU; do not launch it with torchrun")
    if force_cpu:
        return 0, 1, torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; --cpu is only for development smoke tests")
    return 0, 1, torch.device("cuda")


def barrier(world_size, device):
    del world_size, device


def unwrap(model):
    return model


def logger_for(path, enabled):
    logger = logging.getLogger("pretrain_sl_fdsl")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, mode="x", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_lock(path_arg, bank, formal):
    """Fail closed when code, bank, and frozen protocol lock do not agree."""
    if not formal:
        return None
    if not path_arg:
        raise ValueError("Formal runs require --protocol-lock")
    path = Path(path_arg)
    with path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("schema_version") != 1 or lock.get("status") != "frozen":
        raise ValueError("Protocol lock must be frozen schema v1")
    if lock.get("protocol_name") != PROTOCOL_NAME:
        raise ValueError("Protocol lock belongs to a different protocol")
    if lock.get("formula_bank_sha256") != bank.bank_hash:
        raise ValueError("Protocol lock does not match Formula Bank")
    if lock.get("base_bank_admission_report_hash") != bank.admission_report.get("report_hash"):
        raise ValueError("Protocol lock does not match Formula Bank admission report")
    renderer_hash = sha256_file(Path(core.__file__).resolve())
    entry_hash = sha256_file(Path(__file__).resolve())
    if lock.get("renderer_source_sha256") != renderer_hash:
        raise RuntimeError("Renderer source drift: do not relabel this as the frozen FDSL protocol")
    if lock.get("entrypoint_sha256") != entry_hash:
        raise RuntimeError("Entrypoint drift: freeze a new FDSL protocol lock before running")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "lock_id": lock.get("lock_id", "")}


def validate_formal(args, world_size, unique_per_epoch):
    if args.dev_run:
        return False
    expected = {
        "model": "small", "epochs": 500, "global_batch_size": 256,
        "batch_size": 32,
        "lr": 1e-3, "warmup_lr": 1e-4, "min_lr": 1e-5,
        "warmup_epochs": 5, "weight_decay": 0.05, "layer_decay": 0.75,
        "label_smoothing": 0.1, "drop_path": 0.1, "seed": 2026,
    }
    mismatches = [f"{key}={getattr(args, key)!r} (expected {value!r})" for key, value in expected.items() if getattr(args, key) != value]
    if world_size != 1:
        mismatches.append("formal FDSL protocol uses one GPU")
    if args.global_batch_size % args.batch_size:
        mismatches.append("global batch must be an integer multiple of the single-GPU micro-batch")
    if mismatches:
        raise ValueError("Formal FDSL protocol mismatch: " + "; ".join(mismatches))
    return True


def optimizer_groups_with_llrd(model, weight_decay, layer_decay):
    """Build AdamW groups with a stable ViT layer-wise learning-rate scale."""
    depth = len(model.blocks)
    max_layer_id = depth + 1
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("patch_embed.") or name in {"cls_token", "pos_embed"}:
            layer_id = 0
        elif name.startswith("blocks."):
            layer_id = int(name.split(".", 2)[1]) + 1
        else:
            layer_id = max_layer_id
        no_decay = parameter.ndim == 1 or name.endswith(".bias") or name.endswith("cls_token")
        key = (layer_id, no_decay)
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "lr_scale": layer_decay ** (max_layer_id - layer_id),
                "layer_id": layer_id,
            }
        grouped[key]["params"].append(parameter)
    return [grouped[key] for key in sorted(grouped)]


def set_group_learning_rate(optimizer, base_lr):
    for group in optimizer.param_groups:
        group["lr"] = base_lr * group["lr_scale"]


def reduce_values(values, device, world_size):
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def evaluate(model, dataset, device, amp, batch_size):
    """Rank-zero-only evaluation avoids DistributedSampler validation padding."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    model.eval()
    criterion, totals = nn.CrossEntropyLoss(), [0.0, 0.0, 0.0, 0]
    with torch.no_grad():
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device)
            with core.autocast_cuda(enabled=amp):
                logits, loss = model(images), None
                loss = criterion(logits, targets)
            top1, top5 = core._topk_correct(logits, targets)
            count = int(targets.shape[0])
            totals[0] += float(loss.item()) * count
            totals[1] += top1
            totals[2] += top5
            totals[3] += count
    model.train()
    return {"loss": totals[0] / totals[3], "top1": 100 * totals[1] / totals[3], "top5": 100 * totals[2] / totals[3], "samples": totals[3]}


def save_final(path, model, args, bank, runtime, train_metrics, val_metrics):
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    encoder = {key: value for key, value in state.items() if not key.startswith("head.")}
    head = {key[5:]: value for key, value in state.items() if key.startswith("head.")}
    payload = {"variant": args.model, "model_state": encoder, "formula_head_state": head,
               "epoch": args.epochs, "global_step": runtime["total_steps"], "seed": args.seed,
               "config": runtime, "bank_hash": bank.bank_hash, "label_mapping": bank.label_mapping,
               "final_train_metrics": train_metrics, "final_validation_metrics": val_metrics}
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def run(args, rank, world_size, device):
    is_main = rank == 0
    bank = core.FormulaBank(args.formula_bank, args.bank_subset, args.subset_registry)
    unique_per_epoch = bank.num_classes * args.instances_per_class
    formal = validate_formal(args, world_size, unique_per_epoch)
    lock = load_lock(args.protocol_lock, bank, formal)
    if args.global_batch_size % args.batch_size:
        raise ValueError("global batch must be an integer multiple of --batch-size for exact accumulation")
    micro_batches_per_update = args.global_batch_size // args.batch_size
    # FDSL fixes epochs and normally keeps a final incomplete minibatch.  Each
    # rank receives an equal shard, so all ranks execute the same ceil() number
    # of steps while the union of shards is exactly the C x I dataset.
    steps_per_epoch = (unique_per_epoch + args.global_batch_size - 1) // args.global_batch_size
    complete_global_batches, final_global_batch = divmod(unique_per_epoch, args.global_batch_size)
    if final_global_batch == 0:
        final_global_batch = args.global_batch_size
        complete_global_batches -= 1
    final_micro_batch = final_global_batch % args.batch_size or args.batch_size
    output = Path(args.output_dir) / "vit_small"
    if is_main:
        output.mkdir(parents=True, exist_ok=False)
    barrier(world_size, device)
    log = logger_for(output / "train.log", is_main)
    try:
        train_set = core.FormulaExposureDataset(bank, "train", args.seed, args.instances_per_class, unique_per_epoch, args.max_render_attempts)
        val_set = core.FormulaExposureDataset(bank, "val", args.seed, args.val_instances_per_class, max_render_attempts=args.max_render_attempts)
        sampler = DistributedSampler(train_set, num_replicas=1, rank=0, shuffle=True, seed=args.seed, drop_last=True)
        loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda", drop_last=False, persistent_workers=args.num_workers > 0)
        expected_micro_batches = (unique_per_epoch + args.batch_size - 1) // args.batch_size
        if len(loader) != expected_micro_batches:
            raise RuntimeError("DataLoader violates the exact once-per-epoch FDSL schedule")
        runtime = {
            "protocol": PROTOCOL_NAME, "formal_protocol": formal, "formula_bank_sha256": bank.bank_hash,
            "bank_subset": bank.subset, "num_classes": bank.num_classes, "instances_per_class": args.instances_per_class,
            "unique_training_samples_per_epoch": unique_per_epoch, "epochs": args.epochs,
            "global_batch_size": args.global_batch_size, "micro_batch_size": args.batch_size,
            "micro_batches_per_full_update": micro_batches_per_update, "world_size": world_size, "steps_per_epoch": steps_per_epoch,
            "total_steps": args.epochs * steps_per_epoch, "total_exposures": args.epochs * unique_per_epoch,
            "sampler": "single_gpu_each_frozen_instance_once_per_epoch_no_padding_or_drop", "optimizer": "AdamW",
            "complete_global_batches_before_final": complete_global_batches,
            "final_global_batch_size": final_global_batch,
            "final_micro_batch_size": final_micro_batch,
            "final_batch_policy": "retain_incomplete_batch_with_nominal_adamw_lr",
            "peak_lr": args.lr, "warmup_lr": args.warmup_lr, "min_lr": args.min_lr,
            "warmup_epochs": args.warmup_epochs, "weight_decay": args.weight_decay,
            "layer_decay": args.layer_decay, "label_smoothing": args.label_smoothing,
            "drop_path": args.drop_path, "seed": args.seed, "amp": args.amp,
            "renderer_source_sha256": sha256_file(Path(core.__file__).resolve()),
            "entrypoint_sha256": sha256_file(Path(__file__).resolve()), "protocol_lock": lock,
        }
        if is_main:
            log.info("FDSL-style training starts: %s", canonical_json(runtime))
        model = core.FormulaViTClassifier(bank.num_classes, args.model, args.drop_path).to(device)
        optimizer = optim.AdamW(
            optimizer_groups_with_llrd(model, args.weight_decay, args.layer_decay),
            lr=args.lr,
            betas=(0.9, 0.999),
        )
        criterion, amp = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing), device.type == "cuda" and args.amp
        scaler = core.make_grad_scaler(enabled=amp)
        observed = torch.zeros(bank.num_classes, dtype=torch.long, device=device)
        rejected = torch.zeros_like(observed)
        final_metrics, step, started = None, 0, time.time()
        for epoch in range(args.epochs):
            train_set.set_epoch(epoch)
            sampler.set_epoch(epoch)
            totals = [0.0, 0.0, 0.0, 0, 0]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            samples_in_update = 0
            current_update_size = min(args.global_batch_size, unique_per_epoch)
            updates_in_epoch = 0
            for images, targets, rejections in loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                rejections = rejections.to(device, dtype=torch.long, non_blocking=True)
                observed.scatter_add_(0, targets, torch.ones_like(targets, dtype=torch.long))
                rejected.scatter_add_(0, targets, rejections)
                with core.autocast_cuda(enabled=amp):
                    logits = model(images)
                    loss = criterion(logits, targets)
                count = int(targets.shape[0])
                scaler.scale(loss * (count / current_update_size)).backward()
                samples_in_update += count
                if samples_in_update == current_update_size:
                    lr = core._learning_rate_at_step(
                        step,
                        runtime["total_steps"],
                        args.warmup_epochs * steps_per_epoch,
                        args.lr,
                        args.warmup_lr,
                        args.min_lr,
                    )
                    set_group_learning_rate(optimizer, lr)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    updates_in_epoch += 1
                    remaining = unique_per_epoch - updates_in_epoch * args.global_batch_size
                    if remaining > 0:
                        current_update_size = min(args.global_batch_size, remaining)
                    samples_in_update = 0
                elif samples_in_update > current_update_size:
                    raise RuntimeError("micro-batch crossed an effective-batch accumulation boundary")
                top1, top5, count = *core._topk_correct(logits.detach(), targets), int(targets.shape[0])
                totals[0] += float(loss.item()) * count
                totals[1] += top1
                totals[2] += top5
                totals[3] += count
                totals[4] += int(rejections.sum().item())
            if updates_in_epoch != steps_per_epoch or samples_in_update != 0:
                raise RuntimeError("effective-batch accumulation did not finish exactly at the epoch boundary")
            values = reduce_values(totals, device, world_size)
            final_metrics = {"loss": values[0] / values[3], "top1": 100 * values[1] / values[3], "top5": 100 * values[2] / values[3], "samples": int(values[3]), "rejected_attempts": int(values[4]), "lr": lr}
            if is_main:
                log.info("Epoch %d/%d step=%d/%d lr=%.5g loss=%.6f top1=%.3f top5=%.3f", epoch + 1, args.epochs, step, runtime["total_steps"], lr, final_metrics["loss"], final_metrics["top1"], final_metrics["top5"])
        expected = torch.full_like(observed, args.epochs * args.instances_per_class)
        if not torch.equal(observed, expected):
            raise RuntimeError("Class exposure audit failed: not every C x I item appeared once per epoch")
        if is_main:
            validation = evaluate(model, val_set, device, amp, args.batch_size)
            runtime["class_exposure_audit"] = {template["class_id"]: {"exposures": int(observed[i]), "rejected_attempts": int(rejected[i])} for i, template in enumerate(bank.templates)}
            runtime["final_validation"] = validation
            with (output / "run_metadata.json").open("x", encoding="utf-8") as handle:
                json.dump(runtime, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            final = output / "formula_sl_fdsl_vit_small_final.pth"
            save_final(final, unwrap(model), args, bank, runtime, final_metrics, validation)
            log.info("Final validation=%s", canonical_json(validation))
            log.info("Final checkpoint=%s sha256=%s elapsed=%.1fs", final.resolve(), sha256_file(final), time.time() - started)
        barrier(world_size, device)
    except Exception:
        log.exception("FDSL-style pre-training failed")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="FDSL-style AudioPG supervised pre-training")
    parser.add_argument("--formula-bank", required=True)
    parser.add_argument("--bank-subset", required=True)
    parser.add_argument("--subset-registry", default="")
    parser.add_argument("--protocol-lock", default="")
    parser.add_argument("--instances-per-class", required=True, type=int)
    parser.add_argument("--val-instances-per-class", default=50, type=int)
    parser.add_argument("--model", choices=["small"], default="small")
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--global-batch-size", default=256, type=int)
    parser.add_argument("--batch-size", default=32, type=int, help="Single-GPU micro-batch; gradients accumulate to global batch")
    parser.add_argument("--lr", default=1e-3, type=float, help="Peak AdamW learning rate at effective batch 256")
    parser.add_argument("--warmup-lr", default=1e-4, type=float)
    parser.add_argument("--min-lr", default=1e-5, type=float)
    parser.add_argument("--warmup-epochs", default=5, type=int)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--layer-decay", default=0.75, type=float)
    parser.add_argument("--label-smoothing", default=0.1, type=float)
    parser.add_argument("--drop-path", default=0.1, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--max-render-attempts", default=16, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--output-dir", default="experiments/formula_class_instance_fdsl/pretrain")
    parser.add_argument("--dev-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    args = parser.parse_args()
    if args.instances_per_class <= 0 or args.val_instances_per_class <= 0 or args.epochs <= 0 or args.global_batch_size <= 0 or args.batch_size <= 0:
        parser.error("instance counts, epochs, and global batch must be positive")
    if args.num_workers < 0 or args.max_render_attempts <= 0 or args.lr <= 0 or args.warmup_lr < 0 or args.min_lr < 0:
        parser.error("invalid runtime or AdamW learning-rate arguments")
    if args.warmup_epochs < 0 or not 0 < args.layer_decay <= 1 or not 0 <= args.label_smoothing < 1 or not 0 <= args.drop_path < 1:
        parser.error("invalid regularization or LLRD arguments")
    return args


def main():
    args = parse_args()
    # Required by CUDA >= 10.2 when exact deterministic CuBLAS algorithms are
    # requested below.  Set it before CUDA context creation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank, world_size, device = setup_distributed(args.cpu)
    try:
        core.set_seed(args.seed + rank)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        run(args, rank, world_size, device)
    finally:
        pass


if __name__ == "__main__":
    main()
