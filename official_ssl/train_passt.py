"""DDP synthetic supervised pretraining adapter for the official PaSST model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, DistributedSampler

from official_ssl.runtime import (
    barrier,
    cleanup_distributed,
    is_main_process,
    make_logger,
    reduce_sums,
    seed_everything,
    setup_distributed,
    write_json,
)
from official_ssl.passt_compat import prepare_passt_import
from official_ssl.synthetic import SyntheticStream


ROOT = Path(__file__).resolve().parents[1]
prepare_passt_import(ROOT)
from helpers.mixup import my_mixup  # noqa: E402
from helpers.ramp import exp_warmup_linear_down  # noqa: E402
from models.passt import get_model  # noqa: E402
from models.preprocess import AugmentMelSTFT  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--epoch-len", type=int, default=5000)
    parser.add_argument("--val-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8, help="Legacy per-GPU batch size")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=0,
        help="Optimizer batch across all ranks and accumulation steps; 0 keeps legacy batching.",
    )
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def loader_for(dataset, rank, world_size, batch_size, workers, shuffle):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=True
    ) if world_size > 1 else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=workers, pin_memory=True, drop_last=True,
        persistent_workers=workers > 0,
    ), sampler


def resolve_batching(args, world_size):
    """Resolve an optimizer-global batch without silently scaling it under DDP."""
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be positive.")
    if args.global_batch_size <= 0:
        per_gpu_batch = args.batch_size
    else:
        divisor = world_size * args.grad_accum
        if args.global_batch_size % divisor:
            raise ValueError(
                f"--global-batch-size ({args.global_batch_size}) must be divisible by "
                f"WORLD_SIZE * grad_accum ({world_size} * {args.grad_accum})."
            )
        per_gpu_batch = args.global_batch_size // divisor
    if per_gpu_batch <= 0:
        raise ValueError("Resolved per-GPU batch size must be positive.")
    return per_gpu_batch, per_gpu_batch * world_size * args.grad_accum


def fit_time_bins(features, time_bins=998):
    if features.shape[-1] >= time_bins:
        return features[..., :time_bins]
    return torch.nn.functional.pad(features, (0, time_bins - features.shape[-1]))


@torch.no_grad()
def validate(model, mel, loader, device):
    model.eval()
    mel.eval()
    total_loss = total_correct = total_items = 0.0
    for waveform, target in loader:
        waveform, target = waveform.to(device), target.to(device)
        logits, _ = model(fit_time_bins(mel(waveform)).unsqueeze(1))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        total_loss += loss.item() * waveform.shape[0]
        total_correct += ((logits.sigmoid() > 0.5) == target.bool()).float().mean(dim=1).sum().item()
        total_items += waveform.shape[0]
    total_loss, total_correct, total_items = reduce_sums(
        [total_loss, total_correct, total_items], device
    )
    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def checkpoint_payload(model, epoch, args, swa_updates):
    base_model = model.module if isinstance(model, DDP) else model
    if isinstance(base_model, AveragedModel):
        base_model = base_model.module
    return {
        "model_state": {
            key: value.detach().cpu()
            for key, value in base_model.state_dict().items()
            if key != "n_averaged"
        },
        "epoch": epoch,
        "swa_updates": swa_updates,
        "config": {
            "method": "PaSST-noImageNet", "architecture": "passt_deit_bd_p16_384",
            "initialization": "random", "n_classes": 7, "fstride": 10, "tstride": 10,
            "input_fdim": 128, "input_tdim": 998, "s_patchout_t": 40, "s_patchout_f": 4,
            "synth_config": str(Path(args.synth_config).resolve()),
        },
    }


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()
    per_gpu_batch, optimizer_global_batch = resolve_batching(args, world_size)
    seed_everything(args.seed + rank)
    output_dir = Path(args.output_dir)
    if is_main_process():
        if output_dir.exists() and any(output_dir.iterdir()) and not args.smoke:
            raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        logger = make_logger(output_dir / "logs" / "pretrain.log", "official_passt")
        write_json(output_dir / "run.json", {
            "method": "PaSST-noImageNet", "upstream": "kkoutini/PaSST",
            "model": "passt_deit_bd_p16_384", "initialization": "random",
            "epochs": args.epochs, "global_epoch_len": args.epoch_len,
            "world_size": world_size, "per_gpu_batch_size": per_gpu_batch,
            "grad_accum": args.grad_accum,
            "optimizer_global_batch_size": optimizer_global_batch,
            "expected_forward_steps_per_epoch": (args.epoch_len // world_size) // per_gpu_batch,
            "expected_optimizer_updates_per_epoch": (
                (args.epoch_len // world_size) // per_gpu_batch // args.grad_accum
            ),
            "seed": args.seed,
            "synth_config": str(Path(args.synth_config).resolve()),
            "labels": "7 primitive multi-hot labels; 1-3 unique events per scene",
            "patchout": {"time": 40, "frequency": 4}, "mixup_alpha": 0.3,
        })
    else:
        logger = None
    barrier()

    train_set = SyntheticStream(args.synth_config, "passt", args.epoch_len, args.seed)
    val_set = SyntheticStream(args.synth_config, "passt", args.val_len, args.seed, validation=True)
    train_loader, train_sampler = loader_for(
        train_set, rank, world_size, per_gpu_batch, args.workers, True
    )
    val_loader, val_sampler = loader_for(
        val_set, rank, world_size, per_gpu_batch, args.workers, False
    )
    if len(train_loader) % args.grad_accum:
        raise ValueError(
            f"{len(train_loader)} local forward steps is not divisible by "
            f"grad_accum={args.grad_accum}; this would discard a partial optimizer update."
        )
    forward_steps_per_epoch = len(train_loader)
    optimizer_updates_per_epoch = forward_steps_per_epoch // args.grad_accum
    used_samples_per_epoch = forward_steps_per_epoch * per_gpu_batch * world_size
    if is_main_process():
        logger.info(
            "batching: per_gpu=%d optimizer_global=%d forward_steps=%d "
            "optimizer_updates=%d used_samples=%d",
            per_gpu_batch,
            optimizer_global_batch,
            forward_steps_per_epoch,
            optimizer_updates_per_epoch,
            used_samples_per_epoch,
        )
    model = get_model(
        arch="passt_deit_bd_p16_384", pretrained=False, n_classes=7,
        fstride=10, tstride=10, input_fdim=128, input_tdim=998,
        s_patchout_t=40, s_patchout_f=4,
    ).to(device)
    mel = AugmentMelSTFT(
        n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024,
        freqm=48, timem=192, htk=False, fmin=0.0, fmax=None,
        norm=1, fmin_aug_range=10, fmax_aug_range=2000,
    ).to(device)
    if world_size > 1:
        # PaSST's forward uses the averaged distilled representation through
        # ``head``; the upstream ``head_dist`` parameters remain registered but
        # are intentionally not part of this loss graph.
        model = DDP(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, exp_warmup_linear_down(5, 50, 50, 0.01)
    )
    swa_model = AveragedModel(model.module if isinstance(model, DDP) else model).to(device)
    swa_updates = 0
    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch)
        val_set.set_epoch(epoch)
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if val_sampler:
            val_sampler.set_epoch(epoch)
        model.train()
        mel.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_count = 0.0, 0
        updates_this_epoch = 0
        for step, (waveform, target) in enumerate(train_loader, start=1):
            waveform, target = waveform.to(device, non_blocking=True), target.to(device, non_blocking=True)
            features = fit_time_bins(mel(waveform)).unsqueeze(1)
            indices, lam = my_mixup(waveform.shape[0], 0.3)
            indices, lam = indices.to(device), lam.to(device)
            features = features * lam.view(-1, 1, 1, 1) + features[indices] * (1.0 - lam.view(-1, 1, 1, 1))
            target = target * lam.view(-1, 1) + target[indices] * (1.0 - lam.view(-1, 1))
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, _ = model(features)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                updates_this_epoch += 1
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.detach().item() * waveform.shape[0]
            total_count += waveform.shape[0]
            if args.smoke and step >= 1:
                break
        scheduler.step()
        if epoch >= 50 and (epoch - 50) % 5 == 0:
            swa_model.update_parameters(model.module if isinstance(model, DDP) else model)
            swa_updates += 1
        val_loss, val_label_acc = validate(model, mel, val_loader, device)
        total_loss, total_count = reduce_sums([total_loss, total_count], device)
        if is_main_process():
            logger.info(
                "epoch=%d/%d loss=%.5f val_bce=%.5f val_label_acc=%.4f lr=%.2e swa_updates=%d updates=%d",
                epoch, args.epochs, total_loss / max(total_count, 1), val_loss,
                val_label_acc, optimizer.param_groups[0]["lr"], swa_updates, updates_this_epoch,
            )
            if epoch % args.save_every == 0 and not args.smoke:
                selected = swa_model if swa_updates else model
                torch.save(checkpoint_payload(selected, epoch, args, swa_updates), output_dir / "checkpoints" / f"epoch_{epoch:03d}.pth")
        barrier()
        if args.smoke:
            break
    if is_main_process():
        selected = swa_model if swa_updates else model
        final_path = output_dir / "checkpoints" / "passt_base_final.pth"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload(selected, epoch, args, swa_updates), final_path)
        logger.info("saved final checkpoint: %s", final_path)
    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
