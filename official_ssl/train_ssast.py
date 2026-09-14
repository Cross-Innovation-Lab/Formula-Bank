"""DDP pretraining adapter for the official SSAST implementation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
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
from official_ssl.synthetic import SyntheticStream


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "ssast" / "src"))
from models.ast_models import ASTModel  # noqa: E402


class SSASTJointObjective(nn.Module):
    """Keep MPC and MPG inside one DDP forward/reduction graph."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, fbank):
        acc, nce = self.base_model(fbank, "pretrain_mpc", mask_patch=400, cluster=True)
        mse = self.base_model(fbank, "pretrain_mpg", mask_patch=400, cluster=True)
        return acc, nce, mse


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
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
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


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_acc, total_mse, total_count = 0.0, 0.0, 0
    for fbank, _ in loader:
        fbank = fbank.to(device, non_blocking=True)
        acc, _, mse = model(fbank)
        count = fbank.shape[0]
        total_acc += acc.item() * count
        total_mse += mse.item() * count
        total_count += count
    total_acc, total_mse, total_count = reduce_sums(
        [total_acc, total_mse, total_count], device
    )
    return total_acc / max(total_count, 1), total_mse / max(total_count, 1)


def native_state_dict(model):
    module = model.module if isinstance(model, DDP) else model
    return {
        f"module.{key}": value.detach().cpu()
        for key, value in module.base_model.state_dict().items()
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
        logger = make_logger(output_dir / "logs" / "pretrain.log", "official_ssast")
        write_json(output_dir / "run.json", {
            "method": "SSAST", "upstream": "YuanGongND/ssast",
            "model": "base", "initialization": "random", "epochs": args.epochs,
            "global_epoch_len": args.epoch_len, "world_size": world_size,
            "per_gpu_batch_size": per_gpu_batch, "grad_accum": args.grad_accum,
            "optimizer_global_batch_size": optimizer_global_batch,
            "expected_forward_steps_per_epoch": (args.epoch_len // world_size) // per_gpu_batch,
            "expected_optimizer_updates_per_epoch": (
                (args.epoch_len // world_size) // per_gpu_batch // args.grad_accum
            ),
            "seed": args.seed, "synth_config": str(Path(args.synth_config).resolve()),
            "objective": "MPC + 10 * MPG", "mask_patch": 400,
            "masking": "clustered 16x16 non-overlapping patches",
        })
    else:
        logger = None
    barrier()

    train_set = SyntheticStream(args.synth_config, "ssast", args.epoch_len, args.seed)
    val_set = SyntheticStream(args.synth_config, "ssast", args.val_len, args.seed, validation=True)
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
    model = SSASTJointObjective(ASTModel(
        fshape=16, tshape=16, fstride=16, tstride=16,
        input_fdim=128, input_tdim=1024, model_size="base", pretrain_stage=True,
    )).to(device)
    if world_size > 1:
        # DeiT's original classification heads are intentionally unused by the
        # SSAST pretext task but remain in the unmodified upstream backbone.
        model = DDP(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=5e-7, betas=(0.95, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    global_update = 0
    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch)
        val_set.set_epoch(epoch)
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if val_sampler:
            val_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = total_acc = total_nce = total_mse = 0.0
        total_count = 0
        updates_this_epoch = 0
        for step, (fbank, _) in enumerate(train_loader, start=1):
            fbank = fbank.to(device, non_blocking=True)
            acc, nce, mse = model(fbank)
            loss = nce + 10.0 * mse
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                global_update += 1
                updates_this_epoch += 1
                if global_update <= 1000:
                    warmup_lr = 1e-4 * global_update / 1000.0
                    for group in optimizer.param_groups:
                        group["lr"] = warmup_lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            count = fbank.shape[0]
            total_loss += loss.detach().item() * count
            total_acc += acc.detach().item() * count
            total_nce += nce.detach().item() * count
            total_mse += mse.detach().item() * count
            total_count += count
            if args.smoke and step >= 1:
                break
        totals = reduce_sums([total_loss, total_acc, total_nce, total_mse, total_count], device)
        val_acc, val_mse = validate(model, val_loader, device)
        if is_main_process():
            scheduler.step(val_acc)
            message = (
                f"epoch={epoch}/{args.epochs} loss={totals[0] / totals[4]:.5f} "
                f"mpc_acc={totals[1] / totals[4]:.4f} nce={totals[2] / totals[4]:.5f} "
                f"mpg={totals[3] / totals[4]:.5f} val_mpc_acc={val_acc:.4f} "
                f"val_mpg={val_mse:.5f} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"updates={updates_this_epoch}"
            )
            logger.info(message)
            if epoch % args.save_every == 0 and not args.smoke:
                torch.save(native_state_dict(model), output_dir / "checkpoints" / f"epoch_{epoch:03d}.pth")
        if world_size > 1:
            lr = torch.tensor([optimizer.param_groups[0]["lr"]], device=device)
            dist.broadcast(lr, src=0)
            for group in optimizer.param_groups:
                group["lr"] = lr.item()
        barrier()
        if args.smoke:
            break
    if is_main_process():
        final_path = output_dir / "checkpoints" / "ssast_base_final.pth"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(native_state_dict(model), final_path)
        logger.info("saved final checkpoint: %s", final_path)
    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
