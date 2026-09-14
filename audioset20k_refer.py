#!/usr/bin/env python
"""AudioMAE/Scratch full fine-tuning on the downloaded AudioSet-20K shards."""
import argparse
import json
import os
import random
import tarfile
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import average_precision_score
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from esc_refer import AudioMAEClassifier, get_audiomae_config, get_parameter_groups_with_llrd
from fsd50k_refer import AsymmetricLoss, MixupCutmix
from audioset20k_dataset import AudioSet20KTarDataset, waveform_to_fbank


def fbank(waveforms, precomputed=False):
    """Normalize worker-computed log-Mel features without CPU augmentation."""
    features = waveforms if precomputed else torch.stack([
        waveform_to_fbank(waveform) for waveform in waveforms
    ])
    return ((features - (-4.26)) / (4.56 * 2)).unsqueeze(1)


def apply_specaugment(features, freq_width=24, time_width=96, repeats=2):
    """Per-example SpecAugment directly on the GPU.

    The old implementation constructed four torchaudio transforms separately
    for every item of every batch on the main CPU process.  With five scaling
    jobs this consumed ~320 CPU threads and starved the GPUs.  This preserves
    the same two frequency and two time masks per example, but creates their
    boolean masks in one GPU batch operation.
    """
    batch, _, frames, bins = features.shape
    time_index = torch.arange(frames, device=features.device)[None, :, None]
    freq_index = torch.arange(bins, device=features.device)[None, None, :]
    for _ in range(repeats):
        widths = torch.randint(0, freq_width + 1, (batch,), device=features.device)
        starts = torch.floor(torch.rand(batch, device=features.device) *
                             (bins - widths + 1)).long()
        mask = (freq_index >= starts[:, None, None]) & (freq_index < (starts + widths)[:, None, None])
        features.masked_fill_(mask[:, None], 0.0)
        widths = torch.randint(0, time_width + 1, (batch,), device=features.device)
        starts = torch.floor(torch.rand(batch, device=features.device) *
                             (frames - widths + 1)).long()
        mask = (time_index >= starts[:, None, None]) & (time_index < (starts + widths)[:, None, None])
        features.masked_fill_(mask[:, None], 0.0)
    return features


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); predictions, targets = [], []
    for waveform, target in tqdm(loader, desc="AudioSet test", leave=False):
        x = fbank(waveform, precomputed=loader.dataset.precompute_fbank).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            predictions.append(model(x).sigmoid().float().cpu())
        targets.append(target)
    return float(average_precision_score(torch.cat(targets).numpy(), torch.cat(predictions).numpy(), average="macro"))


def _record_key(record):
    """Stable identifier for one clip in a downloaded tar shard."""
    archive, wav_name, _ = record
    return f"{os.path.basename(archive)}::{wav_name}"


def _labels_for_records(records):
    """Read JSON annotations in one pass per tar archive, not one per clip."""
    grouped = {}
    for index, (archive, _, json_name) in enumerate(records):
        grouped.setdefault(archive, []).append((index, json_name))
    output = [[] for _ in records]
    for archive, members in grouped.items():
        with tarfile.open(archive) as handle:
            for index, json_name in members:
                info = json.loads(handle.extractfile(json_name).read())
                output[index] = [int(label) for label in info.get("label_id", [])
                                 if 0 <= int(label) < 527]
    return output


def make_or_load_validation_split(dataset, output_dir, fraction, seed, manifest_path=""):
    """Return a deterministic, label-aware train/validation partition.

    The mirror has no official validation split.  We therefore reserve a
    fixed 10% of the original training clips.  A greedy deficit rule makes
    frequent and rare labels track their requested validation counts, while
    the seeded tie-break prevents scale-dependent splits.  The resulting
    manifest is persisted and reused on restart.
    """
    manifest = manifest_path or os.path.join(output_dir, "validation_split_seed%d.json" % seed)
    os.makedirs(os.path.dirname(manifest) or ".", exist_ok=True)
    all_records = dataset.records
    record_by_key = {_record_key(record): record for record in all_records}
    if os.path.exists(manifest):
        with open(manifest) as handle:
            saved = json.load(handle)
        val_keys = set(saved["validation_keys"])
    else:
        rng = random.Random(seed)
        labels = _labels_for_records(all_records)
        label_count = np.zeros(527, dtype=np.int64)
        for item_labels in labels:
            for label in item_labels:
                label_count[label] += 1
        desired = np.rint(label_count * fraction).astype(np.int64)
        desired[(label_count > 0) & (desired == 0)] = 1
        target_size = int(round(len(all_records) * fraction))
        # Iterate examples in a seeded order.  At every step prefer an example
        # whose labels have the largest remaining validation deficits.
        remaining = list(range(len(all_records)))
        rng.shuffle(remaining)
        val_indices, val_count = set(), np.zeros(527, dtype=np.int64)
        while remaining and len(val_indices) < target_size:
            best_pos = best_score = None
            scan = min(len(remaining), 4096)
            for pos in range(scan):
                idx = remaining[pos]
                item_labels = labels[idx]
                if not item_labels:
                    score = 0.0
                else:
                    score = sum(max(0, desired[label] - val_count[label]) /
                                max(1, desired[label]) for label in item_labels)
                if best_score is None or score > best_score:
                    best_pos, best_score = pos, score
            idx = remaining.pop(best_pos)
            val_indices.add(idx)
            for label in labels[idx]:
                val_count[label] += 1
        val_keys = {_record_key(all_records[idx]) for idx in val_indices}
        payload = {
            "seed": seed,
            "validation_fraction": fraction,
            "num_all_train": len(all_records),
            "num_validation": len(val_keys),
            "validation_keys": sorted(val_keys),
        }
        with open(manifest, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    if not val_keys or len(val_keys) >= len(record_by_key):
        raise RuntimeError("Invalid AudioSet validation split manifest")
    train_records = [record for record in all_records if _record_key(record) not in val_keys]
    val_records = [record for record in all_records if _record_key(record) in val_keys]
    return train_records, val_records, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="datasets/AudioSetMirror")
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--model-size", default="base", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--llrd-decay", type=float, default=.95)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker-fbank", action="store_true",
                        help="Compute deterministic Kaldi fbank in DataLoader workers.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.0,
                        help="Reserve this fraction of original train for model selection."
                             " Set to .1 for new scaling experiments; 0 keeps legacy test selection.")
    parser.add_argument("--validation-manifest", default="",
                        help="Shared JSON manifest for a fixed validation split.")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save resumable training state every N epochs; 0 disables it.")
    parser.add_argument("--resume", default="", help="Optional resumable training state.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    train = AudioSet20KTarDataset(args.data_path, True, 16000, 10.24, True,
                                  precompute_fbank=args.worker_fbank)
    test = AudioSet20KTarDataset(args.data_path, False, 16000, 10.24, False,
                                 precompute_fbank=args.worker_fbank)
    val = None
    if args.validation_fraction:
        if not 0.0 < args.validation_fraction < 0.5:
            raise ValueError("--validation-fraction must lie in (0, .5)")
        train_records, val_records, manifest = make_or_load_validation_split(
            train, args.output_dir, args.validation_fraction, args.seed, args.validation_manifest)
        train.records = train_records
        val = AudioSet20KTarDataset(args.data_path, True, 16000, 10.24, False,
                                    precompute_fbank=args.worker_fbank)
        val.records = val_records
        print(f"AudioSet split train={len(train)} val={len(val)} test={len(test)} manifest={manifest}", flush=True)
    workers = args.workers
    train_loader = DataLoader(train, args.batch_size, shuffle=True, drop_last=True, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    val_loader = (DataLoader(val, args.batch_size, shuffle=False, num_workers=workers, pin_memory=True,
                             persistent_workers=workers > 0) if val is not None else None)
    test_loader = DataLoader(test, args.batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    model = AudioMAEClassifier(num_classes=527, stride=16, **get_audiomae_config(args.model_size))
    if args.pretrained: model.load_pretrained_mae(args.pretrained)
    model.to(device); ema = ModelEmaV2(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(get_parameter_groups_with_llrd(model, weight_decay=.01, lr=args.lr, layer_decay=args.llrd_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion, mix = AsymmetricLoss(), MixupCutmix(.8, 1., .5, .5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.
    start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.module.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best = float(state["best"])
        start_epoch = int(state["epoch"]) + 1
        print(f"resumed epoch={start_epoch - 1} best={best:.6f}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); loss_sum = 0.
        for waveform, target in tqdm(train_loader, desc=f"AudioSet {epoch}/{args.epochs}"):
            x = fbank(waveform, precomputed=args.worker_fbank).to(device, non_blocking=True)
            x = apply_specaugment(x)
            target = target.to(device, non_blocking=True)
            x, target = mix(x, target)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), target)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer); scaler.update(); ema.update(model); loss_sum += loss.item()
        scheduler.step()
        selection_loader = val_loader if val_loader is not None else test_loader
        metric = evaluate(ema.module, selection_loader, device)
        metric_name = "val_mAP" if val_loader is not None else "test_mAP"
        print(f"epoch={epoch}/{args.epochs} loss={loss_sum/len(train_loader):.5f} {metric_name}={metric:.6f}", flush=True)
        if metric > best:
            best = metric; torch.save(ema.module.state_dict(), os.path.join(args.output_dir, "best.pth"))
        if args.save_every and epoch % args.save_every == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(), "ema": ema.module.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best": best},
                       os.path.join(args.output_dir, "last.pth"))
    if val_loader is not None:
        ema.module.load_state_dict(torch.load(os.path.join(args.output_dir, "best.pth"), map_location=device, weights_only=True))
        test_metric = evaluate(ema.module, test_loader, device)
        print(f"finished best_val_mAP={best:.6f} test_mAP={test_metric:.6f}", flush=True)
    else:
        test_metric = best
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as handle:
        handle.write(f"best_{'val' if val_loader is not None else 'test'}_mAP={best:.6f}\n")
        handle.write(f"test_mAP={test_metric:.6f}\n")


if __name__ == "__main__": main()
