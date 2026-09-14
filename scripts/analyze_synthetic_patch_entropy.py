#!/usr/bin/env python
"""Audit discrete patch entropy in the frozen C224 synthetic Formula Bank.

The codebook is fit only on the synthetic audit split.  Consequently these
numbers characterize this corpus; a later Synthetic-vs-AudioSet comparison
must fit one shared codebook rather than compare this run's absolute values.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import pretrain_sl as core


def entropy_from_counts(counts):
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def patchify(fbank):
    # [1, 1024, 128] -> [512, 256], matching the non-overlapping MAE patches.
    array = fbank.squeeze(0).numpy()
    return array.reshape(64, 16, 8, 16).transpose(0, 2, 1, 3).reshape(512, 256)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formula-bank", type=Path, required=True)
    parser.add_argument("--subset-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-seed", type=int, default=2026)
    parser.add_argument("--instances-per-class", type=int, default=50)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--patches-per-clip-for-fit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    bank = core.FormulaBank(str(args.formula_bank), "C224", str(args.subset_registry))
    dataset = core.FormulaExposureDataset(
        bank=bank, split="val", seed=args.pool_seed,
        instances_per_class=args.instances_per_class,
    )
    sampled = []
    for index in range(len(dataset)):
        fbank, _label, _rejected = dataset[index]
        patches = patchify(fbank)
        take = min(args.patches_per_clip_for_fit, len(patches))
        sampled.append(patches[rng.choice(len(patches), size=take, replace=False)])
        if (index + 1) % 500 == 0:
            print(f"codebook samples: {index + 1}/{len(dataset)}", flush=True)
    fit_patches = np.concatenate(sampled, axis=0).astype(np.float32)
    kmeans = MiniBatchKMeans(
        n_clusters=args.codebook_size, batch_size=2048, n_init=1,
        random_state=args.seed, reassignment_ratio=0.0,
    ).fit(fit_patches)
    np.save(args.output_dir / "codebook_centers.npy", kmeans.cluster_centers_)

    global_counts = np.zeros(args.codebook_size, dtype=np.int64)
    active_counts = np.zeros(args.codebook_size, dtype=np.int64)
    onset_counts = np.zeros(args.codebook_size, dtype=np.int64)
    temporal_pairs = np.zeros((args.codebook_size, args.codebook_size), dtype=np.int64)
    clip_entropies = []
    codes_out = np.empty((len(dataset), 64, 8), dtype=np.uint16)
    for index in range(len(dataset)):
        fbank, _label, _rejected = dataset[index]
        patches = patchify(fbank).astype(np.float32)
        codes = kmeans.predict(patches).reshape(64, 8)
        codes_out[index] = codes
        counts = np.bincount(codes.ravel(), minlength=args.codebook_size)
        global_counts += counts
        clip_entropies.append(entropy_from_counts(counts))
        energy = patches.mean(axis=1).reshape(64, 8)
        active = energy >= np.quantile(energy, 0.50)
        onset = np.maximum(np.diff(energy, axis=0, prepend=energy[:1]), 0.0)
        onset_selected = onset >= np.quantile(onset, 0.90)
        active_counts += np.bincount(codes[active], minlength=args.codebook_size)
        onset_counts += np.bincount(codes[onset_selected], minlength=args.codebook_size)
        for frequency in range(8):
            np.add.at(temporal_pairs, (codes[:-1, frequency], codes[1:, frequency]), 1)
        if (index + 1) % 500 == 0:
            print(f"encoding audit set: {index + 1}/{len(dataset)}", flush=True)
    np.savez_compressed(args.output_dir / "audit_patch_codes.npz", codes=codes_out)
    conditional = 0.0
    previous_counts = temporal_pairs.sum(axis=1)
    for code, count in enumerate(previous_counts):
        if count:
            conditional += count * entropy_from_counts(temporal_pairs[code])
    conditional /= temporal_pairs.sum()
    summary = {
        "analysis": "synthetic_patch_code_entropy_v1",
        "corpus": "frozen FormulaBank C224 validation split",
        "pool_seed": args.pool_seed,
        "instances_per_class": args.instances_per_class,
        "clips": len(dataset),
        "patch_grid": [64, 8],
        "codebook": {"algorithm": "MiniBatchKMeans", "size": args.codebook_size, "fit_patches": len(fit_patches), "seed": args.seed},
        "entropy_bits": {
            "marginal_patch_code": entropy_from_counts(global_counts),
            "mean_per_clip_patch_code": float(np.mean(clip_entropies)),
            "active_patch_code": entropy_from_counts(active_counts),
            "top_decile_positive_temporal_change_patch_code": entropy_from_counts(onset_counts),
            "next_time_patch_code_given_previous_same_frequency": float(conditional),
        },
        "interpretation": "This is a within-synthetic descriptive audit, not evidence that synthetic data is simpler than AudioSet. That comparison requires a shared pooled codebook and matched preprocessing.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
