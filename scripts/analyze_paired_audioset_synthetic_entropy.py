#!/usr/bin/env python
"""Compare synthetic and AudioSet patch-code entropies with one shared codebook."""

import argparse
import csv
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.cluster import MiniBatchKMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import pretrain_sl as core


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formula-bank", type=Path, required=True)
    parser.add_argument("--subset-registry", type=Path, required=True)
    parser.add_argument("--audioset-root", type=Path, required=True)
    parser.add_argument("--audioset-train", type=Path, required=True)
    parser.add_argument("--audioset-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fit-clips-per-domain", type=int, default=2000)
    parser.add_argument("--eval-clips-per-domain", type=int, default=2000)
    parser.add_argument("--patches-per-clip-for-fit", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=256)
    return parser.parse_args()


def patchify(fbank):
    array = fbank.squeeze(0).numpy()
    return array.reshape(64, 16, 8, 16).transpose(0, 2, 1, 3).reshape(512, 256)


def entropy(counts):
    values = counts[counts > 0].astype(np.float64)
    values /= values.sum()
    return float(-(values * np.log2(values)).sum())


def fbank_from_wav_bytes(payload):
    waveform, sample_rate = torchaudio.load(io.BytesIO(payload))
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    target_samples = int(16000 * 10.24)
    if waveform.shape[1] < target_samples:
        waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_samples]
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
        window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10,
    )
    if fbank.shape[0] < 1024:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, 1024 - fbank.shape[0]))
    else:
        fbank = fbank[:1024]
    return ((fbank - (-4.26)) / (4.56 * 2)).unsqueeze(0)


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audioset_fbanks(root, rows):
    by_tar = {}
    for row in rows:
        by_tar.setdefault(row["tar_path"], []).append(row)
    result = []
    for relative_tar, group in by_tar.items():
        with tarfile.open(root / relative_tar, "r") as archive:
            for row in group:
                handle = archive.extractfile(row["wav_member"])
                if handle is None:
                    raise FileNotFoundError(f"{relative_tar}:{row['wav_member']}")
                result.append(fbank_from_wav_bytes(handle.read()))
    return result


def summarize(name, fbanks, kmeans):
    size = kmeans.n_clusters
    all_counts = np.zeros(size, dtype=np.int64)
    active_counts = np.zeros(size, dtype=np.int64)
    transient_counts = np.zeros(size, dtype=np.int64)
    pairs = np.zeros((size, size), dtype=np.int64)
    per_clip = []
    for fbank in fbanks:
        patches = patchify(fbank).astype(np.float32)
        codes = kmeans.predict(patches).reshape(64, 8)
        counts = np.bincount(codes.ravel(), minlength=size)
        all_counts += counts
        per_clip.append(entropy(counts))
        energies = patches.mean(axis=1).reshape(64, 8)
        active = energies >= np.quantile(energies, 0.50)
        changes = np.maximum(np.diff(energies, axis=0, prepend=energies[:1]), 0.0)
        transient = changes >= np.quantile(changes, 0.90)
        active_counts += np.bincount(codes[active], minlength=size)
        transient_counts += np.bincount(codes[transient], minlength=size)
        for frequency in range(8):
            np.add.at(pairs, (codes[:-1, frequency], codes[1:, frequency]), 1)
    conditional = sum(count * entropy(pairs[code]) for code, count in enumerate(pairs.sum(axis=1)) if count) / pairs.sum()
    return {
        "domain": name,
        "clips": len(fbanks),
        "marginal_patch_code_bits": entropy(all_counts),
        "mean_per_clip_patch_code_bits": float(np.mean(per_clip)),
        "active_patch_code_bits": entropy(active_counts),
        "transient_patch_code_bits": entropy(transient_counts),
        "next_time_given_previous_same_frequency_bits": float(conditional),
    }


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    bank = core.FormulaBank(str(args.formula_bank), "C224", str(args.subset_registry))
    synthetic_dataset = core.FormulaExposureDataset(bank, "val", seed=2026, instances_per_class=50)
    synthetic_indices = rng.choice(len(synthetic_dataset), size=args.fit_clips_per_domain + args.eval_clips_per_domain, replace=False)
    synthetic_fit = [synthetic_dataset[int(index)][0] for index in synthetic_indices[:args.fit_clips_per_domain]]
    synthetic_eval = [synthetic_dataset[int(index)][0] for index in synthetic_indices[args.fit_clips_per_domain:]]
    audio_train = read_manifest(args.audioset_train)
    audio_audit = read_manifest(args.audioset_audit)
    audio_fit_rows = [audio_train[int(index)] for index in rng.choice(len(audio_train), size=args.fit_clips_per_domain, replace=False)]
    audio_eval_rows = audio_audit[:args.eval_clips_per_domain]
    audio_fit = audioset_fbanks(args.audioset_root, audio_fit_rows)
    audio_eval = audioset_fbanks(args.audioset_root, audio_eval_rows)
    fit_patches = []
    for fbank in synthetic_fit + audio_fit:
        patches = patchify(fbank)
        fit_patches.append(patches[rng.choice(len(patches), size=args.patches_per_clip_for_fit, replace=False)])
    fit_patches = np.concatenate(fit_patches).astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=args.codebook_size, batch_size=2048, n_init=1, random_state=args.seed, reassignment_ratio=0.0).fit(fit_patches)
    np.save(args.output_dir / "shared_codebook_centers.npy", kmeans.cluster_centers_)
    result = {
        "analysis": "paired_synthetic_audioset_patch_entropy_v1",
        "shared_codebook": {"size": args.codebook_size, "fit_patches": len(fit_patches), "equal_fit_clips_per_domain": args.fit_clips_per_domain, "seed": args.seed},
        "evaluation": [summarize("synthetic", synthetic_eval, kmeans), summarize("audioset_balanced_unbalanced", audio_eval, kmeans)],
        "interpretation": "A shared codebook permits cross-domain descriptive comparison. The one-step conditional entropy measures only nearest-neighbor temporal predictability, not random-mask conditional reconstruction difficulty.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
