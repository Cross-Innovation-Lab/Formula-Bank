#!/usr/bin/env python
"""Extract deterministic held-out logits and confidence entropy from full-FT models."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("esc50", "us8k", "fsd50k"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mask-ratio", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def task_components(task):
    if task == "esc50":
        from esc_refer import AudioMAEClassifier, ESC50Dataset, get_audiomae_config
        return (
            AudioMAEClassifier(num_classes=50, stride=16, **get_audiomae_config("small")),
            ESC50Dataset("datasets/ESC-50/ESC-50-master", fold=1, train=False),
            "single_label",
        )
    if task == "us8k":
        from us8k_refer import AudioMAEClassifier, UrbanSoundDataset, get_audiomae_config
        return (
            AudioMAEClassifier(num_classes=10, stride=16, **get_audiomae_config("small")),
            UrbanSoundDataset("datasets/UrbanSound8K", test_fold=10, train=False),
            "single_label",
        )
    from fsd50k_refer import AudioMAEClassifier, FSD50KDataset, get_audiomae_config
    dataset = FSD50KDataset("datasets/FSD50K", split="val")
    return (
        AudioMAEClassifier(num_classes=len(dataset.label2id), stride=16, **get_audiomae_config("small")),
        dataset,
        "multi_label",
    )


def load_checkpoint(model, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("model_state") or payload.get("model") or payload.get("state_dict") or payload
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    result = model.load_state_dict(cleaned, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint mismatch: {result}")


def softmax_entropy(probabilities):
    return -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)


def binary_entropy(probabilities):
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return -(probabilities * np.log(probabilities) + (1 - probabilities) * np.log(1 - probabilities))


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset, kind = task_components(args.task)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    logits_rows, target_rows = [], []
    with torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(loader):
            logits = model(inputs.to(device, non_blocking=True)).float().cpu().numpy()
            logits_rows.append(logits)
            target_rows.append(targets.numpy())
            if (batch_index + 1) % 20 == 0:
                print(f"{args.task} M{args.mask_ratio:.2f}: {batch_index + 1}/{len(loader)} batches", flush=True)
    logits = np.concatenate(logits_rows)
    targets = np.concatenate(target_rows)
    if kind == "single_label":
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        prediction = probabilities.argmax(axis=1)
        correct = prediction == targets
        entropy = softmax_entropy(probabilities)
        sorted_probabilities = np.sort(probabilities, axis=1)
        margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        nll = -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0))
        summary = {
            "task": args.task, "mask_ratio": args.mask_ratio, "samples": len(targets), "classes": logits.shape[1],
            "accuracy": float(correct.mean()), "mean_entropy_nats": float(entropy.mean()),
            "normalized_mean_entropy": float(entropy.mean() / math.log(logits.shape[1])),
            "mean_entropy_correct": float(entropy[correct].mean()),
            "mean_entropy_incorrect": float(entropy[~correct].mean()),
            "mean_top1_top2_probability_margin": float(margin.mean()), "mean_nll": float(nll.mean()),
        }
        payload = dict(logits=logits.astype(np.float32), targets=targets.astype(np.int16), probabilities=probabilities.astype(np.float32), correct=correct, entropy_nats=entropy.astype(np.float32), margin=margin.astype(np.float32))
    else:
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        entropy = binary_entropy(probabilities)
        positive = targets.astype(bool)
        negative = ~positive
        bce = -(targets * np.log(np.clip(probabilities, 1e-12, 1.0)) + (1 - targets) * np.log(np.clip(1 - probabilities, 1e-12, 1.0))).mean(axis=1)
        summary = {
            "task": args.task, "mask_ratio": args.mask_ratio, "samples": len(targets), "classes": logits.shape[1],
            "mean_binary_entropy_all_labels_nats": float(entropy.mean()),
            "normalized_mean_binary_entropy_all_labels": float(entropy.mean() / math.log(2.0)),
            "mean_binary_entropy_ground_truth_positive_labels": float(entropy[positive].mean()),
            "mean_binary_entropy_ground_truth_negative_labels": float(entropy[negative].mean()),
            "mean_binary_cross_entropy": float(bce.mean()),
        }
        payload = dict(logits=logits.astype(np.float32), targets=targets.astype(np.uint8), probabilities=probabilities.astype(np.float32), entropy_per_label_nats=entropy.astype(np.float32), bce_per_sample=bce.astype(np.float32))
    np.savez_compressed(args.output_dir / "heldout_logits.npz", **payload)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
