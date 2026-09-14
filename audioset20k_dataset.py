"""AudioSet-20K dataset and feature-loading utilities.

AudioSet is kept as the downloaded WebDataset tar shards: no labels from the
pre-training protocol are used, and each downstream target comes only from the
JSON paired with the audio clip in the official 20K train/test shards.
"""
from __future__ import annotations

import io
import json
import os
import random
import tarfile
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


def waveform_to_fbank(waveform, sample_rate=16000, frames=1024):
    """Convert a waveform to the AudioMAE-compatible log-Mel input.

    This uses the same Kaldi fbank configuration as the downstream evaluation
    scripts. Moving this deterministic conversion to workers removes a serial
    CPU stage without changing cropping, normalization, or augmentation.
    """
    value = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0), htk_compat=True, sample_frequency=sample_rate,
        use_energy=False, window_type="hanning", num_mel_bins=128,
        dither=0.0, frame_shift=10,
    )
    return value[:frames] if value.shape[0] >= frames else F.pad(value, (0, 0, 0, frames - value.shape[0]))


class AudioSet20KTarDataset(Dataset):
    """527-way multi-label AudioSet-20K reader without extraction."""

    def __init__(self, root, train, sample_rate=16000, duration=10.24, augment=False,
                 precompute_fbank=False):
        self.root = Path(root) / "20k" / ("train" if train else "test")
        self.train, self.augment = bool(train), bool(augment)
        self.sample_rate = int(sample_rate)
        self.target_samples = int(sample_rate * duration)
        self.precompute_fbank = bool(precompute_fbank)
        self.records = []
        for archive in sorted(self.root.glob("*.tar")):
            with tarfile.open(archive) as handle:
                names = {member.name for member in handle if member.isfile()}
            for wav in sorted(name for name in names if name.endswith(".wav")):
                annotation = wav[:-4] + ".json"
                if annotation in names:
                    self.records.append((str(archive), wav, annotation))
        if not self.records:
            raise FileNotFoundError(f"No paired AudioSet wav/json records in {self.root}")
        self._handles = {}

    def __len__(self):
        return len(self.records)

    def _archive(self, path):
        # A Dataset instance belongs to one worker process; each process owns
        # its handles and closes them on process exit.
        handle = self._handles.get(path)
        if handle is None:
            handle = tarfile.open(path)
            self._handles[path] = handle
        return handle

    def _fix_length(self, waveform):
        if waveform.shape[-1] == 0:
            waveform = torch.zeros(1, self.target_samples)
        if waveform.shape[-1] < self.target_samples:
            waveform = waveform.repeat(1, self.target_samples // waveform.shape[-1] + 1)
        max_start = waveform.shape[-1] - self.target_samples
        if self.train and max_start > 0:
            start = random.randint(0, max_start)
        else:
            start = max(0, max_start // 2)
        return waveform[:, start:start + self.target_samples]

    def __getitem__(self, index):
        archive, wav_name, json_name = self.records[index]
        handle = self._archive(archive)
        waveform, sr = torchaudio.load(io.BytesIO(handle.extractfile(wav_name).read()))
        target_info = json.loads(handle.extractfile(json_name).read())
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(0, keepdim=True)
        waveform = self._fix_length(waveform)
        target = torch.zeros(527, dtype=torch.float32)
        for label in target_info.get("label_id", []):
            if 0 <= int(label) < 527:
                target[int(label)] = 1.0
        waveform = waveform.squeeze(0).float()
        if self.precompute_fbank:
            waveform = waveform_to_fbank(waveform, self.sample_rate)
        return waveform, target
