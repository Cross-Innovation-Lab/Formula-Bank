"""Deterministic on-the-fly synthetic data for SSAST and PaSST."""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

import pretrain as synth
from official_ssl.runtime import isolated_rng


PRIMITIVES = (
    "harmonic",
    "inharmonic",
    "modal",
    "fm",
    "chirp",
    "pulse",
    "noise_texture",
)


def load_synth_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return synth._merge_synth_config(json.load(handle))


@contextmanager
def using_synth_config(config):
    """Temporarily set the global legacy synthesizer config for one sample."""
    old_config = synth.SYNTH_CONFIG
    old_hash = synth.SYNTH_CONFIG_HASH
    synth.set_synth_config(copy.deepcopy(config), cfg_hash="official_ssl_stream")
    try:
        yield
    finally:
        synth.SYNTH_CONFIG = old_config
        synth.SYNTH_CONFIG_HASH = old_hash


def sample_f0(config):
    f0_min, f0_max = map(float, config.get("f0_range_hz", [50.0, 2000.0]))
    if not 0.0 < f0_min < f0_max:
        raise ValueError(f"Invalid f0_range_hz: {[f0_min, f0_max]}")
    distribution = str(config.get("f0_distribution", "uniform")).replace("-", "_")
    if distribution == "uniform":
        return float(np.random.uniform(f0_min, f0_max))
    if distribution == "log_uniform":
        return float(np.exp(np.random.uniform(np.log(f0_min), np.log(f0_max))))
    raise ValueError(f"Unsupported f0_distribution: {distribution}")


def waveform_to_ssast_fbank(waveform):
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )
    if fbank.shape[0] < 1024:
        fbank = F.pad(fbank, (0, 0, 0, 1024 - fbank.shape[0]))
    else:
        fbank = fbank[:1024]
    return (fbank - (-4.26)) / (4.56 * 2)


class SyntheticStream(Dataset):
    """Finite deterministic stream; all content is generated in workers on demand."""

    def __init__(self, config_path, mode, epoch_len, seed, validation=False):
        if mode not in {"ssast", "passt"}:
            raise ValueError(f"Unsupported stream mode: {mode}")
        self.config = load_synth_config(config_path)
        self.mode = mode
        self.epoch_len = int(epoch_len)
        self.seed = int(seed) + (1_000_000 if validation else 0)
        self.validation = bool(validation)
        self._epoch = mp.Value("i", 0)

    def __len__(self):
        return self.epoch_len

    def set_epoch(self, epoch):
        self._epoch.value = int(epoch)

    def _seed_for(self, index):
        return self.seed + self._epoch.value * 10_000_019 + int(index)

    def __getitem__(self, index):
        with isolated_rng(self._seed_for(index)), using_synth_config(self.config):
            if self.mode == "ssast":
                waveform = synth.advanced_physics_synth(
                    sample_f0(self.config), duration=10.24, sample_rate=16000
                )
                return waveform_to_ssast_fbank(waveform), torch.tensor(0, dtype=torch.long)

            primitive_count = random.randint(1, 3)
            selected = random.sample(PRIMITIVES, primitive_count)
            waveform, metadata = synth.advanced_physics_synth(
                sample_f0(self.config),
                duration=10.0,
                sample_rate=32000,
                return_metadata=True,
                event_counts_override=[1] * primitive_count,
                primitive_sequence=selected,
            )
            rendered = {event["primitive"] for event in metadata["events"]}
            if rendered != set(selected):
                raise RuntimeError("Sparse PaSST primitive labels did not match rendered events")
            target = torch.zeros(len(PRIMITIVES), dtype=torch.float32)
            for primitive in selected:
                target[PRIMITIVES.index(primitive)] = 1.0
            return waveform, target
