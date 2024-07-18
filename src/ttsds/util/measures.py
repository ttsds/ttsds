from abc import ABC, abstractmethod
import os

import numpy as np
import pyworld as pw
import librosa
import torch


class Measure(ABC):
    def __init__(self, name, description, is_binary=False):
        self.name = name
        self.description = description
        self.is_binary = is_binary
        if self.is_binary:
            self.name += "_binary"

    @staticmethod
    def interpolate(x):
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        nans, y = nan_helper(x)
        x[nans] = np.interp(y(nans), y(~nans), x[~nans])
        return x

    @abstractmethod
    def compute(self, audio, durations, silence_mask=None):
        pass

    def __call__(self, audio, durations, silence_mask=None, include_prior=False):
        measure = self.compute(audio, durations, silence_mask)
        if include_prior:
            return {
                "measure": Measure.interpolate(measure),
                "prior": {
                    "mean": measure[~np.isnan(measure)].mean(),
                    "std": measure[~np.isnan(measure)].std(),
                },
            }
        else:
            return {"measure": Measure.interpolate(measure)}

    def __eq__(self, other):
        return self.name == other.name


class PitchMeasure(Measure):
    def __init__(
        self,
        name="pitch",
        description="Pitch measure",
        sampling_rate=22050,
        hop_length=256,
        pitch_quality=1,
    ):
        global pw
        super().__init__(name, description)
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.dio_speed = int(np.round(1 / pitch_quality))

    def compute(self, audio, durations, silence_mask=None):
        f0, t = pw.dio(
            audio.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
            speed=self.dio_speed,
        )
        pitch = pw.stonemask(
            audio.astype(np.float64), f0, t, self.sampling_rate
        ).astype(np.float32)
        if sum(durations) < len(pitch):
            pitch = pitch[: sum(durations)]
        if silence_mask is not None:
            pitch[silence_mask] = np.nan
        pitch[pitch < 1e-6] = np.nan
        if np.isnan(pitch).all():
            pitch[:] = 1e-6
        return pitch


class VoiceActivityMeasure(Measure):
    def __init__(
        self,
        name="voice_activity",
        description="Voice activity measure",
        sampling_rate=22050,
        win_length=1024,
        hop_length=256,
        pitch_quality=1,
    ):
        global pw
        super().__init__(name, description, is_binary=True)
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.dio_speed = int(np.round(1 / pitch_quality))

    def compute(self, audio, durations, silence_mask=None):
        f0, t = pw.dio(
            audio.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
            speed=self.dio_speed,
        )
        pitch = pw.stonemask(
            audio.astype(np.float64), f0, t, self.sampling_rate
        ).astype(np.float32)
        if sum(durations) < len(pitch):
            pitch = pitch[: sum(durations)]
        if silence_mask is not None:
            pitch[silence_mask] = np.nan
        pitch[pitch < 1e-6] = np.nan
        pitch[~np.isnan(pitch)] = 1
        pitch[np.isnan(pitch)] = 0
        return pitch


class EnergyMeasure(Measure):
    def __init__(
        self,
        name="energy",
        description="Energy measure",
        win_length=1024,
        hop_length=256,
    ):
        super().__init__(name, description)
        self.win_length = win_length
        self.hop_length = hop_length

    def compute(self, audio, durations, silence_mask=None):
        energy_overall = np.sum(audio ** 2) / len(audio)
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length,
            center=True,
        ).astype(np.float32)
        energy = energy.reshape(-1)
        if sum(durations) < len(energy):
            energy = energy[: sum(durations)]
        return energy
