#!/usr/bin/env python3
import audiomentations as AA
import numpy as np
import librosa

from audiomentations import Compose
from noisereduce import reduce_noise
from typing import List, Tuple


AUG_PARAMS = {
    # See a list of possible transforms here: https://iver56.github.io/audiomentations/
    # "p" is the probability of applying the transform
    "AddGaussianNoise": {"min_amplitude": 0.0001, "max_amplitude": 0.005, "p": 0.7},
    "AddGaussianSNR": {"min_snr_db": 5.0, "max_snr_db": 40.0, "p": 0.7},
    "ClippingDistortion": {
        "min_percentile_threshold": 0,
        "max_percentile_threshold": 40,
        "p": 0.7,
    },
    "Gain": {
        "min_gain_db": -12.0,
        "max_gain_db": 12.0,
        "p": 0.7,
    },
    "GainTransition": {
        "min_gain_db": -24.0,
        "max_gain_db": 10.0,
        "min_duration": 0.25,
        "max_duration": 0.25,
        "duration_unit": "fraction",
        "p": 0.7,
    },
    "Mp3Compression": {
        "min_bitrate": 8,
        "max_bitrate": 64,
        "backend": "pydub",
        "p": 0.5,
    },
    "Normalize": {"p": 0.7},
    "TimeStretch": {"min_rate": 0.8, "max_rate": 1.25, "p": 0.7},
    "PitchShift": {"min_semitones": -1.0, "max_semitones": 1.0, "p": 0.7},
    "Shift": {"min_shift": -1.0, "max_shift": 1.0, "p": 0.7},
}


def clean_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    samples = reduce_noise(y=samples, sr=sample_rate)
    samples = librosa.util.normalize(samples)  # Peak normalization
    return samples


def resample(samples: np.ndarray, sample_rate: int, new_sample_rate: int) -> np.ndarray:
    return librosa.resample(samples, orig_sr=sample_rate, target_sr=new_sample_rate)


def apply_augmentation(
    samples: np.ndarray, sample_rate: int, augmentations: List[str]
) -> Tuple[np.ndarray, List[str]]:
    augmentation_list = []

    for augmentation_name in augmentations:
        if augmentation_name in AUG_PARAMS:
            params = AUG_PARAMS[augmentation_name]
            if hasattr(AA, augmentation_name):
                augmentation_list.append(getattr(AA, augmentation_name)(**params))
            else:
                print(f"Invalid augmentation technique: {augmentation_name}")
                exit(1)
        else:
            print(f"Invalid augmentation technique: {augmentation_name}")
            exit(1)

    augment = Compose(augmentation_list)
    augmented_samples = augment(samples=samples, sample_rate=sample_rate)

    transforms_used = []
    for transform in augment.transforms:
        # print(f"{transform.__class__.__name__}: {transform.parameters}")
        if transform.parameters["should_apply"]:
            transforms_used.append(transform.__class__.__name__)

    return augmented_samples, transforms_used
