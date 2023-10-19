#!/usr/bin/env python3
import numpy as np
import audiomentations as AA

from audiomentations import Compose
from typing import List

AUG_PARAMS = {
    # See a list of possible transforms here: https://iver56.github.io/audiomentations/
    # "p" is the probability of applying the transform
    "AddGaussianNoise": {"min_amplitude": 0.001, "max_amplitude": 0.015, "p": 1.0},
    "AddGaussianSNR": {"min_snr_db": 5.0, "max_snr_db": 40.0, "p": 0.5},
    "ClippingDistortion": {
        "min_percentile_threshold": 0,
        "max_percentile_threshold": 40,
        "p": 0.5,
    },
    "Gain": {
        "min_gain_db": -12.0,
        "max_gain_db": 12.0,
        "p": 0.5,
    },
    "GainTransition": {
        "min_gain_db": -24.0,
        "max_gain_db": 10.0,
        "min_duration": 0.25,
        "max_duration": 0.25,
        "duration_unit": "fraction",
        "p": 0.5,
    },
    "Normalize": {"p": 0.5},
    "TimeStretch": {"min_rate": 0.8, "max_rate": 1.25, "p": 0.5},
    "PitchShift": {"min_semitones": -0.5, "max_semitones": 0.5, "p": 0.5},
    "Shift": {"min_shift": -0.5, "max_shift": 0.5, "p": 0.5},
}


def apply_augmentation(
    samples: np.ndarray, sample_rate: float, augmentations: List[str]
) -> np.ndarray:
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

    for transform in augment.transforms:
        print(f"{transform.__class__.__name__}: {transform.parameters}")

    return augmented_samples
