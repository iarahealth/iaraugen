#!/usr/bin/env python3
import audiomentations as AA
import numpy as np

# import os

from audiomentations import Compose
from typing import List, Tuple

# from multiprocessing import Pool

AUG_PARAMS = {
    # See a list of possible transforms here: https://iver56.github.io/audiomentations/
    # "p" is the probability of applying the transform
    "AddGaussianNoise": {"min_amplitude": 0.001, "max_amplitude": 0.015, "p": 0.8},
    "AddGaussianSNR": {"min_snr_db": 5.0, "max_snr_db": 40.0, "p": 0.8},
    "ClippingDistortion": {
        "min_percentile_threshold": 0,
        "max_percentile_threshold": 40,
        "p": 0.8,
    },
    "Gain": {
        "min_gain_db": -12.0,
        "max_gain_db": 12.0,
        "p": 0.8,
    },
    "GainTransition": {
        "min_gain_db": -24.0,
        "max_gain_db": 10.0,
        "min_duration": 0.25,
        "max_duration": 0.25,
        "duration_unit": "fraction",
        "p": 0.8,
    },
    "Normalize": {"p": 0.8},
    "TimeStretch": {"min_rate": 0.8, "max_rate": 1.25, "p": 0.8},
    "PitchShift": {"min_semitones": -0.5, "max_semitones": 0.5, "p": 0.8},
    "Shift": {"min_shift": -0.5, "max_shift": 0.5, "p": 0.8},
}


def apply_augmentation(
    samples: np.ndarray, sample_rate: float, augmentations: List[str]
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


"""
def apply_augmentation_batch(
    samples: List[np.ndarray], sample_rates: List[float], augmentations: List[str]
):
    if isinstance(samples, list) and len(samples) > 1:
        with Pool(processes=os.cpu_count()) as pool:
            augmented_samples = pool.starmap(
                apply_augmentation,
                [
                    (sample, augmentations, sr)
                    for sample, sr in zip(samples, sample_rates)
                ],
            )
        return augmented_samples
    else:
        return apply_augmentation(samples[0], sample_rates[0], augmentations)
"""
