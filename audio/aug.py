#!/usr/bin/env python3
import numpy as np
import audiomentations as AA

from audiomentations import Compose
from typing import List

AUG_PARAMS = {
    "AddGaussianNoise": {"min_amplitude": 0.001, "max_amplitude": 0.015, "p": 1.0},
    "TimeStretch": {"min_rate": 0.8, "max_rate": 1.25, "p": 0.5},
    "PitchShift": {"min_semitones": -4, "max_semitones": 4, "p": 0.5},
    "Shift": {"min_fraction": -0.5, "max_fraction": 0.5, "p": 0.5},
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

    print("Augmentations: ", augmentation_list)
    augment = Compose(augmentation_list)

    # for transform in augment.transforms:
    #    print(f"{transform.__class__.__name__}: {transform.parameters}")

    augmented_samples = augment(samples=samples, sample_rate=sample_rate)

    return augmented_samples
