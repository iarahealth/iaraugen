#!/usr/bin/env python3

import argparse
import os
import random
import numpy as np

from audio.aug import apply_augmentation
from utils.files import load_audio, save_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio augmentation using audiomentations"
    )
    parser.add_argument("input_file", type=str, help="Path to the input audio file")
    parser.add_argument(
        "--output_file",
        type=str,
        default="output",
        help="Path to the output audio file (default: 'output')",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="ogg",
        help="Output format (default: 'ogg')",
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        default="",
        nargs="+",
        help="Audiomentation techniques (e.g., AddGaussianNoise, PitchShift, TimeStretch)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible augmentations (default: None, generates a random seed)",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    audio, sr = load_audio(args.input_file)

    augmented_audio = apply_augmentation(audio, sr, args.augmentations)
    save_audio(augmented_audio, args.output_file, sr, args.output_format)

    print(f"Augmented audio saved to {args.output_file}")
