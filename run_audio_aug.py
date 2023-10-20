#!/usr/bin/env python3

import argparse
import os
import random
import numpy as np

from multiprocessing import Pool, cpu_count
from audio.aug import apply_augmentation, AUG_PARAMS
from utils.files import load_audio, save_audio
from typing import List


def process_audio(input_file: str, augmentations: List[str], output_format: str):
    audio, sr = load_audio(input_file)
    output_filename = os.path.splitext(os.path.basename(input_file))[0]
    augmented_audio, transforms_used = apply_augmentation(audio, sr, augmentations)

    if len(transforms_used) > 0:
        output_filename = output_filename + "_" + "_".join(transforms_used)
        save_audio(augmented_audio, output_filename, sr, output_format)
        print(f"Augmented audio saved to {output_filename}.{output_format}")


if __name__ == "__main__":
    augmentation_choices = list(AUG_PARAMS.keys())
    parser = argparse.ArgumentParser(
        description="Audio augmentation using audiomentations"
    )
    parser.add_argument("input_file", type=str, nargs="+", help="Input audio files")
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
        choices=augmentation_choices,
        nargs="+",
        help="Audiomentation techniques (e.g., AddGaussianNoise PitchShift TimeStretch)",
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

    input_args = [
        (input_file, args.augmentations, args.output_format)
        for input_file in args.input_file
    ]

    pool = Pool(processes=cpu_count())
    pool.starmap(process_audio, input_args)
    pool.close()
    pool.join()
