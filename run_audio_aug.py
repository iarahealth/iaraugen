#!/usr/bin/env python3

import argparse
import os
import random
import numpy as np

from multiprocessing import Pool, cpu_count
from typing import List, Optional

from audio.aug import apply_augmentation, clean_audio, AUG_PARAMS
from utils.files import load_audio, save_audio


def process_audio(
    input_file: str,
    augmentations: List[str],
    output_format: str,
    sample_rate: Optional[int] = None,
    clean: bool = False,
):
    audio, sr = load_audio(input_file, sample_rate)
    output_filename = os.path.splitext(os.path.basename(input_file))[0]
    augmented_audio, transforms_used = apply_augmentation(audio, sr, augmentations)

    if clean:
        augmented_audio = clean_audio(augmented_audio, sr)

    if transforms_used or clean:
        if len(transforms_used):
            output_filename = output_filename + "-aug_" + "_".join(transforms_used)
        if clean:
            output_filename = output_filename + "_clean"
        input_dir = os.path.dirname(input_file)
        output_path = os.path.join(input_dir, f"{output_filename}.{output_format}")
        save_audio(augmented_audio, output_path, sr, output_format)
        print(f"Augmented audio saved to {output_path}")
    else:
        print(f"! No augmentations or cleaning applied to {input_file}")


if __name__ == "__main__":
    augmentation_choices = list(AUG_PARAMS.keys())
    parser = argparse.ArgumentParser(
        description="Audio augmentation using audiomentations"
    )
    parser.add_argument("input_file", type=str, nargs="+", help="Input audio files")
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=None,  # Usual values include 16000, 22050, 44100
        help="Sample rate to load files (default: None, uses native sample rate)",
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
    parser.add_argument(
        "--clean", action="store_true", help="Clean audio before saving it"
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    input_args = [
        (
            input_file,
            args.augmentations,
            args.output_format,
            args.sample_rate,
            args.clean,
        )
        for input_file in args.input_file
    ]

    pool = Pool(processes=cpu_count())
    pool.starmap(process_audio, input_args)
    pool.close()
    pool.join()
