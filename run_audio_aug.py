#!/usr/bin/env python3

import argparse
import os
import random
import numpy as np

from audio.aug import apply_augmentation, AUG_PARAMS
from utils.files import load_audio, save_audio


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

    audios = []
    srs = []
    output_filenames = []
    for input_file in args.input_file:
        audio, sr = load_audio(input_file)
        output_filename = os.path.splitext(os.path.basename(input_file))[0]
        audios.append(audio)
        srs.append(sr)
        output_filenames.append(output_filename)

    for audio, sr, output_filename in zip(audios, srs, output_filenames):
        augmented_audio, transforms_used = apply_augmentation(
            audio, sr, args.augmentations
        )
        if len(transforms_used) > 0:
            output_filename = output_filename + "_" + "_".join(transforms_used)
            save_audio(
                augmented_audio,
                output_filename,
                sr,
                args.output_format,
            )
            print(f"Augmented audio saved to {output_filename}.{args.output_format}")
