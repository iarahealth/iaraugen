#!/usr/bin/env python3
import argparse
import random
from text.aug import create_augmentation_sequence, translate_sentences_api
from utils.text import (
    post_process_sentences,
    print_sentences_comparison,
    remove_equal_sentences,
)
from utils.files import append_sentences_to_file, read_sentences_corpus

"""
example usage:
./txt_aug.py corpus.tok --aug translate random --action delete --maxs 10 --lang en --append
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentence augmentation: back-translate, random delete, random swap, synonym replacement."
    )
    parser.add_argument("corpus", type=str, help="Input corpus file")
    parser.add_argument(
        "--aug",
        nargs="+",
        type=str,
        required=True,
        choices=[
            "backtranslate",
            "translate",
            "synonym",
            "antonym",
            "random_swap",
            "random_del",
            "embed_bert_subs",
            "embed_bert_ins",
            "embed_subs",
            "embed_ins",
        ],
        help="Augmentation type to perform",
    )
    parser.add_argument(
        "--maxs",
        type=str,
        default="10",
        help="Maximum number of sentences to process. Can be a percentage of the total, e.g., 5%% (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=451,
        help="Random seed (default: 451)",
    )
    parser.add_argument(
        "--slang",
        type=str,
        default="pt",
        help="Source language for translation (default: pt)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Target language for translation (default: en)",
    )
    parser.add_argument(
        "--translate_mode",
        type=str,
        choices=["google", "local", "openai"],
        default="local",
        help="Target language for translation (default: local)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Process on CPU or CUDA (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to write augmented sentences in addition to the input corpus",
    )
    parser.add_argument("--append", action="store_true", help="Append to corpus file")
    args = parser.parse_args()

    random.seed(args.seed)

    sentences = read_sentences_corpus(args.corpus, max_sentences=args.maxs)
    print(f"Read {len(sentences)} sentences from {args.corpus}")

    if args.aug == ["translate"]:
        translate_sentences_api(sentences, args.slang, args.lang, args.output)
        exit(0)

    augmentation_sequence = create_augmentation_sequence(
        args.aug, args.translate_mode, args.lang, args.device
    )

    augmented_sentences = sentences
    for i, aug_fn in enumerate(augmentation_sequence):
        print(f"Augmentation step {i + 1} of {len(augmentation_sequence)}:")
        augmented_sentences = aug_fn(augmented_sentences)

    print_sentences_comparison(sentences, augmented_sentences)
    augmented_sentences = post_process_sentences(augmented_sentences)
    sentences = post_process_sentences(sentences)

    print("Removing equal sentences...")
    augmented_sentences = remove_equal_sentences(sentences, augmented_sentences)

    print("\nFinal results:")
    print("-------------------")
    for sentence in augmented_sentences:
        print(sentence)
    print(f"\nTotal: {len(augmented_sentences)} sentences")
    print("-------------------\n")

    if args.append:
        print(f"Appending augmented sentences to {args.corpus}...")
        append_sentences_to_file(args.corpus, augmented_sentences)

    if args.output:
        print(f"Appending augmented sentences to {args.output}...")
        append_sentences_to_file(args.output, augmented_sentences)
