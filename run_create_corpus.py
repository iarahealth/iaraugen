#!/usr/bin/env python3
import argparse

from utils.files import read_file
from text.dataset import create_corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets to create a corpus")
    parser.add_argument("files", nargs="+", help="List of text files to read")
    parser.add_argument(
        "--lang",
        type=str,
        choices=["pt", "en", "es"],
        required="True",
        help="Language of the text files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="corpus.tok",
        help="Name of the output file",
    )

    args = parser.parse_args()

    sentences = []
    print(f"Reading {len(args.files)} files")
    for f in args.files:
        sentences.extend(read_file(f))

    print(f"Cleaning {len(sentences)} sentences")
    cleaned_sentences = create_corpus(sentences, args.lang)

    with open(args.output, "w") as f:
        print(f"Writing {len(cleaned_sentences)} cleaned sentences to {args.output}")
        for sentence in cleaned_sentences:
            f.write(sentence + "\n")
