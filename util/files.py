#!/usr/bin/env python3
import re

from typing import List, Optional
from .text import replacement_dict, add_period_and_capitalize, pre_process_sentences


def read_sentences_corpus(
    filename: str, max_sentences: Optional[str] = None
) -> List[str]:
    """
    Reads sentences from a corpus file, and returns a list of sentences.

    Args:
        filename (str): The name of the input file.
        max_sentences (str): Optional. The maximum number of sentences to read.
                             Can be % or number.

    Returns:
        List[str]: A list of sentences.

    Note:
        To make the augmentation useful, replace punctuation words with
        punctuation marks, capitalize the sentences and add a dot at the end.
        The sentences will be post-processed to change the punctuation marks
        back to words again, unless stated otherwise.
    """
    sentences = []
    with open(filename, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
        f.seek(0)
        if max_sentences not in [None, -1, "100%"]:
            max_sentences = (
                int(max_sentences)
                if max_sentences[-1] != "%"
                else round(line_count * (float(max_sentences[:-1]) / 100.0))
            )
        for line in f:
            sentences.append(line.strip())
            if max_sentences not in [None, -1, "100%"]:
                if len(sentences) == max_sentences:
                    break

    sentences = pre_process_sentences(sentences)
    return sentences


def append_sentences_to_file(filename: str, sentences: List[str]):
    """
    Appends sentences to a file.

    Args:
        filename (str): The name of the output file.
        sentences (List[str]): The list of sentences to be written to the file.
    """
    with open(filename, "a", encoding="utf-8") as outfile:
        for sentence in sentences:
            outfile.write("\n" + sentence)


def read_file(filename: str) -> List[str]:
    with open(filename, "r", encoding="utf-8") as f:
        return f.readlines()
