#!/usr/bin/env python3
import random
import re
from typing import List
from num2words import num2words


replacement_dict = {
    # Warning: order matters!
    "ponto de exclamação": "!",
    "exclamação": "!",
    "ponto de interrogação": "?",
    "interrogação": "?",
    "dois pontos": ":",
    "reticências": "...",
    "ponto final": ".",
    "ponto": ".",
    "vírgula": ",",
    "ponto e vírgula": ";",
    "ponto vírgula": ";",
    "travessão": "--",
    # From here on will be excluded from the final augmented sentences.
    "nova linha": "",
    "novo parágrafo": "",
    "parágrafo": "",
    "nova linha": "",
    "hífen": "",
    "abre aspas": '"',
    "abrir aspas": '"',
    "fecha aspas": '"',
    "fechar aspas": '"',
    "aspas": '"',
    "aspa": '"',
    "abre parênteses": "(",
    "abre parêntese": "(",
    "abrir parênteses": "(",
    "abrir parêntese": "(",
    "fechar parênteses": ")",
    "fechar parêntese": ")",
    "fecha parênteses": ")",
    "fecha parêntese": ")",
}

reverse_replacement_dict = {
    # Warning: order matters!
    "...": " reticências",
    ".": " ponto",
    ":": " dois pontos",
    ",": " vírgula",
    ";": " ponto e vírgula",
    "\n": " parágrafo",
    "‒-": " travessão",
    "!": " ponto de exclamação",
    "?": " ponto de interrogação",
    "(": "abre parênteses ",
    ")": " fecha parênteses",
    "-": "",
    "/": " barra ",
    '"': "",
}


def add_period_and_capitalize(sentence: str) -> str:
    """
    Adds a period at the end of the sentence if it doesn't already have one,
    capitalizes the sentences, and returns the modified sentence.

    Args:
        sentence (str): The input sentence.

    Returns:
        str: The modified sentence with added period and capitalized.
    """
    if sentence[-1] != ".":
        sentence += "."
    sentences = sentence.split(".")
    return ". ".join(s.strip().capitalize() for s in sentences).strip()


def pre_process_sentences(sentences: List[str]) -> List[str]:
    sentences_processed = []
    for s in sentences:
        s = s.strip()
        for word, punctuation in replacement_dict.items():
            s = s.replace(word, punctuation)
        if s.isspace() or s == "":
            continue
        s = s.strip()
        s = add_period_and_capitalize(s)
        # "Glue" the punctuation marks to the previous character.
        s = re.sub(r"\s+([.,;!?:)]|(\.{3,}))", r"\1", s)
        # "Glue" the ( to the next character.
        s = re.sub(r"\(\s*(\w)", r"(\1", s)
        sentences_processed.append(s)
    return sentences_processed


def post_process_sentences(sentences: List[str], modify=True) -> List[str]:
    """
    Post-processes a list of sentences by changing punctuation marks back to words
    and applying additional modifications.

    Args:
        sentences (List[str]): The list of sentences to be post-processed.

    Returns:
        List[str]: A list of post-processed sentences.
    """
    post_processed_sentences = []
    for sentence in sentences:
        original_sentence = sentence
        for punctuation, word in reverse_replacement_dict.items():
            sentence = sentence.replace(punctuation, word)
        sentence = re.sub(
            r"\d+",
            lambda x: num2words(int(x.group()), lang="pt_BR", to="cardinal"),
            sentence,
        ).replace(",", "")
        sentence = sentence.lower().strip()
        if modify and len(original_sentence.split()) > 1:
            if sentence.endswith("ponto") and random.random() < 0.33:
                sentence = sentence[:-6]  # Remove "ponto" from the end
            if random.random() < 0.25:
                sentence = random.choice(
                    ["parágrafo " + sentence, "nova linha " + sentence]
                )
            elif random.random() < 0.25:
                sentence = random.choice(
                    ["ponto parágrafo " + sentence, "ponto nova linha " + sentence]
                )
            if not sentence.endswith("ponto"):
                if random.random() < 0.25:
                    sentence += random.choice([" ponto parágrafo", " ponto nova linha"])
            else:
                if random.random() < 0.25:
                    sentence += random.choice([" parágrafo", " nova linha"])
        post_processed_sentences.append(sentence.strip())
    return post_processed_sentences


def print_sentences_comparison(sentences: List[str], augmented_sentences: List[str]):
    """
    Prints the original and augmented sentences for comparison.

    Args:
        sentences (List[str]): The original sentences.
        augmented_sentences (List[str]): The augmented sentences.
    """
    print("\nResults:")
    print("-------------")
    for i, (original, augmented) in enumerate(zip(sentences, augmented_sentences)):
        print(f"src {i + 1}: {original.strip()}")
        print(f"aug {i + 1}: {augmented.strip()}\n")


def remove_equal_sentences(
    sentences: List[str], final_sentences: List[str]
) -> List[str]:
    """
    Removes duplicate sentences from the final list of sentences.

    Args:
        sentences (List[str]): The original list of sentences.
        final_sentences (List[str]): The final list of sentences.

    Returns:
        List[str]: A list of unique sentences.
    """
    sentences_set = set(sentences)
    modified_sentences = []

    for sentence in final_sentences:
        if sentence in sentences_set:
            continue
        modified_sentences.append(sentence.strip())

    return modified_sentences
