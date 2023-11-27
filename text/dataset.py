#!/usr/bin/env python3

import re
import random

from typing import List
from num2words import num2words
from text_to_num import text2num
from .utils import reverse_replacement_dict


def replace_period_comma_with_dot(words: List[str]) -> List[str]:
    """
    English: "period" between two spoken numbers is replaced by "dot".
    """

    def is_spoken_number(word: str) -> bool:
        try:
            text2num(word, lang="en")
            return True
        except ValueError:
            return False

    for i in range(len(words) - 2):
        if is_spoken_number(words[i]) and is_spoken_number(words[i + 2]):
            if words[i + 1] in ["period", "comma"]:
                words[i + 1] = "dot"
            elif words[i + 1] in ["hyphen"]:
                words[i + 1] = ""

    return words


def create_corpus(sentences: List[str], lang: str) -> List[str]:
    clean_sentences: List[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        for punctuation, word in reverse_replacement_dict[lang].items():
            sentence = sentence.replace(punctuation, word)
        sentence = re.sub(
            r"\d+",
            lambda x: num2words(
                int(x.group()),
                lang="pt_BR" if lang == "pt" else lang,
                to="cardinal",
            ),
            sentence,
        ).replace(",", "")
        sentence = sentence.lower().strip()
        sentence = re.findall(r"\b[A-Za-zÀ-ÖØ-öø-ÿ]+\b", sentence)
        if not sentence:
            continue
        if sentence[-1] == "punto" and lang == "es":
            if random.random() < 0.25:
                sentence[-1] = "punto y aparte"
            elif random.random() > 0.9:
                sentence[-1] = "punto final"
        elif sentence[-1] == "period" and lang == "en":
            if random.random() < 0.25:
                sentence[-1] = "period new line"
            elif random.random() > 0.9:
                sentence[-1] = "period new paragraph"
        elif sentence[-1] == "ponto" and lang == "pt":
            if random.random() < 0.25:
                sentence[-1] = "ponto parágrafo"
            elif random.random() > 0.9:
                sentence[-1] = "ponto nova linha"
        if lang == "en":
            sentence = replace_period_comma_with_dot(sentence)
        sentence = " ".join(sentence)
        clean_sentences.append(sentence)

    return clean_sentences
