#!/usr/bin/env python3
import random
import re

from typing import List
from num2words import num2words
from text_to_num import text2num


def add_period_and_capitalize(sentence: str, add_period: bool = False) -> str:
    """
    Adds a period at the end of the sentence if it doesn't already have one,
    capitalizes the sentences, and returns the modified sentence.

    Args:
        sentence (str): The input sentence.

    Returns:
        str: The modified sentence with added period and capitalized.
    """
    if add_period:
        if sentence[-1] not in [
            "...",
            ".",
            ":",
            "!",
            "?",
            ";",
            ",",
            "--",
            "-",
            "/",
            "+",
            "(",
            "{",
        ]:
            sentence += "."
    sentences = sentence.split(".")
    return ". ".join(s.strip().capitalize() for s in sentences).strip()


def remove_non_alphabet_words(input_string: str) -> str:
    """
    Removes non-alphabetic characters from a string.

    Args:
        input_string (str): The input string.

    Returns:
        str: String with non-alphabetic characters remove.
    """
    words = re.findall(r"\b[A-Za-zÀ-ÖØ-öø-ÿ]+\b", input_string)
    cleaned_string = " ".join(words)
    return cleaned_string


def pre_process_sentences(sentences: List[str], add_period: bool = False) -> List[str]:
    sentences_processed = []
    for s in sentences:
        s = s.strip()
        for word, punctuation in replacement_dict.items():
            s = s.replace(word, punctuation)
        if s.isspace() or s == "":
            continue
        s = s.strip()
        s = add_period_and_capitalize(s, add_period)
        # "Glue" the punctuation marks to the previous character.
        s = re.sub(r"\s+([.,;!?:)}]|(\.{3,}))", r"\1", s)
        # "Glue" the ( and { to the next character.
        s = re.sub(r"\(\s*(\w)", r"(\1", s)
        s = re.sub(r"\{\s*(\w)", r"{\1", s)  # Glue "{" to the next character
        sentences_processed.append(s)
    return sentences_processed


def post_process_sentences(
    sentences: List[str], lang: str = "pt", modify: bool = True
) -> List[str]:
    """
    Post-processes a list of sentences by changing punctuation marks back to words
    and applying additional modifications.

    Args:
        sentences (List[str]): The list of sentences to be post-processed.
        lang (str, optional): The language of the sentences. Defaults to "pt".
        modify (bool, optional): Whether to apply additional modifications to the

    Returns:
        List[str]: A list of post-processed sentences.
    """
    post_processed_sentences = []
    for sentence in sentences:
        original_sentence = sentence
        for punctuation, word in reverse_replacement_dict[lang].items():
            sentence = sentence.replace(punctuation, word)
        sentence = re.sub(
            r"\d+",
            lambda x: num2words(
                int(x.group()), lang="pt_BR" if lang == "pt" else lang, to="cardinal"
            ),
            sentence,
        ).replace(",", "")
        sentence = sentence.lower().strip()
        if modify and len(original_sentence.split()) > 1:
            if (
                sentence.endswith("ponto")
                or sentence.endswith("punto")
                or sentence.endswith("period")
            ) and random.random() < 0.33:
                sentence = " ".join(sentence.split()[:-1])  # Remove period
            if random.random() < 0.25:
                if lang == "pt":
                    sentence = random.choice(
                        ["parágrafo " + sentence, "nova linha " + sentence]
                    )
                elif lang == "en":
                    sentence = random.choice(
                        ["paragraph " + sentence, "new line " + sentence]
                    )
                elif lang == "es":
                    sentence = random.choice(
                        ["párrafo " + sentence, "nueva línea " + sentence]
                    )
            elif random.random() < 0.25:
                if lang == "pt":
                    sentence = random.choice(
                        ["ponto parágrafo " + sentence, "ponto nova linha " + sentence]
                    )
                elif lang == "en":
                    sentence = random.choice(
                        ["period paragraph " + sentence, "period new line " + sentence]
                    )
                elif lang == "es":
                    sentence = random.choice(
                        [
                            "punto párrafo " + sentence,
                            "punto nueva línea " + sentence,
                        ]
                    )
            if (
                not sentence.endswith("ponto")
                or not sentence.endswith("punto")
                or not sentence.endswith("period")
            ):
                if random.random() < 0.25:
                    if lang == "pt":
                        sentence += random.choice(
                            [" ponto parágrafo", " ponto nova linha"]
                        )
                    elif lang == "en":
                        sentence += random.choice(
                            [" period paragraph", " period new line"]
                        )
                    elif lang == "es":
                        sentence += random.choice(
                            [" punto y aparte", " punto nueva línea"]
                        )
            else:
                if random.random() < 0.25:
                    if lang == "pt":
                        sentence += random.choice([" parágrafo", " nova linha"])
                    elif lang == "en":
                        sentence += random.choice([" paragraph", " new line"])
                    elif lang == "es":
                        sentence += random.choice([" y aparte", " nueva línea"])
        if sentence in ["", " "]:
            continue
        post_processed_sentences.append(sentence.strip())

    post_processed_sentences = [
        " ".join(remove_non_alphabet_words(re.sub(r"\bá\b", "à", sentence)).split())
        for sentence in post_processed_sentences
    ]
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


def replace_period_comma_with_dot(words: List[str]) -> List[str]:
    def is_spoken_number(word):
        """Returns True if the word is a spoken number."""
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


STOPWORDS = [
    "o",
    "a",
    "à",
    "e",
    "é",
    "de",
    "do",
    "da",
    "das",
    "dos",
    "em",
    "que",
    "para",
    "os",
    "as",
    "um",
    "uma",
    "uns",
    "umas",
    "no",
    "na",
    "nos",
    "nas",
    "com",
    "por",
    "se",
    "não",
    "mais",
    "como",
    "foi",
    "ser",
    "são",
    "sua",
]

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
    "abre chaves": "{",
    "fecha chaves": "}",
    "abre chave": "{",
    "fecha chave": "}",
    "barra": "/",
    "uma cruz": "+",
    "duas cruzes": "++",
    "três cruzes": "+++",
}

reverse_replacement_dict = {
    # Warning: order matters!
    "pt": {
        "...": " reticências",
        ".": " ponto ",
        ":": " dois pontos",
        ",": " vírgula ",
        ";": " ponto e vírgula",
        "\n": " parágrafo ",
        "‒-": " travessão ",
        "!": " ponto de exclamação",
        "?": " ponto de interrogação",
        "(": "abre parênteses ",
        ")": " fecha parênteses",
        "-": "",
        "/": " barra ",
        '"': "",
        "+++": "três cruzes",
        "++": "duas cruzes",
        "+": "uma cruz",
        "{": "abre chaves ",
        "}": " fecha chaves",
        "[unk]": "",
        "[UNK]": "",
    },
    "en": {
        "approx.": "approximately",
        "...": " ellipsis",
        ".": " period ",
        ":": " colon",
        ",": " comma",
        ";": " semicolon",
        "\n": " paragraph ",
        "‒-": " dash ",
        "!": " exclamation point",
        "?": " question mark",
        "(": "open parenthesis ",
        ")": " close parenthesis",
        "-": "",
        "/": " slash ",
        '"': random.choice([" quote ", " quotation mark "]),
        "+++": " three crosses",
        "++": " two crosses",
        "+": " one cross",
        "{": "open curly brace ",
        "}": " close curly brace",
        "[unk]": "",
        "[UNK]": "",
    },
    "es": {
        "...": random.choice([" elipsis", " puntos suspensivos"]),
        ".": " punto ",
        ":": " dos puntos",
        ",": " coma",
        ";": " punto y coma",
        "\n": " párrafo ",
        "--": " raya ",
        "!": " signo de exclamación",
        "?": " signo de interrogación",
        "(": " abrir paréntesis ",
        ")": " cerrar paréntesis ",
        "-": "",
        "/": random.choice([" barra diagonal ", " barra "]),
        '"': " comillas ",
        "+++": "tres cruces",
        "++": "dos cruces",
        "+": "una cruz",
        "{": " abrir llaves ",
        "}": " cerrar llaves ",
        "[unk]": "",
        "[UNK]": "",
    },
}

special_words = {
    "en": [
        "from",
        "by",
        "of",
        "at",
        "in",
        "on",
        "to",
        "with",
        "without",
        "for",
        "and",
        "or",
        "nor",
        "but",
        "so",
        "yet",
        "after",
        "before",
        "since",
        "when",
        "while",
        "although",
        "though",
        "even",
        "if",
        "unless",
        "until",
        "as",
        "because",
        "that",
        "whether",
        "where",
        "who",
        "which",
        "what",
        "how",
        "why",
        "whom",
        "whose",
        "whichever",
        "whatever",
        "whoever",
        "whomever",
        "whenever",
        "wherever",
        "however",
    ],
    "es": [
        "de",
        "por",
        "a",
        "en",
        "sobre",
        "a",
        "con",
        "sin",
        "para",
        "y",
        "o",
        "ni",
        "pero",
        "así que",
        "aún",
        "después",
        "antes",
        "desde",
        "cuando",
        "mientras",
        "aunque",
        "a pesar de que",
        "incluso",
        "si",
        "a menos que",
        "hasta",
        "como",
        "porque",
        "que",
        "si",
        "si",
        "donde",
        "quien",
        "cual",
        "que",
        "cómo",
        "por qué",
        "a quien",
        "cuyo",
        "cualquiera",
        "lo que",
        "quienquiera",
        "a quien sea",
        "cuando sea",
        "donde sea",
        "como sea",
    ],
    "pt": [
        "de",
        "por",
        "em",
        "na",
        "sobre",
        "a",
        "com",
        "sem",
        "para",
        "e",
        "ou",
        "nem",
        "mas",
        "então",
        "ainda",
        "depois",
        "antes",
        "desde",
        "quando",
        "enquanto",
        "embora",
        "apesar de",
        "mesmo",
        "se",
        "a menos que",
        "até",
        "como",
        "porque",
        "que",
        "se",
        "se",
        "onde",
        "quem",
        "qual",
        "o que",
        "como",
        "por quê",
        "a quem",
        "cujos",
        "qualquer",
        "o que",
    ],
}
