#!/usr/bin/env python3
import argparse
import random
import torch
from typing import List
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
from utils.text_augmenter import SentenceAugmenter
from utils.text import (
    post_process_sentences,
    print_sentences_comparison,
    remove_equal_sentences,
)
from utils.files import read_sentences_corpus, append_sentences_to_file

"""
example usage:
./txt_aug.py corpus.tok --aug translate random --action delete --maxs 10 --lang en --append
"""


def backtranslate_sentences_api(
    sentences: List[str], source_lang: str, target_lang: str
) -> List[str]:
    """
    Backtranslates a list of sentences from the source language to the target language
    using the Google Translator API.

    Args:
        sentences (List[str]): The list of sentences to be backtranslated.
        source_lang (str): The source language code (e.g., "pt" for Portuguese).
        target_lang (str): The target language code (e.g., "en" for English).

    Returns:
        List[str]: A list of backtranslated sentences.
    """
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translations = translator.translate_batch(sentences)
    backtranslator = GoogleTranslator(source=target_lang, target=source_lang)
    backtranslations = backtranslator.translate_batch(translations)

    return backtranslations


def backtranslate_sentences_local(
    sentences: List[str], source_lang: str, target_lang: str, device: str = "cpu"
) -> List[str]:
    """
    Backtranslates a list of sentences from the source language to the target language,
    and then back to the source language using a local model.

    Args:
        sentences (List[str]): The list of sentences to be backtranslated.
        source_lang (str): The source language code (e.g., "pt" for Portuguese).
        target_lang (str): The target language code (e.g., "en" for English).
        device (str): The device to run the model on (e.g., "cpu" or "cuda").

    Returns:
        List[str]: A list of backtranslated sentences.

    Note:
        nlpaug has a backtranslation module, but it only officially supports Helsinki-NLP,
        but we do not have a Helsinki model for Portuguese -> English. So we use the T5 model
        directly from HuggingFace.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        f"unicamp-dl/translation-{source_lang}-{target_lang}-t5"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        f"unicamp-dl/translation-{source_lang}-{target_lang}-t5"
    )
    model.to(torch.device(device))
    backtokenizer = AutoTokenizer.from_pretrained(
        f"unicamp-dl/translation-{target_lang}-{source_lang}-t5"
    )
    backmodel = AutoModelForSeq2SeqLM.from_pretrained(
        f"unicamp-dl/translation-{target_lang}-{source_lang}-t5"
    )
    backmodel.to(torch.device(device))
    pten_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    enpt_pipeline = pipeline(
        "text2text-generation", model=backmodel, tokenizer=backtokenizer
    )

    print(f"Backtranslating {len(sentences)} sentences...")
    translations: List[str] = []
    for sentence in tqdm(sentences):
        transl = pten_pipeline("translate Portuguese to English: " + sentence)[0][
            "generated_text"
        ]
        backtransl = enpt_pipeline("translate English to Portuguese: " + transl)[0][
            "generated_text"
        ]
        translations.append(backtransl)

    return translations


def translation_pipeline(
    sentences: List[str], translate_mode: str, lang: str, device: str
) -> List[str]:
    """
    Runs the translation pipeline to backtranslate a list of sentences.

    Args:
        sentences (List[str]): The list of sentences to be translated.
        translate_mode (str): Use local model or API to translate.
        lang (str): The target language code (e.g., "en" for English).
        device (str): The device to run the model on (e.g., "cpu" or "cuda").

    Returns:
        List[str]: A list of translated sentences.
    """
    augmented_sentences: List[str] = []
    print(f"Backtranslating sentences pt->{lang}->pt...")
    if translate_mode == "local":
        augmented_sentences = backtranslate_sentences_local(
            sentences, "pt", lang, device
        )
    elif translate_mode == "google":
        augmented_sentences = backtranslate_sentences_api(sentences, "pt", lang)
    assert len(augmented_sentences)
    return augmented_sentences


def create_augmentation_sequence(
    augmentations: List[str], action: str, translate_mode: str, lang: str, device: str
) -> List[callable]:
    augmentation_sequence = []
    for aug in augmentations:
        if aug == "random" or aug == "synonym":
            augmenter = SentenceAugmenter(aug, action=action)
            augmentation_sequence.append(lambda x: augmenter.augment_sentences(x))
        elif aug == "translate":
            augmentation_sequence.append(
                lambda x: translation_pipeline(x, translate_mode, lang, device)
            )
    return augmentation_sequence


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
        choices=["random", "translate", "synonym"],
        help="Augmentation type to perform",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["delete", "swap"],
        default="delete",
        help="Action to perform",
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

    augmentation_sequence = create_augmentation_sequence(
        args.aug, args.action, args.translate_mode, args.lang, args.device
    )

    augmented_sentences = sentences
    for i, aug_fn in enumerate(augmentation_sequence):
        print(f"Augmentation step {i + 1} of {len(augmentation_sequence)}:")
        augmented_sentences = aug_fn(augmented_sentences)

    augmented_sentences = post_process_sentences(augmented_sentences)
    sentences = post_process_sentences(sentences)
    print_sentences_comparison(sentences, augmented_sentences)

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
