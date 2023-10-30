#!/usr/bin/env python3
import torch
import re
import random

from typing import List, Callable
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
from num2words import num2words

# from utils.files import download_and_extract
from .utils import (
    reverse_replacement_dict,
    special_words,
    replace_period_comma_with_dot,
)
from nlpaug.augmenter.word import (
    RandomWordAug,
    SynonymAug,
    AntonymAug,
    ContextualWordEmbsAug,
    WordEmbsAug,
)


class SentenceAugmenter:
    def __init__(
        self,
        augmenter_type: str,
        lang: str = "por",
        action: str = None,
        aug_min: int = 1,
        aug_max: int = 10,
        aug_p: float = 0.3,
        device: str = "cpu",
    ):
        self.lang = lang
        self.action = action
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.aug_p = aug_p
        self.device = device

        if "random" in augmenter_type:
            self.augmenter = RandomWordAug(
                action=self.action,
                aug_min=self.aug_min,
                aug_max=self.aug_max,
                aug_p=self.aug_p,
                # stopwords=STOPWORDS,
            )
        elif augmenter_type == "synonym":
            self.augmenter = SynonymAug(
                # aug_src="ppdb",
                # model_path="ppdb-1.0-xxxl-lexical",
                aug_src="wordnet",
                aug_min=self.aug_min,
                aug_max=self.aug_max,
                aug_p=self.aug_p,
                lang=self.lang,
                # stopwords=STOPWORDS,
            )
        elif augmenter_type == "antonym":
            self.augmenter = AntonymAug(
                aug_min=self.aug_min,
                aug_max=self.aug_max,
                aug_p=self.aug_p,
                lang=self.lang,
                # stopwords=STOPWORDS,
            )
        elif "embed_bert" in augmenter_type:
            self.augmenter = ContextualWordEmbsAug(
                model_path="neuralmind/bert-large-portuguese-cased",
                # model_path="bert-base-multilingual-cased",
                action=self.action,
                aug_min=self.aug_min,
                aug_max=self.aug_max,
                aug_p=self.aug_p,
                device=self.device,
                batch_size=32,
            )
        elif "embed" in augmenter_type:
            # Warning: this method sucks.
            # download_and_extract(
            #    "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz",
            #    "cc.pt.300.vec",
            # )
            self.augmenter = WordEmbsAug(
                model_type="fasttext",
                model_path="cc.pt.300.vec",
                action=self.action,
                aug_min=self.aug_min,
                aug_max=self.aug_max,
                aug_p=self.aug_p,
            )
        else:
            raise ValueError("Invalid augmenter_type")

    def augment_sentences(self, sentences: List[str]) -> List[str]:
        print(f"Augmenting {len(sentences)} sentences with {self.augmenter}...")
        augmented_sentences = self.augmenter.augment(sentences)
        if augmented_sentences == sentences:
            print("Warning: augmented sentences are equal to original sentences")
        return augmented_sentences


def translate_sentences_api(
    sentences: List[str], source_lang: str, target_lang: str, output_file: str
) -> None:
    """
    Translates a list of sentences from the source language to the target language
    using the Google Translator API.

    Args:
        sentences (List[str]): The list of sentences to be translated.
        source_lang (str): The source language code (e.g., "pt" for Portuguese).
        target_lang (str): The target language code (e.g., "en" for English).
        output_file (str): The output file to write the translated sentences to.

    Note:
        This function is currently not used in the pipeline and appends the results
        to an output file on-the-fly, since it is more error-prone.
    """
    if output_file is None:
        output_file = "translated.txt"
    proxy = {
        "http": "",
        "https": "",
    }
    translator = GoogleTranslator(source=source_lang, target=target_lang, proxies=proxy)

    skips_count = 0
    with open(output_file, "a") as f:
        for sentence in tqdm(sentences):
            if skips_count > 30:
                print(
                    "[x] Too many translations skipped in a row. IP blocked? Aborting the program."
                )
                exit(1)
            # If the sentence is composed only of special characters, skip it.
            if re.fullmatch(r"[\s\W]*", sentence):
                continue
            else:
                try:
                    translation = translator.translate(sentence)
                except Exception as e:
                    print(f"[!] Skipping translating sentence '{sentence}': {e}")
                    skips_count += 1
                    continue
            skips_count = 0
            if (
                len(translation.split()) > 1
                and random.random() <= 0.35
                and translation.split()[-2] != " "
                and translation.split()[-1] not in special_words[target_lang]
            ):
                translation += "."
                if random.random() <= 0.4:
                    if target_lang == "en":
                        translation += random.choice([" new line", " paragraph"])
                    elif target_lang == "pt":
                        translation += random.choice([" nova linha", " parágrafo"])
                    elif target_lang == "es":
                        translation += random.choice([" nueva línea", " párrafo"])

            for punctuation, word in reverse_replacement_dict[target_lang].items():
                translation = translation.replace(punctuation, word)
            translation = re.sub(
                r"\d+",
                lambda x: num2words(int(x.group()), lang=target_lang, to="cardinal"),
                translation,
            ).replace(",", "")
            translation = translation.lower().strip()
            translation = re.findall(r"\b[A-Za-zÀ-ÖØ-öø-ÿ]+\b", translation)
            if target_lang == "en":
                translation = replace_period_comma_with_dot(translation)
            translation = " ".join(translation)

            f.write(translation + "\n")
            f.flush()


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
    # proxies = {"http": None, "https": None}
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    backtranslator = GoogleTranslator(source=target_lang, target=source_lang)
    backtranslations = []

    skips_count = 0
    for sentence in tqdm(sentences):
        # If the sentence is composed only of special characters, skip it
        if re.fullmatch(r"[\s\W]*", sentence):
            continue
        if skips_count > 30:
            print(
                "[x] Too many translations skipped in a row. IP blocked? Aborting the program."
            )
            exit(1)
        try:
            translation = translator.translate(sentence)
        except Exception as e:
            print(f"[!] Skipping translating sentence '{sentence}': {e}")
            skips_count += 1
            continue
        try:
            backtranslator = backtranslator.translate(translation)
        except Exception as e:
            print(f"[!] Skipping backtranslating sentence '{translation}': {e}")
            skips_count += 1
            continue
        skips_count = 0
        backtranslation = backtranslator.translate(translation)
        backtranslations.append(backtranslation)
        print(f"{sentence} -> {translation} -> {backtranslation}")

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
        if re.fullmatch(r"[\s\W]*", sentence):
            continue
        transl = pten_pipeline("translate Portuguese to English: " + sentence)[0][
            "generated_text"
        ]
        backtransl = enpt_pipeline("translate English to Portuguese: " + transl)[0][
            "generated_text"
        ]
        translations.append(backtransl)

    return translations


def backtranslation_pipeline(
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
    augmentations: List[str], translate_mode: str, lang: str, device: str
) -> List[Callable]:
    """
    Creates an augmentation sequence based on a list of augmentation techniques.

    This function generates a sequence of augmentation functions to be applied to input sentences.
    It supports various text augmentation techniques.

    Args:
        augmentations (List[str]): A list of augmentation techniques to apply.
        translate_mode (str): The translation mode when using translation augmentation techniques.
                              'local' uses a local model, for example.
        lang (str): The language code to be used for translation.
        device (str): The device to use for translation (e.g., 'cpu' or 'cuda').

    Returns:
        List[Callable]: A list of callable functions, where each function represents an augmentation
        technique to apply to input sentences.
    """
    augmentation_sequence = []
    action = "delete"
    for aug in augmentations:
        if "random" in aug or "nym" in aug or "embed" in aug:
            if "swap" in aug or "swp" in aug:
                action = "swap"
            elif "del" in aug or "delete" in aug:
                action = "delete"
            elif "subs" in aug or "substitute" in aug:
                action = "substitute"
            elif "ins" in aug or "insert" in aug:
                action = "insert"
            augmenter = SentenceAugmenter(aug, action=action)
            augmentation_sequence.append(
                lambda x, augmenter=augmenter: augmenter.augment_sentences(x)
            )
        elif aug == "translate" or aug == "backtranslate":
            augmentation_sequence.append(
                lambda x: backtranslation_pipeline(x, translate_mode, lang, device)
            )
    return augmentation_sequence
