#!/usr/bin/env python3
import torch
from typing import List, Callable
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nlpaug.augmenter.word import RandomWordAug, SynonymAug
from tqdm import tqdm


class SentenceAugmenter:
    def __init__(
        self,
        augmenter_type: str,
        lang: str = "por",
        action: str = None,
        aug_min: int = 1,
        aug_max: int = 10,
        aug_p: float = 0.3,
    ):
        self.lang = lang
        self.action = action
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.aug_p = aug_p
        if "random" in augmenter_type:
            self.augmenter = RandomWordAug(
                action=self.action,
                aug_min=self.aug_min,
                aug_max=self.aug_max,
                aug_p=self.aug_p,
            )
        elif augmenter_type == "synonym":
            self.augmenter = SynonymAug(
                aug_src="wordnet",
                aug_min=self.aug_min,
                aug_max=self.aug_max,
                aug_p=self.aug_p,
                lang=self.lang,
            )
        else:
            raise ValueError("Invalid augmenter_type")

    def augment_sentences(self, sentences: List[str]) -> List[str]:
        print(f"Augmenting {len(sentences)} sentences with {self.augmenter}...")
        return self.augmenter.augment(sentences)


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
    augmentations: List[str], translate_mode: str, lang: str, device: str
) -> List[Callable]:
    augmentation_sequence = []
    action = "delete"
    for aug in augmentations:
        if "random" in aug or "synonym" in aug:
            if "swap" in aug or "swp" in aug:
                action = "swap"
            augmenter = SentenceAugmenter(aug, action=action)
            augmentation_sequence.append(
                lambda x, augmenter=augmenter: augmenter.augment_sentences(x)
            )
        elif aug == "translate":
            augmentation_sequence.append(
                lambda x: translation_pipeline(x, translate_mode, lang, device)
            )
    return augmentation_sequence
