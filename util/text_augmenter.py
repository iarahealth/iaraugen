#!/usr/bin/env python3
from typing import List
from nlpaug.augmenter.word import RandomWordAug, SynonymAug


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
        if augmenter_type == "random":
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
