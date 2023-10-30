#!/usr/bin/env python3
import requests
import gzip
import numpy as np
import librosa
import soundfile as sf

from pathlib import Path
from typing import List, Optional, Tuple
from text.utils import pre_process_sentences


def read_sentences_corpus(
    filename: str, max_sentences: Optional[str] = None, only_unique: bool = False
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
    unique_sentences = set()
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
            sentence = line.strip()
            if only_unique:
                unique_sentences.add(sentence)
            else:
                sentences.append(sentence)
            if max_sentences not in [None, -1, "100%"]:
                if (
                    len(sentences) == max_sentences
                    or len(unique_sentences) == max_sentences
                ):
                    break

    if not only_unique:
        sentences = pre_process_sentences(sentences)
    else:
        sentences = pre_process_sentences(list(unique_sentences))
    return sentences


def append_sentences_to_file(filename: str, sentences: List[str]) -> None:
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


def download_and_extract(url: str, target_file: str) -> None:
    if not Path(target_file).exists():
        print(f"Downloading file from {url}...")
        response = requests.get(url)

        if response.status_code == 200:
            with open(target_file, "wb") as file:
                file.write(response.content)

            # Extract the downloaded file
            with gzip.open(target_file, "rb") as gz_file:
                with open(target_file[:-3], "wb") as output_file:
                    output_file.write(gz_file.read())

            print(f"File '{target_file}' downloaded and extracted successfully.")
        else:
            print(
                f"Failed to download the file from {url}. Status code: {response.status_code}"
            )


def load_audio(
    audio_file: str, sample_rate: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Read an audio file using Librosa.

    Parameters:
    audio_file (str): The path to the audio file.

    Returns:
    Union[tuple, None]: A tuple containing the audio data (numpy.ndarray) and the sample rate (int).
    If an error occurs during loading, the function returns None.
    """
    try:
        y, sr = librosa.load(audio_file, sr=sample_rate)
        return y, sr
    except Exception as e:
        print(f"Error loading the audio file: {e}")
        raise e


def save_audio(
    audio: np.ndarray, output_path: str, sample_rate: int, fformat: str = "ogg"
) -> None:
    """
    Save audio data using soundfile.

    Parameters:
    audio_data (numpy.ndarray): The audio data to be saved.
    sample_rate (int): The sample rate of the audio data.
    output_file (str): The path where the Ogg Vorbis file will be saved.
    """
    if not output_path.endswith("." + fformat):
        output_path += "." + fformat
    try:
        sf.write(output_path, audio, sample_rate, format="ogg")
    except Exception as e:
        print(f"Error saving the audio as Ogg Vorbis: {e}")
