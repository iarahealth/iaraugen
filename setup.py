from setuptools import setup, find_packages

authors = ["Pedro Probst", "Bernardo Henz"]

setup(
    name="iaraugen",
    version="1.0.0",
    author=", ".join(authors),
    description="Data augmentation/generation functions used at Iara Health (speech-to-text).",
    packages=find_packages(),
    install_requires=[
        "torch",
        "deep-translator",
        "transformers",
        "sentencepiece",
        "tenacity",
        "openai",
        "nlpaug",
        "nltk",
        "num2words",
        "tqdm",
    ],
)
