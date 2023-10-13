#!/usr/bin/env python3
import argparse
import random
import re
import openai

from typing import List
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from util.files import append_sentences_to_file, read_file
from util.text import post_process_sentences

MAX_TOKENS = {
    # https://platform.openai.com/docs/models/gpt-4
    # https://platform.openai.com/docs/models/gpt-3-5
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
}


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def make_chatgpt_query(
    query: str, api_key: str, return_type: str, model: str = "gpt-3.5-turbo"
) -> List[str]:
    """
    Makes a query to the ChatGPT model and returns the generated response.

    Args:
        query (str): The user's query.
        api_key (str): The API key for accessing the ChatGPT model.
        model (str): The name of the ChatGPT model.

    Returns:
        List[str]: A list containing the model's response.
    """
    max_tokens = MAX_TOKENS[model]
    openai.api_key = api_key
    # check max_tokens
    response = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": query}]
    )
    is_truncated = response["usage"]["total_tokens"] >= max_tokens
    response = [
        line
        for line in response["choices"][0]["message"]["content"].split("\n")
        if line.strip() != ""
    ]
    if is_truncated and len(response) > 0:
        response.pop()
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentence/word generation using ChatGPT"
    )
    parser.add_argument(
        "--input_file", type=str, default=None, help="Input file with words"
    )
    parser.add_argument(
        "--num", type=int, default=None, help="Number of sentences or words to generate"
    )
    parser.add_argument(
        "--context",
        type=str,
        default="radiologia médica",
        help="Context of the generated sentences",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="A query to OpenAI's ChatGPT; the first number detected in the query will be replaced by the number of sentences to generate",
    )
    parser.add_argument(
        "--return_type",
        type=str,
        default="frases",
        help="Type of data to generate (default: frases)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-16k",
        help="ChatGPT model to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=451,
        help="Random seed (default: 451)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to write generated sentences",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if args.query is None:
        if args.return_type == "frases":
            args.query = f"No contexto de {args.context}, gere {args.num} {args.return_type} contendo o termo '[MASK]', separadas por nova linha."
        else:
            args.query = f"No contexto de {args.context}, gere {args.num} {args.return_type} separadas por nova linha."
    else:
        args.num = (
            int(re.search(r"\d+", args.query).group())
            if re.search(r"\d+", args.query)
            else None
        )

    if args.input_file:
        wordlist = read_file(args.input_file)
    else:
        if args.return_type == "frases" and "[MASK]" in args.query:
            wordlist = []
            while True:
                word = input("Enter a word (or press Enter to finish): ")
                if word == "":
                    break
                wordlist.append(word)
        else:
            wordlist = [""]

    response_sentences: List[str] = []
    original_query = args.query
    for word in tqdm(wordlist):
        word = word.strip()
        query = re.sub(r"\[MASK\]", word, original_query)
        number_of_sentences_left = args.num

        while number_of_sentences_left > 0:
            print(f"\nNumber of sentences left: {number_of_sentences_left}")
            print(f"Querying OpenAI's {args.model} with '{query}'...")
            query_response = make_chatgpt_query(
                query, args.api_key, return_type=args.return_type, model=args.model
            )
            print(query_response)
            response_sentences.extend(
                [s.split(" ", 1)[1] if s[0].isdigit() else s for s in query_response]
            )
            number_of_sentences_left -= len(query_response)
            query = re.sub(r"\d+", str(number_of_sentences_left), query)
        print()

    generated_sentences = post_process_sentences(response_sentences, modify=True)

    print("\nFinal results:")
    print("-------------------")
    for sentence in generated_sentences:
        print(sentence)
    print(f"\nTotal: {len(generated_sentences)} sentences")
    print("-------------------\n")

    if args.output:
        print(f"Appending generated sentences to {args.output}...")
        append_sentences_to_file(args.output, generated_sentences)
