#!/usr/bin/env python3

from openai import OpenAI
from typing import List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

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
    client, query: str, return_type: str, model: str = "gpt-3.5-turbo"
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
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": query}]
    )
    is_truncated = response.usage.total_tokens >= max_tokens
    response = [
        line
        for line in response.choices[0].message.content.split("\n")
        if line.strip() != ""
    ]
    if is_truncated and len(response) > 0:
        response.pop()
    return response
