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
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-4o-mini": 128000,
}


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def make_chatgpt_query(
    client, query: str, return_type: str, model: str = "gpt-4o-mini"
) -> List[str]:
    """
    Makes a query to the ChatGPT model and returns the generated response.

    Args:
        query (str): The user's query.
        model (str): The name of the ChatGPT model.

    Returns:
        List[str]: A list containing the model's response.
    """
    max_tokens = MAX_TOKENS[model]
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": query}]
    )
    is_truncated = response.usage.total_tokens >= max_tokens
    response_text = response.choices[0].message.content
    response_lines = [line for line in response_text.split("\n") if line.strip() != ""]
    if is_truncated and len(response_lines) > 0:
        response_lines.pop()
    return response_lines
