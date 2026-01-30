# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import openai

"""
NOTE: 
    Available functions:
        - call_vllm_api: using vllm self-served models
        - openai_generate: using openai models
"""
########################################################################################################
import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY not found in environment or .env file.")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")  # exemple


def custom_api(prompt, model, temperature=0.0, max_tokens=1024, **kwargs):
    """
    Call OpenRouter API with dynamic model selection.

    Args:
        prompt: The prompt to send
        model: The model identifier (e.g., 'openai/gpt-4o-mini')
        temperature: Sampling temperature
        max_tokens: Max tokens in response

    Returns:
        Generated text from the model
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optionnels (recommandés par OpenRouter pour l'analytics/classement)
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
        "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "HalluLens"),
    }

    payload = {
        "model": model,  # Use the dynamically provided model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def generate(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0):
    """
    Generate text using OpenRouter API with dynamic model selection.

    Args:
        prompt: The prompt to send
        model: The model identifier (e.g., 'openai/gpt-4o-mini')
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Max tokens in response
        port: Unused (for backward compatibility)
        i: Unused (for backward compatibility)

    Returns:
        Generated text from the model
    """
    # Use OpenRouter API with the provided model
    return custom_api(
        prompt, model=model, temperature=temperature, max_tokens=max_tokens
    )
    # Alternative: return call_vllm_api(prompt, model, temperature, top_p, max_tokens, port, i)


CUSTOM_SERVER = "0.0.0.0"  # you may need to change the port

model_map = {
    "meta-llama/Llama-3.1-405B-Instruct-FP8": {
        "name": "llama3.1_405B",
        "server_urls": [f"http://{CUSTOM_SERVER}:8000/v1"],
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "name": "llama3.3_70B",
        "server_urls": [f"http://{CUSTOM_SERVER}:8000/v1"],
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "name": "llama3.1_70B",
        "server_urls": [f"http://{CUSTOM_SERVER}:8000/v1"],
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "name": "llama3.1_8B",
        "server_urls": [f"http://{CUSTOM_SERVER}:8000/v1"],
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "name": "mistral7B",
        "server_urls": [f"http://{CUSTOM_SERVER}:8000/v1"],
    },
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "name": "Mistral-Nemo-Instruct-2407",
        "server_urls": [f"http://{CUSTOM_SERVER}:8000/v1"],
    },
}
########################################################################################################


def call_vllm_api(
    prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0
):
    # OpenRouter-only mode: route vLLM calls to OpenRouter
    return custom_api(
        prompt, model=model, temperature=temperature, max_tokens=max_tokens
    )


def openai_generate(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512):
    # OpenRouter-only mode: route OpenAI calls to OpenRouter
    return custom_api(
        prompt, model=model, temperature=temperature, max_tokens=max_tokens
    )
