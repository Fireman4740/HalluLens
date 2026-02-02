# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import random
import threading

"""
NOTE: 
    Available functions:
        - generate: Main entry point - routes to OpenRouter or LM Studio
        - custom_api: Call OpenRouter API
        - lm_studio_api: Call LM Studio local API
        - call_vllm_api: using vllm self-served models (routes to custom_api)
        - openai_generate: using openai models (routes to custom_api)
"""
########################################################################################################
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

# LM Studio configuration
LM_STUDIO_URL = os.getenv(
    "LM_STUDIO_URL", "http://10.10.12.21:1234/v1/chat/completions"
)
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "openai/gpt-oss-20b")
USE_LM_STUDIO = os.getenv("USE_LM_STUDIO", "false").lower() in ("true", "1", "yes")

if not OPENROUTER_API_KEY and not USE_LM_STUDIO:
    print("Warning: OPENROUTER_API_KEY not found in environment or .env file.")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")  # exemple

# Global request coordination (shared across all threads)
_MAX_IN_FLIGHT = max(1, int(os.getenv("OPENROUTER_MAX_IN_FLIGHT", "64")))
_IN_FLIGHT_SEM = threading.Semaphore(_MAX_IN_FLIGHT)
_COOLDOWN_LOCK = threading.Lock()
_COOLDOWN_UNTIL = 0.0
_SESSION = requests.Session()


def _wait_for_cooldown():
    while True:
        with _COOLDOWN_LOCK:
            now = time.time()
            if now >= _COOLDOWN_UNTIL:
                return
            sleep_s = _COOLDOWN_UNTIL - now
        time.sleep(min(sleep_s, 1.0))


def _set_cooldown(seconds: float):
    if seconds <= 0:
        return
    with _COOLDOWN_LOCK:
        global _COOLDOWN_UNTIL
        _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.time() + seconds)


def _get_retry_after_seconds(response):
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except ValueError:
        return None


def _post_with_retry(
    url,
    headers,
    payload,
    timeout,
    max_retries=5,
    base_sleep=1.0,
    max_sleep=30.0,
):
    for attempt in range(max_retries + 1):
        try:
            _wait_for_cooldown()
            _IN_FLIGHT_SEM.acquire()
            try:
                r = _SESSION.post(url, headers=headers, json=payload, timeout=timeout)
            finally:
                _IN_FLIGHT_SEM.release()

            if r.status_code == 429:
                if attempt >= max_retries:
                    r.raise_for_status()
                retry_after = _get_retry_after_seconds(r)
                sleep_s = (
                    retry_after if retry_after is not None else base_sleep * (2**attempt)
                )
                sleep_s = min(max_sleep, sleep_s) + random.uniform(0, 0.5)
                _set_cooldown(sleep_s)
                time.sleep(sleep_s)
                continue

            if r.status_code in (408, 500, 502, 503, 504):
                if attempt >= max_retries:
                    r.raise_for_status()
                sleep_s = min(max_sleep, base_sleep * (2**attempt)) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return r
        except requests.exceptions.RequestException:
            if attempt >= max_retries:
                raise
            sleep_s = min(max_sleep, base_sleep * (2**attempt)) + random.uniform(0, 0.5)
            time.sleep(sleep_s)


def _extract_content(data):
    if not isinstance(data, dict):
        return ""
    if data.get("error"):
        return ""
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def lm_studio_api(
    prompt,
    model=None,
    temperature=0.0,
    max_tokens=1024,
    top_p=1.0,
    **kwargs,
):
    """
    Call LM Studio local API.

    Args:
        prompt: The prompt to send
        model: The model identifier (optional, uses LM_STUDIO_MODEL if not provided)
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        top_p: Top-p sampling parameter

    Returns:
        Generated text from the model
    """
    url = os.getenv("LM_STUDIO_URL", LM_STUDIO_URL)
    model_name = model or os.getenv("LM_STUDIO_MODEL", LM_STUDIO_MODEL)

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"LM Studio API error: {e}")
        raise


def custom_api(prompt, model, temperature=0.0, max_tokens=1024, top_p=1.0, **kwargs):
    """
    Call OpenRouter API with dynamic model selection.

    Args:
        prompt: The prompt to send
        model: The model identifier (e.g., 'openai/gpt-4o-mini')
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        top_p: Top-p sampling parameter

    Returns:
        Generated text from the model
    """
    # Check if we should use LM Studio instead (read dynamically at call time)
    use_lm_studio_now = os.getenv("USE_LM_STUDIO", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    if use_lm_studio_now or model.startswith("lm-studio/"):
        actual_model = (
            model.replace("lm-studio/", "") if model.startswith("lm-studio/") else None
        )
        return lm_studio_api(
            prompt, model=actual_model, temperature=temperature, max_tokens=max_tokens, top_p=top_p
        )

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

    max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "5"))
    base_sleep = float(os.getenv("OPENROUTER_RETRY_BASE_SECONDS", "1.0"))
    max_sleep = float(os.getenv("OPENROUTER_RETRY_MAX_SECONDS", "30.0"))
    empty_retries = int(os.getenv("OPENROUTER_EMPTY_RETRIES", "2"))

    r = _post_with_retry(
        url,
        headers=headers,
        payload=payload,
        timeout=180,
        max_retries=max_retries,
        base_sleep=base_sleep,
        max_sleep=max_sleep,
    )

    for attempt in range(empty_retries + 1):
        data = r.json()
        content = _extract_content(data)
        if content:
            return content
        if attempt >= empty_retries:
            raise ValueError("OpenRouter returned empty response content")
        sleep_s = min(max_sleep, base_sleep * (2**attempt)) + random.uniform(0, 0.5)
        time.sleep(sleep_s)
        r = _post_with_retry(
            url,
            headers=headers,
            payload=payload,
            timeout=180,
            max_retries=max_retries,
            base_sleep=base_sleep,
            max_sleep=max_sleep,
        )


def generate(prompt, model, temperature=0.0, top_p=1.0, max_tokens=512, port=None, i=0):
    """
    Generate text using OpenRouter API or LM Studio with dynamic model selection.

    Args:
        prompt: The prompt to send
        model: The model identifier (e.g., 'openai/gpt-4o-mini' or 'lm-studio/model-name')
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Max tokens in response
        port: Unused (for backward compatibility)
        i: Unused (for backward compatibility)

    Returns:
        Generated text from the model

    Usage:
        # Use OpenRouter (default)
        generate("Hello", "openai/gpt-4o-mini")

        # Use LM Studio explicitly
        generate("Hello", "lm-studio/openai/gpt-oss-20b")

        # Use LM Studio via environment variable (set USE_LM_STUDIO=true)
        generate("Hello", "any-model")
    """
    # Use OpenRouter API or LM Studio based on configuration
    return custom_api(
        prompt, model=model, temperature=temperature, max_tokens=max_tokens, top_p=top_p
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
