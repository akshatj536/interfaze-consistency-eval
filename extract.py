"""The extraction request: one prompt, one JSON schema, two OpenAI-compatible endpoints.

Both models are called through the same `openai` client code with the same request body.
The only things that differ per model are the base URL, the API key and the model ID.
"""

import base64
import hashlib
import json
import os
import random
import time

import openai
from openai import OpenAI

# --- The task ------------------------------------------------------------------

PROMPT = """You are extracting data from a photo of a store receipt.

Return JSON with these fields:
- total_price: the final total amount printed on the receipt.
- subtotal_price: the subtotal amount printed on the receipt, or null if no subtotal is printed.
- items: one entry per purchased line item, in the order printed, with its name. Do not list sub-items, toppings or modifiers printed beneath an item as separate entries.

Copy every value exactly as printed on the receipt, including currency symbols, commas and dots. Do not reformat."""

# {"total_price": string, "subtotal_price": string|null, "items": [{"name": string}]}
SCHEMA = {
    "type": "object",
    "properties": {
        "total_price": {"type": "string"},
        "subtotal_price": {"type": ["string", "null"]},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["total_price", "subtotal_price", "items"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "receipt", "strict": True, "schema": SCHEMA},
}

TEMPERATURE = 0  # the most favourable setting for determinism

# --- The two models ------------------------------------------------------------

MODELS = {
    "interfaze": {
        "model": "interfaze-1",
        "base_url": "https://api.interfaze.ai/v1",
        "key_env": "INTERFAZE_API_KEY",
        # By default Interfaze returns a repeated identical request from its "verified cache"
        # (response field vcache=true). That would make runs 2-5 copies of run 1, so the
        # consistency score would measure the cache, not the model. We switch it off.
        "extra_headers": {"x-interfaze-bypass-cache": "true"},
    },
    "gemini": {
        "model": "gemini-3.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "extra_headers": {},  # Gemini has no response cache to switch off
    },
}

# USD per 1M tokens, used only for a rough cost estimate (see DECISIONS.md for sources).
PRICES = {
    "interfaze": {"input": 1.50, "output": 3.50},
    "gemini": {"input": 1.50, "output": 9.00},
}

# --- Calling the API -------------------------------------------------------------

MAX_ATTEMPTS = 8  # Gemini returned bursts of 503 "high demand" during setup
# Errors worth retrying: rate limits, timeouts, dropped connections and 5xx.
# Anything else (e.g. 400 bad request, 401 bad key) fails immediately.
RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def make_client(model_name: str) -> OpenAI:
    cfg = MODELS[model_name]
    # max_retries=0: we do our own retries below, so every attempt is visible and counted.
    return OpenAI(api_key=os.environ[cfg["key_env"]], base_url=cfg["base_url"],
                  timeout=300, max_retries=0)


def build_request(model_name: str, image_bytes: bytes) -> dict:
    """The full chat-completions request body. Identical for both models except `model`."""
    data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
    return {
        "model": MODELS[model_name]["model"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "response_format": RESPONSE_FORMAT,
        "temperature": TEMPERATURE,
        "extra_headers": MODELS[model_name]["extra_headers"],  # HTTP headers, not body
    }


def loggable_request(request: dict, image_bytes: bytes) -> dict:
    """The request with the base64 image swapped for a short fingerprint, for saving to disk."""
    logged = json.loads(json.dumps(request))
    logged["messages"][0]["content"][1]["image_url"]["url"] = (
        f"<base64 jpeg, {len(image_bytes)} bytes, sha256={hashlib.sha256(image_bytes).hexdigest()}>"
    )
    return logged


def call_with_retries(client: OpenAI, request: dict) -> dict:
    """Send one request, retrying transient errors with exponential backoff.

    Returns the full response as a dict, the latency of the successful attempt in seconds,
    and the number of attempts used. Latency excludes time spent waiting between retries.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(**request)
            latency = time.perf_counter() - start
            return {"response": response.model_dump(), "latency_s": latency, "attempts": attempt}
        except RETRYABLE as e:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = min(2 ** attempt, 60) + random.uniform(0, 1)  # 2s, 4s, 8s ... capped, plus jitter
            print(f"  retry {attempt}/{MAX_ATTEMPTS - 1} in {wait:.0f}s ({type(e).__name__})")
            time.sleep(wait)
