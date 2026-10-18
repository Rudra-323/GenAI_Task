"""Small LLM helper shared by Problem 1 and Problem 2.

One OpenAI-compatible client works for OpenAI, Gemini, Groq and OpenRouter,
so switching model/provider is only a change in .env (no code change).

Settings are read from environment variables with a prefix:
    GEN_PROVIDER, GEN_MODEL, ...      -> the model that writes answers
    JUDGE_PROVIDER, JUDGE_MODEL, ...  -> the model that grades answers
"""
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                    OpenAI, RateLimitError)

load_dotenv()

# provider -> (base_url, name of the env var that holds the API key)
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


class LLM:
    """A chat model configured from env vars like <PREFIX>_PROVIDER / <PREFIX>_MODEL."""

    def __init__(self, prefix, provider=None, model=None, fallback_prefix=None):
        def get(name):
            value = os.getenv(f"{prefix}_{name}")
            if not value and fallback_prefix:
                value = os.getenv(f"{fallback_prefix}_{name}")
            return value

        self.provider = (provider or get("PROVIDER") or "").lower()
        self.model = model or get("MODEL")
        if self.provider not in PROVIDERS:
            raise ValueError(f"{prefix}_PROVIDER must be one of {list(PROVIDERS)} (got '{self.provider}'). Check your .env")
        if not self.model:
            raise ValueError(f"{prefix}_MODEL is not set. Check your .env")

        base_url, key_name = PROVIDERS[self.provider]
        api_key = os.getenv(key_name)
        if not api_key:
            raise ValueError(f"{key_name} is not set. Add it to your .env file")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
        self.temperature = float(get("TEMPERATURE") or 0)
        self.max_tokens = int(get("MAX_TOKENS") or 1500)
        self.price_in = float(get("PRICE_IN_PER_M") or 0)    # USD per 1M input tokens
        self.price_out = float(get("PRICE_OUT_PER_M") or 0)  # USD per 1M output tokens

    def chat(self, messages, temperature=None, max_tokens=None):
        """Send messages, return {text, prompt_tokens, completion_tokens, latency_ms}.
        Retries with a growing wait when the provider says "slow down" (rate limit)."""
        wait = 4
        for attempt in range(6):
            start = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                break
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as err:
                if attempt == 5:
                    raise
                print(f"  [{self.provider}] {type(err).__name__}, retrying in {wait}s ...")
                time.sleep(wait)
                wait *= 2
        usage = resp.usage
        return {
            "text": resp.choices[0].message.content or "",
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    def cost(self, prompt_tokens, completion_tokens):
        """Estimated USD cost (0 if prices are not set in .env)."""
        return (prompt_tokens * self.price_in + completion_tokens * self.price_out) / 1_000_000


# ---------------------------------------------------------------------------
# Robust JSON parsing: LLMs often wrap JSON in ```fences```, add chatter,
# leave trailing commas, or (reasoning models) add <think> blocks.
# ---------------------------------------------------------------------------
def _first_json_object(text):
    """Return the first balanced {...} block in text (string-aware), or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json(text):
    """Parse a JSON object out of LLM text. Raises ValueError if impossible."""
    if not text or not text.strip():
        raise ValueError("empty response")
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    for candidate in (cleaned, _first_json_object(cleaned)):
        if not candidate:
            continue
        no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", candidate)
        for attempt in (candidate, no_trailing_commas):
            try:
                obj = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("could not parse a JSON object from the response")


def chat_json(llm, messages, check=None, retries=1, temperature=None):
    """Call the LLM and parse a JSON answer.

    If the reply is not valid JSON (or `check` raises), ask the model once more,
    showing it its own bad reply. Returns (obj_or_None, calls) where `calls` is the
    list of every raw call made (so callers can log prompts and raw responses).
    """
    calls = []
    current = list(messages)
    for _ in range(retries + 1):
        out = llm.chat(current, temperature=temperature)
        out["messages"] = current
        try:
            obj = extract_json(out["text"])
            if check:
                check(obj)
            out["error"] = None
            calls.append(out)
            return obj, calls
        except (ValueError, KeyError, TypeError) as err:
            out["error"] = f"{type(err).__name__}: {err}"
            calls.append(out)
            current = list(messages) + [
                {"role": "assistant", "content": out["text"] or "(empty)"},
                {"role": "user", "content": f"That reply was invalid ({err}). Reply again with ONLY the JSON "
                                            "object in the exact format requested, and no other text."},
            ]
    return None, calls
