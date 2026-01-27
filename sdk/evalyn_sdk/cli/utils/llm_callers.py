"""LLM API caller utilities for CLI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from typing import Callable, List, Optional

from .config import get_config_default, load_config
from .ui import Spinner
from .loaders import _load_callable


def _parse_json_array(text: str) -> List[dict]:
    """Parse JSON array from text, handling malformed responses."""
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _ollama_caller(model: str) -> Callable[[str], List[dict]]:
    """Create an Ollama-based LLM caller."""

    def _call(prompt: str) -> List[dict]:
        if not shutil.which("ollama"):
            raise RuntimeError(
                "ollama CLI not found. Install Ollama or choose --llm-mode api."
            )
        proc = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=False,
        )
        text = proc.stdout.strip()
        try:
            return json.loads(text)
        except Exception:
            return []

    return _call


def _openai_caller(
    model: str, api_base: Optional[str] = None, api_key: Optional[str] = None
) -> Callable[[str], List[dict]]:
    """Create an OpenAI/Gemini-based LLM caller."""

    def _call(prompt: str) -> List[dict]:
        config = load_config()

        # Gemini shortcut using google-genai if model name starts with "gemini"
        if model.lower().startswith("gemini"):
            key = (
                api_key
                or get_config_default(config, "api_keys", "gemini")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("GEMINI_API_KEY")
            )
            if not key:
                raise RuntimeError(
                    "Missing Gemini API key. Set in evalyn.yaml or GEMINI_API_KEY env var."
                )
            try:
                from google.genai import Client  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "google-genai package not installed. Install with: pip install google-genai"
                ) from exc

            try:
                client = Client(api_key=key)
                guard = (
                    "Return ONLY a JSON array of metric objects. "
                    'Each object: {"id": "metric_id", "config": {...}}. '
                    "No prose. If unsure, return [].\n\n"
                )
                full_prompt = guard + prompt
                resp = client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config={"temperature": 0},
                )
                text = getattr(resp, "text", None) or ""
                parsed = _parse_json_array(text)
                if parsed:
                    return parsed
                raise RuntimeError(f"Gemini call returned non-JSON: {text[:200]}")
            except Exception as exc:
                raise RuntimeError(f"Gemini call failed: {exc}") from exc

        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                'openai package not installed. Install with extras: pip install -e "sdk[llm]"'
            ) from exc

        key = (
            api_key
            or get_config_default(config, "api_keys", "openai")
            or os.getenv("OPENAI_API_KEY")
        )
        client = (
            openai.OpenAI(api_key=key, base_url=api_base)
            if (key or api_base)
            else openai.OpenAI()
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        parsed = _parse_json_array(text)
        return parsed

    return _call


def _with_spinner(
    caller: Callable[[str], List[dict]], message: str = "Calling LLM"
) -> Callable[[str], List[dict]]:
    """Wrap a caller function with a spinner for visual feedback."""

    def _wrapped(prompt: str) -> List[dict]:
        with Spinner(message):
            return caller(prompt)

    return _wrapped


def _build_llm_caller(args: argparse.Namespace) -> Callable[[str], List[dict]]:
    """Build an LLM caller based on CLI arguments."""
    model_name = getattr(args, "model", "LLM")
    spinner_msg = f"Querying {model_name}"
    if args.llm_mode == "local":
        return _with_spinner(_ollama_caller(args.model), spinner_msg)
    if args.llm_caller:
        return _with_spinner(_load_callable(args.llm_caller), spinner_msg)
    return _with_spinner(
        _openai_caller(args.model, api_base=args.api_base, api_key=args.api_key),
        spinner_msg,
    )


__all__ = [
    "_parse_json_array",
    "_ollama_caller",
    "_openai_caller",
    "_with_spinner",
    "_build_llm_caller",
]
