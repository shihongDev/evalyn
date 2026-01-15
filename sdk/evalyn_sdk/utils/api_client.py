"""Shared API clients for LLM services."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional


class GeminiClient:
    """HTTP client for Gemini API.

    Usage:
        client = GeminiClient(model="gemini-2.5-flash-lite")
        response = client.generate("What is 2+2?")
    """

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        """Initialize Gemini API client.

        Args:
            model: Gemini model name (default: gemini-2.5-flash-lite)
            temperature: Generation temperature (default: 0.0)
            api_key: Optional API key (default: from GEMINI_API_KEY env var)
            timeout: Request timeout in seconds (default: 60)
        """
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self.timeout = timeout

    def _get_api_key(self) -> str:
        """Get API key from instance or environment."""
        key = (
            self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set the environment variable or pass api_key."
            )
        return key

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Call Gemini API and return text response.

        Args:
            prompt: The prompt to send to the model
            temperature: Optional temperature override for this request

        Returns:
            The generated text response

        Raises:
            RuntimeError: If the API call fails
        """
        api_key = self._get_api_key()
        url = self.API_URL.format(model=self.model)

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature
                if temperature is not None
                else self.temperature
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Gemini API error ({e.code}): {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Gemini API connection error: {e.reason}") from e

        # Extract text from response
        try:
            candidates = response_data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        except Exception:
            pass

        return ""


def call_gemini_api(
    prompt: str,
    model: str = "gemini-2.5-flash-lite",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """Convenience function to call Gemini API.

    Args:
        prompt: The prompt to send to the model
        model: Gemini model name (default: gemini-2.5-flash-lite)
        api_key: Optional API key (default: from GEMINI_API_KEY env var)
        temperature: Generation temperature (default: 0.0)

    Returns:
        The generated text response
    """
    client = GeminiClient(model=model, api_key=api_key, temperature=temperature)
    return client.generate(prompt)


__all__ = ["GeminiClient", "call_gemini_api"]
