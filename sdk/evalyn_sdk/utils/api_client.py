"""Shared API clients for LLM services."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from ..defaults import DEFAULT_EVAL_MODEL


@dataclass
class GenerateResult:
    """Result from LLM generation with token usage info."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


def _http_post(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    error_prefix: str,
) -> dict[str, Any]:
    """Make HTTP POST request and return JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"{error_prefix} error ({e.code}): {error_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"{error_prefix} connection error: {e.reason}") from e


class GeminiClient:
    """HTTP client for Gemini API.

    Usage:
        client = GeminiClient()
        response = client.generate("What is 2+2?")
    """

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(
        self,
        model: str = DEFAULT_EVAL_MODEL,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        """Initialize Gemini API client.

        Args:
            model: Gemini model name (default: DEFAULT_EVAL_MODEL)
            temperature: Generation temperature (default: 0.0)
            api_key: Optional API key (default: from GEMINI_API_KEY env var)
            timeout: Request timeout in seconds (default: 60)
        """
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self.timeout = timeout

    def _get_api_key(self) -> str:
        """Get API key from instance, config, or environment."""
        if self._api_key:
            return self._api_key

        # Check YAML config
        try:
            from ..cli.utils.config import get_config_default, load_config

            config = load_config()
            config_key = get_config_default(config, "api_keys", "gemini")
            if config_key:
                return config_key
        except Exception:
            pass  # Config loading failed, fall back to env

        # Fall back to environment variables
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set in evalyn.yaml, environment variable, or pass api_key."
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
        result = self.generate_with_usage(prompt, temperature)
        return result.text

    def generate_with_usage(
        self, prompt: str, temperature: Optional[float] = None
    ) -> GenerateResult:
        """Call Gemini API and return text with token usage.

        Args:
            prompt: The prompt to send to the model
            temperature: Optional temperature override for this request

        Returns:
            GenerateResult with text and token counts
        """
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature
                if temperature is not None
                else self.temperature
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._get_api_key(),
        }
        url = self.API_URL.format(model=self.model)
        response_data = _http_post(url, payload, headers, self.timeout, "Gemini API")

        # Extract text from response
        text = ""
        candidates = response_data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text = parts[0].get("text", "")

        # Extract token usage from usageMetadata
        usage = response_data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)

        return GenerateResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )


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


class OpenAIClient:
    """HTTP client for OpenAI API with logprobs support.

    Usage:
        client = OpenAIClient(model="gpt-4o-mini")
        response = client.generate("What is 2+2?")

        # With confidence (logprobs)
        text, confidence = client.generate_with_confidence("What is 2+2?")
    """

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self.timeout = timeout

    def _get_api_key(self) -> str:
        """Get API key from instance, config, or environment."""
        if self._api_key:
            return self._api_key

        # Check YAML config
        try:
            from ..cli.utils.config import get_config_default, load_config

            config = load_config()
            config_key = get_config_default(config, "api_keys", "openai")
            if config_key:
                return config_key
        except Exception:
            pass  # Config loading failed, fall back to env

        # Fall back to environment variable
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Set in evalyn.yaml, environment variable, or pass api_key."
            )
        return key

    def _call_api(
        self, prompt: str, temperature: Optional[float], with_logprobs: bool = False
    ) -> dict[str, Any]:
        """Make API call and return raw response data."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if with_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_api_key()}",
        }
        return _http_post(self.API_URL, payload, headers, self.timeout, "OpenAI API")

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Call OpenAI API and return text response."""
        result = self.generate_with_usage(prompt, temperature)
        return result.text

    def generate_with_usage(
        self, prompt: str, temperature: Optional[float] = None
    ) -> GenerateResult:
        """Call OpenAI API and return text with token usage."""
        response_data = self._call_api(prompt, temperature)

        text = ""
        choices = response_data.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "")

        # Extract token usage
        usage = response_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return GenerateResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )

    def generate_with_confidence(
        self, prompt: str, temperature: Optional[float] = None
    ) -> tuple[str, float]:
        """Call OpenAI API with logprobs and return (text, confidence).

        Confidence is calculated as exp(mean(logprobs)) - higher is more confident.
        Returns value between 0.0 and 1.0.
        """
        response_data = self._call_api(prompt, temperature, with_logprobs=True)

        text = ""
        confidence = 0.5  # Default if no logprobs

        choices = response_data.get("choices", [])
        if choices:
            choice = choices[0]
            text = choice.get("message", {}).get("content", "")

            # Calculate confidence from logprobs
            content_logprobs = choice.get("logprobs", {}).get("content", [])
            if content_logprobs:
                logprobs_values = [lp.get("logprob", 0.0) for lp in content_logprobs]
                if logprobs_values:
                    mean_logprob = sum(logprobs_values) / len(logprobs_values)
                    confidence = max(0.0, min(1.0, math.exp(mean_logprob)))

        return text, confidence

    def generate_with_logprobs(
        self, prompt: str, temperature: Optional[float] = None
    ) -> tuple[str, list[float]]:
        """Call OpenAI API and return (text, raw_logprobs).

        Returns the raw token logprobs for use with confidence estimators
        like DeepConf that need custom aggregation strategies.
        """
        response_data = self._call_api(prompt, temperature, with_logprobs=True)

        text = ""
        logprobs_values = []

        choices = response_data.get("choices", [])
        if choices:
            choice = choices[0]
            text = choice.get("message", {}).get("content", "")

            content_logprobs = choice.get("logprobs", {}).get("content", [])
            if content_logprobs:
                logprobs_values = [lp.get("logprob", 0.0) for lp in content_logprobs]

        return text, logprobs_values


class OllamaClient:
    """HTTP client for Ollama API with logprobs support.

    Usage:
        client = OllamaClient(model="llama3.2")
        response = client.generate("What is 2+2?")

        # With confidence (logprobs)
        text, confidence = client.generate_with_confidence("What is 2+2?")
    """

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.0,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call_api(
        self,
        prompt: str,
        temperature: Optional[float],
        extra_options: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Make API call and return raw response data."""
        options = {
            "temperature": temperature if temperature is not None else self.temperature
        }
        if extra_options:
            options.update(extra_options)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        headers = {"Content-Type": "application/json"}
        url = f"{self.base_url}/api/generate"
        return _http_post(url, payload, headers, self.timeout, "Ollama API")

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Call Ollama API and return text response."""
        result = self.generate_with_usage(prompt, temperature)
        return result.text

    def generate_with_usage(
        self, prompt: str, temperature: Optional[float] = None
    ) -> GenerateResult:
        """Call Ollama API and return text with token usage."""
        response_data = self._call_api(prompt, temperature)

        text = response_data.get("response", "")

        # Extract token usage from Ollama response
        input_tokens = response_data.get("prompt_eval_count", 0)
        output_tokens = response_data.get("eval_count", 0)

        return GenerateResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )

    def generate_with_confidence(
        self, prompt: str, temperature: Optional[float] = None
    ) -> tuple[str, float]:
        """Call Ollama API with logprobs and return (text, confidence).

        Note: Ollama support for logprobs varies by model. Falls back to 0.5
        if logprobs not available.
        """
        response_data = self._call_api(
            prompt, temperature, extra_options={"num_predict": 512}
        )

        text = response_data.get("response", "")
        confidence = 0.5  # Default - Ollama logprobs support is limited

        # Heuristic: use generation speed as proxy for confidence
        # Ollama doesn't expose logprobs directly like OpenAI
        total_duration = response_data.get("total_duration", 0)
        eval_count = response_data.get("eval_count", 0)
        if eval_count > 0 and total_duration > 0:
            tokens_per_ns = eval_count / total_duration
            confidence = min(0.9, max(0.3, tokens_per_ns * 1e7))

        return text, confidence

    def generate_with_logprobs(
        self, prompt: str, temperature: Optional[float] = None
    ) -> tuple[str, list[float]]:
        """Call Ollama API and return (text, raw_logprobs).

        Note: Ollama doesn't expose token-level logprobs like OpenAI.
        This returns an empty list for logprobs. Use generate_with_confidence()
        for Ollama's heuristic-based confidence instead.
        """
        response_data = self._call_api(
            prompt, temperature, extra_options={"num_predict": 512}
        )
        text = response_data.get("response", "")
        # Ollama doesn't expose token-level logprobs
        return text, []


__all__ = [
    "GenerateResult",
    "GeminiClient",
    "OpenAIClient",
    "OllamaClient",
    "call_gemini_api",
]
