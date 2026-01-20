"""Batch API providers for LLM evaluation.

Implements batch processing for:
- Gemini (Google AI / Vertex AI)
- OpenAI
- Anthropic

All providers offer ~50% cost savings and higher rate limits compared to real-time APIs.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BatchJob:
    """Represents a submitted batch job."""

    id: str
    provider: str
    status: str  # pending, processing, completed, failed, cancelled
    created_at: datetime
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return self.status in ("completed", "failed", "cancelled", "expired")

    def is_success(self) -> bool:
        return self.status == "completed"


@dataclass
class BatchResult:
    """Result from a single request in a batch."""

    custom_id: str
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Provider
# =============================================================================


class BatchProvider(ABC):
    """Abstract base class for batch API providers."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        self.model = model
        self._api_key = api_key
        self.timeout = timeout

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (gemini, openai, anthropic)."""
        pass

    @abstractmethod
    def _get_api_key(self) -> str:
        """Get API key from instance or environment."""
        pass

    @abstractmethod
    def submit(
        self,
        requests: List[Dict[str, Any]],
        description: Optional[str] = None,
    ) -> BatchJob:
        """Submit a batch of requests.

        Args:
            requests: List of dicts with 'custom_id' and 'prompt' keys
            description: Optional description for the batch

        Returns:
            BatchJob with job ID and initial status
        """
        pass

    @abstractmethod
    def get_status(self, job_id: str) -> BatchJob:
        """Get current status of a batch job."""
        pass

    @abstractmethod
    def get_results(self, job_id: str) -> List[BatchResult]:
        """Get results from a completed batch job."""
        pass

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a running batch job."""
        pass

    def wait(
        self,
        job_id: str,
        poll_interval: float = 30.0,
        timeout: Optional[float] = None,
        progress_callback: Optional[callable] = None,
    ) -> BatchJob:
        """Wait for a batch job to complete.

        Args:
            job_id: The batch job ID
            poll_interval: Seconds between status checks (default: 30)
            timeout: Maximum seconds to wait (default: None = no limit)
            progress_callback: Optional callback(job) for progress updates

        Returns:
            Final BatchJob status
        """
        start_time = time.time()

        while True:
            job = self.get_status(job_id)

            if progress_callback:
                progress_callback(job)

            if job.is_complete():
                return job

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Batch job {job_id} did not complete within {timeout}s")

            time.sleep(poll_interval)


# =============================================================================
# Gemini Batch Provider
# =============================================================================


class GeminiBatchProvider(BatchProvider):
    """Batch provider for Google Gemini API.

    Uses the Gemini API batch mode for 50% cost savings.
    Requires GEMINI_API_KEY environment variable.

    Note: For production use with large batches, consider using Vertex AI
    batch prediction which supports Cloud Storage input/output.
    """

    # Gemini batch API endpoint (AI Studio)
    BATCH_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchGenerateContent"

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        super().__init__(model=model, api_key=api_key, timeout=timeout)
        self._pending_jobs: Dict[str, Dict] = {}  # In-memory job tracking

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_api_key(self) -> str:
        key = self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Missing GEMINI_API_KEY environment variable")
        return key

    def submit(
        self,
        requests: List[Dict[str, Any]],
        description: Optional[str] = None,
    ) -> BatchJob:
        """Submit batch using Gemini's batchGenerateContent API.

        Note: Gemini's batch API processes synchronously but with higher throughput.
        For true async batch processing, use Vertex AI.
        """
        api_key = self._get_api_key()
        url = self.BATCH_URL.format(model=self.model)

        # Build batch request payload
        batch_requests = []
        for req in requests:
            batch_requests.append({
                "contents": [{"parts": [{"text": req["prompt"]}]}],
                "generationConfig": {"temperature": 0.0},
            })

        payload = {"requests": batch_requests}
        data = json.dumps(payload).encode("utf-8")

        http_req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )

        job_id = f"gemini-batch-{int(time.time() * 1000)}"

        try:
            with urllib.request.urlopen(http_req, timeout=self.timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))

            # Store results for later retrieval
            self._pending_jobs[job_id] = {
                "requests": requests,
                "response": response_data,
                "created_at": datetime.now(),
            }

            return BatchJob(
                id=job_id,
                provider=self.provider_name,
                status="completed",  # Gemini batch is synchronous
                created_at=datetime.now(),
                total_requests=len(requests),
                completed_requests=len(requests),
            )

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Gemini batch API error ({e.code}): {error_body}") from e

    def get_status(self, job_id: str) -> BatchJob:
        """Get job status. Gemini batch is synchronous so always complete."""
        if job_id not in self._pending_jobs:
            raise ValueError(f"Unknown job ID: {job_id}")

        job_data = self._pending_jobs[job_id]
        return BatchJob(
            id=job_id,
            provider=self.provider_name,
            status="completed",
            created_at=job_data["created_at"],
            total_requests=len(job_data["requests"]),
            completed_requests=len(job_data["requests"]),
        )

    def get_results(self, job_id: str) -> List[BatchResult]:
        """Get results from completed batch."""
        if job_id not in self._pending_jobs:
            raise ValueError(f"Unknown job ID: {job_id}")

        job_data = self._pending_jobs[job_id]
        requests = job_data["requests"]
        response = job_data["response"]

        results = []
        responses = response.get("responses", [])

        for i, req in enumerate(requests):
            custom_id = req["custom_id"]

            if i < len(responses):
                resp = responses[i]
                # Extract text from response
                try:
                    candidates = resp.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                            results.append(BatchResult(
                                custom_id=custom_id,
                                success=True,
                                response=text,
                            ))
                            continue
                except Exception as e:
                    logger.warning(f"Failed to parse response for {custom_id}: {e}")

                # Check for error
                error = resp.get("error", {}).get("message", "Unknown error")
                results.append(BatchResult(
                    custom_id=custom_id,
                    success=False,
                    error=error,
                ))
            else:
                results.append(BatchResult(
                    custom_id=custom_id,
                    success=False,
                    error="No response received",
                ))

        return results

    def cancel(self, job_id: str) -> bool:
        """Cancel not supported for synchronous Gemini batch."""
        return False


# =============================================================================
# OpenAI Batch Provider
# =============================================================================


class OpenAIBatchProvider(BatchProvider):
    """Batch provider for OpenAI API.

    Uses OpenAI's Batch API for 50% cost savings and 24-hour turnaround.
    Requires OPENAI_API_KEY environment variable.
    """

    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__(model=model, api_key=api_key, timeout=timeout)

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_api_key(self) -> str:
        key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable")
        return key

    def _api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[bytes] = None,
        headers: Optional[Dict] = None,
    ) -> Dict:
        """Make API request to OpenAI."""
        api_key = self._get_api_key()
        url = f"{self.BASE_URL}/{endpoint}"

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if headers:
            default_headers.update(headers)

        req = urllib.request.Request(
            url,
            data=data,
            headers=default_headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"OpenAI API error ({e.code}): {error_body}") from e

    def submit(
        self,
        requests: List[Dict[str, Any]],
        description: Optional[str] = None,
    ) -> BatchJob:
        """Submit batch to OpenAI Batch API."""
        # Step 1: Create JSONL content
        lines = []
        for req in requests:
            line = {
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": req["prompt"]}],
                    "temperature": 0.0,
                },
            }
            lines.append(json.dumps(line))
        jsonl_content = "\n".join(lines)

        # Step 2: Upload file
        api_key = self._get_api_key()
        boundary = f"----BatchBoundary{int(time.time() * 1000)}"

        body_parts = [
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n',
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="file"; filename="batch.jsonl"\r\n'
            f'Content-Type: application/jsonl\r\n\r\n{jsonl_content}\r\n',
            f'--{boundary}--\r\n',
        ]
        body = "".join(body_parts).encode("utf-8")

        file_req = urllib.request.Request(
            f"{self.BASE_URL}/files",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(file_req, timeout=self.timeout) as resp:
                file_response = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"OpenAI file upload error ({e.code}): {error_body}") from e

        file_id = file_response["id"]

        # Step 3: Create batch
        batch_payload = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
        if description:
            batch_payload["metadata"] = {"description": description}

        batch_response = self._api_request(
            "POST",
            "batches",
            json.dumps(batch_payload).encode("utf-8"),
        )

        return BatchJob(
            id=batch_response["id"],
            provider=self.provider_name,
            status=self._map_status(batch_response["status"]),
            created_at=datetime.fromtimestamp(batch_response["created_at"]),
            total_requests=len(requests),
            metadata={"file_id": file_id},
        )

    def _map_status(self, openai_status: str) -> str:
        """Map OpenAI status to our standard status."""
        mapping = {
            "validating": "pending",
            "in_progress": "processing",
            "finalizing": "processing",
            "completed": "completed",
            "failed": "failed",
            "expired": "failed",
            "cancelling": "processing",
            "cancelled": "cancelled",
        }
        return mapping.get(openai_status, "pending")

    def get_status(self, job_id: str) -> BatchJob:
        """Get batch job status."""
        response = self._api_request("GET", f"batches/{job_id}")

        counts = response.get("request_counts", {})

        return BatchJob(
            id=job_id,
            provider=self.provider_name,
            status=self._map_status(response["status"]),
            created_at=datetime.fromtimestamp(response["created_at"]),
            total_requests=counts.get("total", 0),
            completed_requests=counts.get("completed", 0),
            failed_requests=counts.get("failed", 0),
            metadata={
                "output_file_id": response.get("output_file_id"),
                "error_file_id": response.get("error_file_id"),
            },
        )

    def get_results(self, job_id: str) -> List[BatchResult]:
        """Get results from completed batch."""
        job = self.get_status(job_id)

        if not job.is_complete():
            raise RuntimeError(f"Batch job {job_id} is not complete (status: {job.status})")

        output_file_id = job.metadata.get("output_file_id")
        if not output_file_id:
            raise RuntimeError(f"No output file for batch job {job_id}")

        # Download output file
        api_key = self._get_api_key()
        req = urllib.request.Request(
            f"{self.BASE_URL}/files/{output_file_id}/content",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Failed to download results ({e.code}): {error_body}") from e

        # Parse JSONL results
        results = []
        for line in content.strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            custom_id = data["custom_id"]

            if data.get("error"):
                results.append(BatchResult(
                    custom_id=custom_id,
                    success=False,
                    error=data["error"].get("message", "Unknown error"),
                ))
            else:
                response_body = data.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    results.append(BatchResult(
                        custom_id=custom_id,
                        success=True,
                        response=text,
                    ))
                else:
                    results.append(BatchResult(
                        custom_id=custom_id,
                        success=False,
                        error="No response content",
                    ))

        return results

    def cancel(self, job_id: str) -> bool:
        """Cancel a running batch job."""
        try:
            self._api_request("POST", f"batches/{job_id}/cancel")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel batch {job_id}: {e}")
            return False


# =============================================================================
# Anthropic Batch Provider
# =============================================================================


class AnthropicBatchProvider(BatchProvider):
    """Batch provider for Anthropic Claude API.

    Uses Anthropic's Message Batches API for 50% cost savings.
    Requires ANTHROPIC_API_KEY environment variable.
    """

    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(
        self,
        model: str = "claude-3-5-haiku-latest",
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__(model=model, api_key=api_key, timeout=timeout)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_api_key(self) -> str:
        key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY environment variable")
        return key

    def _api_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[bytes] = None,
    ) -> Dict:
        """Make API request to Anthropic."""
        api_key = self._get_api_key()
        url = f"{self.BASE_URL}/{endpoint}"

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "message-batches-2024-09-24",
            "Content-Type": "application/json",
        }

        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Anthropic API error ({e.code}): {error_body}") from e

    def submit(
        self,
        requests: List[Dict[str, Any]],
        description: Optional[str] = None,
    ) -> BatchJob:
        """Submit batch to Anthropic Message Batches API."""
        batch_requests = []
        for req in requests:
            batch_requests.append({
                "custom_id": req["custom_id"],
                "params": {
                    "model": self.model,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": req["prompt"]}],
                },
            })

        payload = {"requests": batch_requests}
        response = self._api_request(
            "POST",
            "messages/batches",
            json.dumps(payload).encode("utf-8"),
        )

        counts = response.get("request_counts", {})

        return BatchJob(
            id=response["id"],
            provider=self.provider_name,
            status=self._map_status(response["processing_status"]),
            created_at=datetime.fromisoformat(response["created_at"].replace("Z", "+00:00")),
            total_requests=counts.get("processing", 0) + counts.get("succeeded", 0) + counts.get("errored", 0),
        )

    def _map_status(self, anthropic_status: str) -> str:
        """Map Anthropic status to our standard status."""
        mapping = {
            "in_progress": "processing",
            "ended": "completed",
            "canceling": "processing",
        }
        return mapping.get(anthropic_status, "pending")

    def get_status(self, job_id: str) -> BatchJob:
        """Get batch job status."""
        response = self._api_request("GET", f"messages/batches/{job_id}")

        counts = response.get("request_counts", {})
        total = counts.get("processing", 0) + counts.get("succeeded", 0) + counts.get("errored", 0) + counts.get("canceled", 0) + counts.get("expired", 0)

        return BatchJob(
            id=job_id,
            provider=self.provider_name,
            status=self._map_status(response["processing_status"]),
            created_at=datetime.fromisoformat(response["created_at"].replace("Z", "+00:00")),
            total_requests=total,
            completed_requests=counts.get("succeeded", 0),
            failed_requests=counts.get("errored", 0),
            metadata={"results_url": response.get("results_url")},
        )

    def get_results(self, job_id: str) -> List[BatchResult]:
        """Get results from completed batch."""
        job = self.get_status(job_id)

        if not job.is_complete():
            raise RuntimeError(f"Batch job {job_id} is not complete (status: {job.status})")

        results_url = job.metadata.get("results_url")
        if not results_url:
            raise RuntimeError(f"No results URL for batch job {job_id}")

        # Download results
        api_key = self._get_api_key()
        req = urllib.request.Request(
            results_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "message-batches-2024-09-24",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Failed to download results ({e.code}): {error_body}") from e

        # Parse JSONL results
        results = []
        for line in content.strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            custom_id = data["custom_id"]
            result_data = data.get("result", {})

            if result_data.get("type") == "succeeded":
                message = result_data.get("message", {})
                content_blocks = message.get("content", [])
                text = ""
                for block in content_blocks:
                    if block.get("type") == "text":
                        text += block.get("text", "")
                results.append(BatchResult(
                    custom_id=custom_id,
                    success=True,
                    response=text,
                ))
            else:
                error = result_data.get("error", {})
                results.append(BatchResult(
                    custom_id=custom_id,
                    success=False,
                    error=error.get("message", "Unknown error"),
                ))

        return results

    def cancel(self, job_id: str) -> bool:
        """Cancel a running batch job."""
        try:
            self._api_request("POST", f"messages/batches/{job_id}/cancel")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel batch {job_id}: {e}")
            return False


# =============================================================================
# Factory Function
# =============================================================================


def create_batch_provider(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BatchProvider:
    """Create a batch provider instance.

    Args:
        provider: Provider name (gemini, openai, anthropic)
        model: Optional model override
        api_key: Optional API key override

    Returns:
        Configured BatchProvider instance
    """
    provider = provider.lower()

    if provider == "gemini":
        return GeminiBatchProvider(
            model=model or "gemini-2.5-flash-lite",
            api_key=api_key,
        )
    elif provider == "openai":
        return OpenAIBatchProvider(
            model=model or "gpt-4o-mini",
            api_key=api_key,
        )
    elif provider == "anthropic":
        return AnthropicBatchProvider(
            model=model or "claude-3-5-haiku-latest",
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: gemini, openai, anthropic")
