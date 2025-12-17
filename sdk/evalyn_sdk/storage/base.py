from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Dict, Any

from ..models import Annotation, EvalRun, FunctionCall


class StorageBackend(ABC):
    """Abstract interface for persisting traces, eval runs, and annotations."""

    @abstractmethod
    def store_call(self, call: FunctionCall) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_call(self, call_id: str) -> Optional[FunctionCall]:
        raise NotImplementedError

    @abstractmethod
    def list_calls(self, limit: int = 100) -> List[FunctionCall]:
        raise NotImplementedError

    @abstractmethod
    def store_eval_run(self, run: EvalRun) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_eval_runs(self, limit: int = 20) -> List[EvalRun]:
        raise NotImplementedError

    @abstractmethod
    def get_eval_run(self, run_id: str) -> Optional[EvalRun]:
        raise NotImplementedError

    @abstractmethod
    def store_annotations(self, annotations: Iterable[Annotation]) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_annotations(self, target_id: Optional[str] = None, limit: int = 100) -> List[Annotation]:
        raise NotImplementedError

    def list_spans(self, call_id: str) -> List[Dict[str, Any]]:
        """Optional hook for span-backed views (OTel)."""
        return []

    def close(self) -> None:
        """Optional teardown hook."""
        return None
