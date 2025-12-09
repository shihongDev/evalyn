from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from ..models import Annotation, EvalRun, FunctionCall


class StorageBackend(ABC):
    """Abstract interface for persisting traces, eval runs, and annotations."""

    @abstractmethod
    def store_call(self, call: FunctionCall) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_calls(self, limit: int = 100) -> List[FunctionCall]:
        raise NotImplementedError

    @abstractmethod
    def store_eval_run(self, run: EvalRun) -> None:
        raise NotImplementedError

    @abstractmethod
    def store_annotations(self, annotations: Iterable[Annotation]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Optional teardown hook."""
        return None
