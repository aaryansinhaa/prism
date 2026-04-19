"""Interface for model query cache backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class QueryCacheBackend(ABC):
    @abstractmethod
    def make_model_key(self, model_id: str, version: str | None) -> str:
        raise NotImplementedError

    @abstractmethod
    def lookup(self, model_key: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def maybe_store(
        self,
        model_key: str,
        payload: Dict[str, Any],
        response: Dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError
