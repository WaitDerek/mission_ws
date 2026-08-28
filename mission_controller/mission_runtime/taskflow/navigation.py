"""Navigation gateway protocol plus disabled and deterministic test adapters."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterable
from typing import Protocol

from .model import NavigationRequest, NavigationResult


class NavigationGateway(Protocol):
    def navigate(
        self,
        request: NavigationRequest,
        cancel_requested: Callable[[], bool],
    ) -> NavigationResult: ...

    def cancel_active(self) -> None: ...

    def close(self) -> None: ...


class DisabledNavigationGateway:
    """Safe default: never pretends that the platform moved."""

    def navigate(
        self,
        request: NavigationRequest,
        cancel_requested: Callable[[], bool],
    ) -> NavigationResult:
        del cancel_requested
        return NavigationResult(
            False,
            "unavailable",
            f"navigation adapter is disabled; point {request.point_id} was not sent",
        )

    def cancel_active(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeNavigationGateway:
    """Deterministic test adapter with optional blocking responses."""

    def __init__(
        self,
        results: Iterable[NavigationResult] = (),
        *,
        block: bool = False,
    ) -> None:
        self._results = deque(results)
        self._block = bool(block)
        self._release = threading.Event()
        self._canceled = threading.Event()
        self.requests: list[NavigationRequest] = []

    def release(self) -> None:
        self._release.set()

    def navigate(
        self,
        request: NavigationRequest,
        cancel_requested: Callable[[], bool],
    ) -> NavigationResult:
        self.requests.append(request)
        while self._block and not self._release.wait(0.01):
            if self._canceled.is_set() or cancel_requested():
                return NavigationResult(False, "canceled", "navigation canceled")
        if self._canceled.is_set() or cancel_requested():
            return NavigationResult(False, "canceled", "navigation canceled")
        if self._results:
            return self._results.popleft()
        return NavigationResult(True, "succeeded", "fake navigation succeeded")

    def cancel_active(self) -> None:
        self._canceled.set()
        self._release.set()

    def close(self) -> None:
        self.cancel_active()
