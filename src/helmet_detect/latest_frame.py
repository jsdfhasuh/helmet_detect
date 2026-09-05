"""One-slot, latest-only transport. Decoder queues/network delay remain external."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Condition
from typing import Any


@dataclass(frozen=True, slots=True)
class FramePacket:
    sequence: int
    image: Any
    received_at: float
    source_seconds: float | None = None


class LatestFrameSlot:
    """A producer replaces unread frames instead of building a latency queue.

    Images must not be mutated after publication. Each packet can be consumed
    once. Closing wakes waiters and still allows the final unread packet.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._packet: FramePacket | None = None
        self._consumed = -1
        self._sequence = 0
        self._closed = False
        self.dropped = 0
        self.error: str | None = None

    def publish(self, image: Any, *, source_seconds: float | None = None) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot publish to a closed frame slot")
            if self._packet is not None and self._packet.sequence > self._consumed:
                self.dropped += 1
            self._packet = FramePacket(self._sequence, image, time.monotonic(), source_seconds)
            self._sequence += 1
            self._condition.notify_all()

    def take(self, timeout: float = 1.0) -> FramePacket | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._packet is None or self._packet.sequence <= self._consumed:
                if self._closed:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            self._consumed = self._packet.sequence
            return self._packet

    def close(self, error: str | None = None) -> None:
        with self._condition:
            self._closed = True
            self.error = error
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed
