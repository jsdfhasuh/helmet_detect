"""Bounded scheduling for head checks while rider evidence is still pending."""
from __future__ import annotations

from dataclasses import dataclass

from .dynamic_config import EarlyCheckConfig


@dataclass(slots=True)
class _Probe:
    first_seen: float
    last_seen: float
    last_check: float = float("-inf")


class EarlyCheckScheduler:
    def __init__(self, config: EarlyCheckConfig, *, maximum_age: float = 4.0) -> None:
        self.config = config
        self.maximum_age = maximum_age
        self._probes: dict[int, _Probe] = {}

    def select(self, people: list[tuple[int, bool]], now: float) -> set[int]:
        self._probes = {
            key: value for key, value in self._probes.items()
            if now - value.last_seen <= self.maximum_age
        }
        selected = {track for track, eligible in people if eligible}
        pending: list[tuple[float, int]] = []
        for track, eligible in people:
            probe = self._probes.setdefault(track, _Probe(now, now))
            probe.last_seen = now
            if eligible:
                probe.last_check = now
            elif (
                self.config.enabled
                and now - probe.first_seen <= self.config.probe_seconds
                and now - probe.last_check + 1e-9 >= self.config.interval_seconds
            ):
                pending.append((probe.last_check, track))
        for _, track in sorted(pending)[:self.config.maximum_pending_people]:
            selected.add(track)
            self._probes[track].last_check = now
        return selected

    def reset(self) -> None:
        self._probes.clear()
