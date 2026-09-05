#!/usr/bin/env python3
"""Summarize measured processing times and per-canonical-track stage delays.

This report does not infer ground truth from an existing alarm. Track IDs are
algorithmic identities, not a count of unique real people. Video timestamps are
not wall-clock RTSP frame age.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


def summarize(records_path: Path) -> tuple[dict, list[dict]]:
    timings: list[float] = []
    tracks: dict[int, dict] = {}
    frame_count = 0
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            frame_count += 1
            t = float(frame["timestamp_seconds"])
            ms = frame.get("processing_ms", frame.get("diagnostics", {}).get("pipeline_ms"))
            if ms is not None and math.isfinite(float(ms)) and float(ms) >= 0:
                timings.append(float(ms))
            for person in frame.get("persons", []):
                key = int(person["track_id"])
                track = tracks.setdefault(key, {
                    "track_id": key, "first_seen": t,
                    "first_helmet_check": None, "first_valid_head": None,
                    "first_rider_evidence": None, "first_event": None,
                    "maximum_no_helmet_score": 0.0,
                })
                conditions = {
                    "first_helmet_check": person.get("helmet_evaluated", False),
                    "first_valid_head": person.get("state") in {"helmet", "no_helmet"},
                    "first_rider_evidence": person.get("rider_eligible", False),
                    "first_event": person.get("event_triggered", False),
                }
                for field, condition in conditions.items():
                    if condition and track[field] is None:
                        track[field] = t
                track["maximum_no_helmet_score"] = max(
                    track["maximum_no_helmet_score"], person.get("no_helmet_score", 0.0)
                )
    ordered = sorted(timings)
    summary = {
        "frames": frame_count, "timed_frames": len(timings),
        "mean_processing_ms": mean(timings) if timings else None,
        "median_processing_ms": median(timings) if timings else None,
        "p95_processing_ms": ordered[math.ceil(.95 * len(ordered)) - 1] if ordered else None,
        "canonical_track_count": len(tracks),
        "acceptance": "MANUAL_GROUND_TRUTH_REQUIRED",
        "note": "Processing time excludes RTSP capture-to-alert latency; no accuracy claim.",
    }
    return summary, list(tracks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, tracks = summarize(args.records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "latency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if tracks:
        with (args.output / "track_stages.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(tracks[0]))
            writer.writeheader()
            writer.writerows(tracks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
