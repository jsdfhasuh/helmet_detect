# Early-detection repair: code fixes, NOT a production acceptance

## Scope and result

This repair addresses observed software defects without pretending that software
changes fix a weak model. The **early-entry camera acceptance still FAILS**. The
single-context experimental profile must not replace the deployed profile.
No input footage, image pixels or personal identifying details are committed here.

## Fixed mechanisms

- Run one raw scene prediction and attach ByteTrack IDs by its source-index output.
  New/unconfirmed detections are not discarded just because another object has a
  confirmed track. Unsupported tracker types fail clearly in this mode.
- Reset the scene tracker as well as local histories between video sessions.
- Reserve primary IDs before assigning fallback IDs; input order cannot assign
  the same ID to two objects in one frame.
- Enforce `minimum_vehicle_hits` on the current matched frame (previous code
  bypassed this setting).
- Optional bounded early head checks while vehicle evidence is still pending.
  These observations **cannot emit an event** until rider and alarm-zone
  eligibility hold. There is no ROI or input timestamp shortcut.
- Give observations a 1.5-second TTL. Do not count cached results as fresh frames.
  A current HELMET observation vetoes an alert from cached NO_HELMET evidence.
- Do not consume an event outside the alarm zone and suppress its later entry.
- Merge detections across context crops; assign each physical head to at most one
  plausible person, and leave ambiguous ownership unresolved.
- In the experimental profile, close HELMET/NO_HELMET scores are UNKNOWN, and a
  higher HELMET score cannot lose through the old 0.8 ratio.
- Distinguish not evaluated, rider pending, low-score candidate and alarm in the
  display. Low-score candidate boxes are not drawn as confirmed violations.
- Add per-frame scene/helmet/pipeline times, crop counts and actual model devices.
  Add measured processing FPS (not the nominal sampling/output video FPS).
- Add `--no-video` to omit continuous rendering/encoding while retaining JSONL
  and event snapshots. Video handles are released in a `finally` block.

## Reproduction

Default models are unchanged: official YOLO11n + iam-tsr YOLOv8n. This is not a
newly trained or calibrated helmet model. Larger scene candidates were also
probed during diagnosis, but are not included in the default pipeline because
they did not establish reliable early-entry recognition.

```powershell
python -m pip install -e ".[dynamic,dev]"
python scripts\prepare_dynamic_models.py
python -m pytest -q
python scripts\run_dynamic_regression.py

# EXPERIMENTAL only; not an accepted deployment configuration.
python -m helmet_detect.cli video `
  --config configs\dynamic_early_check.json `
  --source "D:\videos\camera.mp4" `
  --output artifacts\early\annotated.mp4 `
  --records artifacts\early\frames.jsonl `
  --summary artifacts\early\summary.json `
  --sample-fps 5 --cpu-threads 4

python scripts\report_latency.py `
  --records artifacts\early\frames.jsonl --output artifacts\early\timing
```

For a compute-only comparison, add `--no-video`. `--output` still supplies the
stem for the event-snapshot directory, but no continuous MP4 is then written.
This is **not** a new RTSP reader: latest-frame capture, network/camera frame age,
reconnection and asynchronous event transport remain separate work.

## Same-pixel A/B experiment

Both versions decoded the same original HEVC video without re-encoding. They
sampled frames on the same 5 FPS grid in three windows: 52–88, 122–133, 168–177
seconds. The pipeline was reset at each window start. This is 280 sampled frames,
not a fresh whole-video run, and not an independent held-out accuracy benchmark.
Runtime: CPU only, Torch 2.10.0, Ultralytics 8.4.79, four Torch threads. The values
below time `detect_frame`, exclude display/encoding, and include startup effects.

| Metric | V3 baseline | Experimental repair |
|---|---:|---:|
| Mean processing time | 316.79 ms | 241.13 ms |
| P95 processing time | 608.23 ms | 446.51 ms |
| Emitted events in these windows | 3 | 1 |
| Early bare-head entry at ~57 s | FAIL | FAIL |

The mean compute reduction is approximately 24%, **not a ninefold speedup** and
not proof of real-time operation. The one-crop policy lost the old valid 85.6 s
observation. Fewer alarms are not automatically better: they can be missed events.
The remaining event was at 175.2 s; this alone is not sufficient acceptance.

The person entering around 125 s has visible head covering/shell/visor features.
The previous late NO_HELMET alarm must not be treated as a positive ground-truth
label merely because the software raised it. Independently review helmet status
and riding status before defining that case's expected alarm.

## Acceptance gate before production

Need separately labelled earliest usable head frames, rider status, target boxes
and alarm deadlines, including real helmet-wearing negatives. Check raw person
recall, correct head ownership, first valid head observation, first rider evidence
and first correct event. Report a missed deadline as a failure even when a later
frame eventually triggers. Do not lower thresholds to make this one clip green.
Passing the existing small real-model CI regression proves API/model compatibility,
not this camera's early-detection accuracy or capture-to-alert latency.
