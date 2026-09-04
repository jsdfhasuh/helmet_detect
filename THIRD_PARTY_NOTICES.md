# Third-party model notices

本仓库不提交第三方模型权重。`scripts/prepare_dynamic_models.py` 会在用户本地或 GitHub Actions 中下载权重，并在使用前检查固定 SHA256。

## 1. Ultralytics official YOLO11n scene model

- Project: Ultralytics YOLO
- Downloaded file: `yolo11n.pt`
- Local name: `models/yolo11n.pt`
- SHA256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
- Role: full-frame COCO `person`, `bicycle`, and `motorcycle` detection
- Tracking: ByteTrack through the Ultralytics tracking interface

The Ultralytics software, pretrained weights, and commercial deployment terms must be reviewed independently. This repository's MIT license applies only to code authored for this repository and does not relicense Ultralytics software or weights.

## 2. YOLOv8n helmet-state model

- Model page: https://huggingface.co/iam-tsr/yolov8n-helmet-detection
- Downloaded file: `best.pt`
- Local name: `models/helmet_yolov8n.pt`
- SHA256: `c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d`
- Classes observed in the weight: `With Helmet`, `Without Helmet`
- Role: per-person dynamic-context helmet state detection
- Model-page license marker at integration time: MIT

## 3. Optional YOLO11m rider helmet model

- Model page: https://huggingface.co/nnsohamnn/helmet-detection-yolo11
- Downloaded file: `yolov11m(100epochs).pt`
- Local name: `models/rider_yolo11m.pt`
- SHA256: `5769c45395a73739cb35cf28ed16e5d0acc3a0c90e1fac588fdb8a7403b3a930`
- Classes observed in the weight: `With Helmet`, `Without Helmet`
- Role: optional comparison/fallback model; not enabled by the V3 default configuration
- Model-page license marker at integration time: MIT

## Important licensing note

A model-page license marker does not prove that every training image, annotation, dependency, or downstream use has the same license. Before commercial deployment, review:

1. model-card and weight license terms;
2. training-data provenance and usage rights;
3. Ultralytics framework and pretrained-weight terms;
4. privacy, retention, and employee-monitoring requirements applicable to the deployment site.
