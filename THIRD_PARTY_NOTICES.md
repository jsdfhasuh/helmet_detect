# Third-party model notices

本仓库不提交第三方模型权重。`scripts/prepare_models.py` 在用户执行或 GitHub Actions 运行时从公开模型仓库下载权重。

## 1. YOLOv8n helmet detection

- Model page: https://huggingface.co/iam-tsr/yolov8n-helmet-detection
- Downloaded file: `best.pt`
- Local name: `helmet_yolov8n.pt`
- SHA256: `c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d`
- Classes observed in the weight: `With Helmet`, `Without Helmet`
- Model page license marker at the time of integration: MIT

## 2. YOLO11m rider helmet detection

- Model page: https://huggingface.co/nnsohamnn/helmet-detection-yolo11
- Downloaded file: `yolov11m(100epochs).pt`
- Local name: `rider_yolo11m.pt`
- SHA256: `5769c45395a73739cb35cf28ed16e5d0acc3a0c90e1fac588fdb8a7403b3a930`
- Classes observed in the weight: `With Helmet`, `Without Helmet`
- Model page license marker at the time of integration: MIT

## Important licensing note

A model-page license marker does not automatically prove that every training image, annotation, dependency, or downstream use has the same license. The user is responsible for reviewing the model cards, dataset provenance, Ultralytics licensing, and the intended commercial deployment before production use.
