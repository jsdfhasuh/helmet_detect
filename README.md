# helmet_detect

[![CI](https://github.com/jsdfhasuh/helmet_detect/actions/workflows/ci.yml/badge.svg)](https://github.com/jsdfhasuh/helmet_detect/actions/workflows/ci.yml)

固定监控摄像头下的电动车／摩托车骑行人员安全头盔检测。项目重点解决远距离小目标、整图缩放后头部像素不足、单帧漏检，以及停放车辆和后视镜等背景干扰。

这不是只展示调用方式的空壳示例：GitHub Actions 会下载并校验公开权重、导出固定输入尺寸的 ONNX、运行真实模型推理，并用来自现场视频的隐私遮蔽帧完成图像和视频回归。未能检出预期的未戴头盔目标时，CI 会直接失败。

## 已验证结果

回归样本来自用户提供的 1920×1080 固定摄像头视频。仓库只保存一张缩放为 448×448 的 WebP 隐私遮蔽帧：保留检测目标附近区域，其余位置使用纯色覆盖，不包含原视频、时间水印和摄像头名称。CI 会在运行时将该真实帧重复 9 次生成短视频，再执行真实视频推理和时序投票。

当前固定环境基准结果：

| 回归场景 | 期望 | 实际 | 未戴头盔最高置信度 |
|---|---:|---:|---:|
| 同一真实画面的空检测门区域 | 不报警 | 正确不报警 | 0.0000 |
| 真实摄像头未戴头盔人员 | 报警 | 正确报警 | 约 0.2723 |
| 由真实帧生成的 9 帧视频 | 至少 8 帧报警、至少 1 个事件 | 9 帧报警、1 个事件 | 约 0.2723 |

这些数值是本摄像头回归样本上的工程结果，不代表通用精度，也不能替代独立测试集评估。CI 固定 NumPy 2.3.5 和 OpenCV 4.13.0.92，以避免不同推理运行时造成阈值漂移。

## 检测流程

```text
固定摄像头视频
    ↓
裁剪有效区域 ROI，避免把 1920×1080 整图缩小后丢失头部细节
    ↓
YOLOv8n 头盔模型（快速模式）
或 YOLOv8n + YOLO11m 骑行头盔模型（融合配置）
    ↓
只保留检测门 gate 内的头部结果
    ↓
按模型独立阈值判断未戴头盔
    ↓
最近 5 个采样帧中至少 2 帧命中
    ↓
生成一次事件、标注视频、逐帧 JSONL 和汇总 JSON
```

当前自动回归使用轻量 YOLOv8n，能够在隐私遮蔽后的真实现场画面中输出 `Without Helmet`。仓库也保留 YOLO11m 融合配置，用于处理完整监控视频时进行对比；长期生产部署更适合使用该摄像头数据微调一套单模型，降低算力开销并减少模型冲突。

## 环境安装

建议使用 Python 3.10～3.12。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[runtime,export]"
```

Linux：

```bash
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[runtime,export]"
```

下载公开 PT 权重、校验 SHA256，并导出固定 960×960 ONNX：

```bash
python scripts/prepare_models.py
```

只准备 CI 使用的轻量模型：

```bash
python scripts/prepare_models.py --models helmet_yolov8n
```

默认生成：

```text
models/helmet_yolov8n_960.onnx
models/rider_yolo11m_960.onnx
```

模型文件不会提交到 Git 仓库。GitHub Actions 会缓存导出的 ONNX，减少后续运行时间。

## 测试单张图片

仓库自带隐私遮蔽后的真实现场帧：

```bash
helmet-detect image \
  --config configs/parking_exit_roi.json \
  --source testdata/camera_exit/images/event_frame_6.webp \
  --output artifacts/image.jpg \
  --json artifacts/image.json
```

## 测试视频

先根据真实帧生成 9 帧短视频：

```bash
python scripts/build_sample_video.py \
  --output artifacts/sample/no_helmet_event.avi
```

再执行轻量模型和时序投票：

```bash
helmet-detect video \
  --config configs/parking_exit_fast.json \
  --source artifacts/sample/no_helmet_event.avi \
  --output artifacts/sample/annotated.mp4 \
  --records artifacts/sample/frames.jsonl \
  --summary artifacts/sample/summary.json \
  --sample-fps 6 \
  --fail-on-no-alarm
```

处理原始 1920×1080 摄像头视频时，使用完整画面坐标配置：

```bash
helmet-detect video \
  --config configs/parking_exit_full_frame.json \
  --source input.mp4 \
  --output artifacts/annotated.mp4 \
  --records artifacts/frames.jsonl \
  --summary artifacts/summary.json \
  --sample-fps 5
```

仅运行轻量模型时增加：

```text
--models helmet_yolov8n
```

## 配置说明

`configs/parking_exit_full_frame.json` 中的关键参数：

```json
{
  "roi": [360, 220, 920, 780],
  "gate": [520, 420, 790, 545],
  "candidate_threshold": 0.02,
  "temporal": {
    "window": 5,
    "min_hits": 2,
    "cooldown_seconds": 3.0
  }
}
```

- `roi`：从完整画面裁剪出的推理区域，坐标格式为 `x1,y1,x2,y2`。
- `gate`：只允许检测框中心落在该区域的结果参与报警，降低停放车辆和背景目标干扰。
- `candidate_threshold`：进入后处理的最低置信度，不是最终报警阈值。
- `alarm_threshold`：每个模型独立配置；当前为 YOLOv8n 0.20、YOLO11m 0.10。
- `window/min_hits`：时序投票参数，用于降低单帧误检。

固定摄像头位置、焦距或分辨率变化后，必须重新测量 ROI 和 gate，不能直接照搬当前坐标。

## 自动化验证

`.github/workflows/ci.yml` 在每次 push、Pull Request 和手动触发时执行：

1. Ruff 静态检查和 pytest 单元测试；
2. 下载公开 YOLOv8n PT 权重并校验 SHA256；
3. 导出 OpenCV DNN 可运行的固定 960×960 ONNX；
4. 对检测门负样本规则和真实未戴头盔正样本执行模型推理；
5. 检查实际报警状态、指定模型和最低置信度；
6. 根据真实帧生成 9 帧无损 FFV1 AVI，并执行视频推理；
7. 检查视频至少产生 8 个报警帧、1 个时序事件，且最高置信度至少为 0.20；
8. 上传标注图片、标注视频、逐帧 JSONL、CSV 和报告作为 Actions Artifact。

本地运行同一套回归：

```bash
pip install -e ".[runtime,dev]"
python scripts/run_regression.py
```

`.github/workflows/manual-video.yml` 可在 Actions 页面手动运行：不填写 URL 时测试仓库样本；填写可直接下载的视频 URL 时处理指定视频。公共仓库的 Actions 与 Artifact 不适合上传敏感监控视频，涉及人员隐私时应使用私有仓库或在内网运行。

## 模型来源

- `iam-tsr/yolov8n-helmet-detection`：YOLOv8n，权重中的类别为 `With Helmet / Without Helmet`。
- `nnsohamnn/helmet-detection-yolo11`：YOLO11m，权重中的类别为 `With Helmet / Without Helmet`。

下载地址、文件摘要和许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。正式商业部署前，应重新核对模型、训练数据和 Ultralytics 的许可证要求。

## 已知限制与生产化方向

当前代码证明了该摄像头画面中的未戴头盔人员可以被模型检测出来，但公共模型不能直接视为生产验收模型：

- 远距离头部只有约 20～35 像素，姿态、压缩和运动模糊会导致置信度波动；
- 公共模型可能在普通帽子、遮挡和侧面场景上产生不同判断；
- 当前时序投票以检测门区域为单位，尚未按人员 ID 分轨迹投票；
- 多人同时通过时，应增加人员／骑行人员检测与 ByteTrack；
- 建议从该摄像头标注 300～500 张困难样本，训练一套单模型生产权重；
- 正式验收应按“人员事件”统计召回率、误报率和重复报警率，而不是只看单帧 mAP。
