# helmet_detect

[![CI](https://github.com/jsdfhasuh/helmet_detect/actions/workflows/ci.yml/badge.svg)](https://github.com/jsdfhasuh/helmet_detect/actions/workflows/ci.yml)

固定监控摄像头下的电动车／摩托车骑行人员安全头盔检测。项目重点解决远距离小目标、整图缩放后头部像素不足、单帧漏检，以及停放车辆和后视镜等背景干扰。

这不是只展示调用方式的空壳示例：GitHub Actions 会下载并校验公开权重、导出固定输入尺寸的 ONNX、运行真实模型推理，并用来自现场视频的隐私遮蔽帧完成图像和视频回归。未能检出预期的未戴头盔目标时，CI 会直接失败。

## 已验证结果

回归样本来自用户提供的 1920×1080 固定摄像头视频。仓库只保存一张缩放为 448×448 的隐私遮蔽帧，并以 Base64 文本形式保存：保留检测目标附近区域，其余位置使用纯色覆盖，不包含原视频、时间水印和摄像头名称。回归脚本会校验 SHA256 后还原 WebP，CI 再将该真实帧重复 9 次生成短视频，执行真实视频推理和时序投票。

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
按模型独立配置；当前为 YOLOv8n 0.20、YOLO11m 0.10。
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
