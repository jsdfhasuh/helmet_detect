# helmet_detect

[![CI](https://github.com/jsdfhasuh/helmet_detect/actions/workflows/ci.yml/badge.svg)](https://github.com/jsdfhasuh/helmet_detect/actions/workflows/ci.yml)

面向固定监控视频的电动车／摩托车骑行人员安全头盔检测。V2 已从固定 ROI / Gate 重构为**全画面人员检测、目标跟踪、按人员动态裁剪、头盔状态判断和按轨迹投票**，人员不再必须经过某一组硬编码坐标。

## V2 使用的模型已经确定

默认生产验证链路使用两级模型：

1. **场景模型：Ultralytics 官方 `yolo11n.pt`（COCO 预训练）**
   - 在完整画面中检测 `person`、`bicycle`、`motorcycle`；
   - 视频模式通过 **ByteTrack** 为人员分配稳定 ID；
   - 默认输入尺寸为 1280，优先保证远距离人员召回；
   - 它只负责“人在哪里”，不负责判断头盔。
2. **头盔模型：`iam-tsr/yolov8n-helmet-detection`**
   - 类别为 `With Helmet` / `Without Helmet`；
   - 对每一个人员周围自动生成的多尺度上下文图单独推理；
   - 默认输入尺寸为 960；
   - 它负责“这个人的头部是否佩戴安全头盔”。

此前测试过的 `nnsohamnn/helmet-detection-yolo11`（YOLO11m）保留为可选对比模型，但没有放进 V2 默认链路：它在 CPU 上明显更慢，而且公开模型在普通帽子、背面和遮挡场景中仍会与轻量模型发生判断冲突。正式上线前，仍建议使用本机位标注数据微调最终的头盔模型。

## 为什么不是固定 ROI

旧版流程为：

```text
固定 ROI → 头盔模型 → 固定 Gate → 区域级报警
```

它只能证明指定位置上的目标可识别，摄像头角度或人员路线变化后容易失效。

V2 流程为：

```text
完整视频帧
    ↓
YOLO11n 全画面寻找 person / bicycle / motorcycle
    ↓
ByteTrack 给每个人分配 ID
    ↓
根据每个人的位置和身高动态生成多尺度方形上下文图
    ↓
YOLOv8n 判断 With Helmet / Without Helmet
    ↓
只关联落在该人员头部候选区内的头盔结果
    ↓
按 track_id 独立进行多帧投票和报警去重
```

`alarm_zone` 仍可选配为归一化多边形，但它只限制**哪里允许报警**，不会限制模型只能看哪里。默认配置中 `alarm_zone` 为 `null`，表示全画面均可检测和报警。

## 动态裁剪不是固定“人体顶部 30%”

骑行人员会低头、弯腰、背对摄像头或被立杆遮挡，简单取人体框顶部固定比例并不稳定。V2 会根据：

- 当前人员框高度；
- 当前画面宽高；
- 多个上下文尺度；
- 人员头部附近的中心区域；

生成最多四个不同尺度的动态上下文图。头盔模型的检测框映射回原画面后，只有中心落在该人员头部候选区中的结果才参与该人员的状态判断，避免把旁边车辆上挂着的头盔错误关联给当前人员。

## GitHub Actions 的真实验收

`.github/workflows/ci.yml` 每次提交都会执行：

1. Ruff 静态检查与 pytest 单元测试；
2. 下载并校验官方 YOLO11n 和公共头盔模型的 SHA256；
3. 将同一个隐私遮蔽后的真实现场目标放到 1280×720 画面的左、中、右三个不同位置；
4. 分别执行真实 YOLO 推理，检查三个位置都能找到人员并判断为 `Without Helmet`；
5. 生成连续视频，运行 ByteTrack 与按人员 ID 的时序投票；
6. 要求产生稳定人员轨迹和至少一个未戴头盔事件；
7. 上传标注图片、标注视频、逐帧 JSONL、CSV 和报告。

这个回归专门防止代码退回到“只有目标落在固定 ROI 才能通过”的实现。仓库不保存原始监控视频，只保存去除摄像头名称、时间水印并遮蔽无关背景后的远距离目标片段，运行时会先校验 SHA256。

## 本地安装

推荐 Python 3.10～3.12：

```bash
git clone https://github.com/jsdfhasuh/helmet_detect.git
cd helmet_detect
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dynamic,dev]"
```

下载并校验默认模型：

```powershell
python scripts\prepare_dynamic_models.py
```

将生成：

```text
models/yolo11n.pt
models/helmet_yolov8n.pt
models/dynamic_model_sources.json
models/DYNAMIC_SHA256SUMS.txt
```

运行和 GitHub Actions 相同的动态回归：

```powershell
python scripts\run_dynamic_regression.py
```

成功时应看到：

```text
Dynamic V2 helmet regression
Result: PASS
```

结果位于：

```text
artifacts/dynamic-regression/
```

## 检测自己的视频

```powershell
helmet-detect video `
  --config .\configs\dynamic_full_frame.json `
  --source "D:\videos\camera.mp4" `
  --output .\artifacts\my-video\annotated.mp4 `
  --records .\artifacts\my-video\frames.jsonl `
  --summary .\artifacts\my-video\summary.json `
  --sample-fps 5 `
  --show-contexts
```

不需要查看动态裁剪框时，删除 `--show-contexts`。要把“完全没有形成报警事件”视为命令失败，可增加：

```text
--fail-on-no-alarm
```

单张图片：

```powershell
helmet-detect image `
  --config .\configs\dynamic_full_frame.json `
  --source .\test.jpg `
  --output .\artifacts\image\annotated.jpg `
  --json .\artifacts\image\result.json `
  --show-contexts
```

## 配置重点

`configs/dynamic_full_frame.json` 中没有固定 ROI 或 Gate。常用参数：

- `scene_model.image_size`：全画面人员检测分辨率，默认 1280；
- `scene_model.minimum_person_height`：忽略过小、几乎不可辨识的人员；
- `scene_model.device`：`auto`、`cpu`、`0` 等；
- `association.require_vehicle`：是否要求人员必须与自行车／摩托车匹配；当前默认关闭，避免 COCO 对电动车漏检导致整条链路失效；
- `helmet_model.context_height_multipliers`：按人员高度生成动态裁剪尺度；
- `helmet_model.context_frame_scales`：按画面尺寸增加上下文尺度；
- `helmet_model.no_helmet_threshold`：未戴头盔单次判断阈值；
- `temporal.window`：每条人员轨迹保存的状态窗口；
- `temporal.minimum_no_helmet_hits`：形成事件所需的未戴头盔命中次数；
- `alarm_zone`：可选归一化报警多边形，`null` 表示全画面。

更换摄像头时通常需要重新评估分辨率、最小人员高度、头盔阈值和上下文尺度，但不需要重新画一个固定矩形 ROI。

## 手动 Actions 视频检测

在 GitHub 的 **Actions → Manual dynamic video detection → Run workflow** 中，可以填写一个可直接下载的视频地址。工作流会使用 V2 全画面动态链路，并上传：

```text
annotated.mp4
frames.jsonl
summary.json
事件截图目录
```

公共仓库的 Actions 日志和 Artifact 不适合处理敏感监控内容。涉及人员隐私的视频应在本地、内网或私有仓库中运行。

## 当前限制

- COCO 模型的 `motorcycle` 类并不等同于所有中国电动车，因此默认不会强制要求检测到车辆；
- 第一阶段人员漏检时，第二阶段无法判断该人员头盔；
- 远距离头部仅有二三十像素时，姿态、压缩、运动模糊仍会造成波动；
- 公共头盔模型只能用于验证和冷启动，不能代替本机位独立验证；
- 正式验收应以“人员轨迹事件”为单位统计召回率、误报率和重复报警率，而不是只查看单帧 mAP；
- 建议从现场收集 300～500 张代表性困难样本，微调 `With Helmet / Without Helmet` 模型。

模型来源、摘要和许可注意事项见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
