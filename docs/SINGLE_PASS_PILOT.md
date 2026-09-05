# 单模型骑行人员与头部状态试验

这是明确选择才启用的试验入口，不替换 V3 默认配置，不表示生产验收通过。
代码与训练脚本可以公开；本机位模型、原始录像、训练图和标签不上传公开仓库。

## 改动

一个 YOLO11n 前向同时预测 `rider / helmet_head / no_helmet_head`，不再串行运行
COCO 人车关联和每人多尺度头盔检测。输入为完整画面，无固定 ROI、无录像时间捷径。
头部只能关联给一个几何关系合理的骑行人员；归属歧义或类别冲突输出 UNKNOWN。
轻量几何跟踪与短时身份拼接用于去重（本试验入口不是 ByteTrack）。
0.8 秒内至少两个不同时间的未戴盔观察才产生候选事件；未知状态不累计违规票。
同一持续可见轨迹不会因为几帧 HELMET 闪烁而重新报警。长时间失联仍可能导致 ID
重建，跨人遮挡与长时间重复报警需要独立现场验证。

## 数据与可复现性

首轮训练为 60 epochs，官方 YOLO11n 初始化，冻结前 10 层，训练输入 640，CPU 4 线程，
固定使用最后 epoch 权重，未根据测试录像选择 best checkpoint。
只有 8 张人工审阅目标源帧（4 张裸头骑出、4 张戴盔骑入）和 4 张背景源帧，
派生出 48 张局部训练图。派生图数量不是独立样本数量。train=val 的数字只是拟合诊断，
不得作为泛化指标。约 57 秒经过的帧未参与训练，但仍来自反复调试过的同一录像；
约 125 秒的戴盔经过参与了训练，属于拟合检查。多人交叉、推车、陌生头盔、夜间和
其他日期没有独立验收。训练标签也需要现场人员复核。

在 PRIVATE 本地数据目录准备标准 YOLO 数据集和 YAML，三个类别必须严格一致：

```powershell
python scripts\train_rider_head_pilot.py --data D:\private_data\data.yaml --base-weights models\yolo11n.pt --output D:\private_runs\pilot --fit-only
```

`--fit-only` 是允许 train=val 的显式提示，不允许将拟合结果报成独立验证。

## 离线试跑

先切换到本 PR 分支，安装 `python -m pip install -e ".[dynamic,dev]"`。
从私有交付包取得 `rider_head_site_pilot.pt` 并核对包内 SHA256。不要用原来两分类权重替换。

```powershell
python -m helmet_detect.pilot_run --weights models\rider_head_site_pilot.pt --source D:\videos\camera.mp4 --output-dir artifacts\pilot_run_01 --imgsz 960 --sample-fps 5 --save-video --accept-experimental
```

输出为 `frames.jsonl / summary.json / events / annotated.mp4`。
帧时间是源视频时间，不是实时采集延迟。平均吞吐不能代替提前报警的事件评估。
不加 `--save-video` 可排除连续录像保存开销；事件小记录先输出，截图由有界异步队列保存。

## 短时摄像头试验

通过环境变量提供流地址，命令参数不包含摄像头密码：

```powershell
python -m helmet_detect.pilot_run --weights models\rider_head_site_pilot.pt --stream-env CAMERA_RTSP --output-dir artifacts\live_pilot_01 --imgsz 960 --duration 60 --accept-experimental
```

独立取流线程持续解码，只保留一个最新帧；读取的每帧最多推理一次，不重复计票。
先预热模型再开始取流，网络捕获有打开/读取超时。`post_decode_to_result_ms` 只反映
本程序收到解码帧之后的时间，不包含摄像头编码、传输及底层解码队列延迟。
实时模式不允许同步连续录像。断流会明确失败，不具备无人值守自动重连/磁盘轮转服务。
未在用户实际摄像头、显卡和网络上验收。不要直接接入门禁或安全联锁。

## CI 的边界

CI 覆盖最新帧覆盖、关闭唤醒、不同帧投票、证据过期、歧义头部归属、类别冲突和
会话去重。原有公共模型回归继续执行；这不是对私有新模型进行的场景准确率验收。
必须使用独立日期与独立人员的视频，标注正确的违规、戴盔、推车和行人事件及报警截止
时间，才能决定是否替换主分支。当前仅推荐有人值守试验。
