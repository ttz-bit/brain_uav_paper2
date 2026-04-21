# Phase 0 Freeze Note

## 1. 当前阶段目标

当前阶段为论文2工程的“统一接口与实验口径冻结阶段”。

本阶段目标不是获得实验结果，而是统一以下内容：

- 工程目录结构
- 配置文件入口
- 核心状态接口
- 视觉观测接口
- 数据集统一 schema
- 实验模式边界
- 输出目录规范
- 指标命名规范

---

## 2. 当前已经冻结的内容

### 2.1 工程结构冻结
- `configs/`
- `docs/`
- `outputs/`
- `src/paper2/common/`
- `src/paper2/datasets/`
- `src/paper2/env_adapter/`
- `src/paper2/eval/`
- `scripts/`

### 2.2 状态接口冻结
- `AircraftState`
- `NoFlyZoneState`
- `TargetTruthState`
- `VisionObservation`
- `TargetEstimateState`

### 2.3 环境协议冻结
- `Paper2EnvProtocol`

### 2.4 数据 record 统一字段冻结
以 `src/paper2/datasets/unified_schema.py` 中的 `REQUIRED_KEYS` 为准。

### 2.5 实验模式命名冻结
- `public_validation`
- `synthetic_visual`
- `closed_loop`

### 2.6 输出目录命名冻结
- `outputs/synth`
- `outputs/train`
- `outputs/eval`

---

## 3. 当前明确不冻结的内容

以下内容暂不在当前阶段冻结，等待论文1物理口径修正后统一：

- 最终世界单位
- 最终时间步长 dt
- 最终飞行器速度
- 最终禁飞区半径范围
- 最终 render GSD
- 最终闭环实验公式与统计口径

---

## 4. 三类实验边界说明

### 4.1 公开数据外部视觉验证
公开数据仅用于验证视觉模块在真实公开数据上的有效性，不直接用于论文2闭环规划。

### 4.2 Synthetic 视觉训练与预验证
synthetic 数据用于：
- 打通从状态到图像的生成流程
- 训练视觉模块
- 做视觉模块的预验证

### 4.3 闭环实验
闭环实验暂不在当前阶段开启。待论文1环境与物理口径修正完成后，通过桥接层接入。

---

## 5. 当前阶段完成判据

满足以下条件即认为 Phase 0 完成：

1. `check_interface.py` 可运行通过
2. `experiment.yaml` 可读取
3. `metrics_spec.py` 可导入
4. `phase0_freeze.md` 已写明冻结项与非冻结项
5. 输出目录规范已建立