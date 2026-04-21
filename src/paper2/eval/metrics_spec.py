VISUAL_METRICS = {
    "detection_rate": "检测成功率。用于统计目标是否被正确检出。",
    "center_l2_error": "检测中心点与标注中心点的欧氏距离误差。",
    "bbox_iou": "预测框与标注框的交并比。若当前任务仅做中心点定位，可先保留为占位指标。",
}

TRACKING_METRICS = {
    "obs_conf": "视觉观测或状态估计输出的置信度。",
    "obs_age": "距离最近一次有效观测经过的时间。",
}

PLANNING_METRICS_FUTURE = {
    "success_rate": "闭环任务成功率。当前阶段仅占位，等待论文1环境桥接。",
    "path_length_ratio": "实际路径长度与参考路径长度之比。当前阶段仅占位。",
    "min_clearance": "飞行轨迹与禁飞区边界的最小安全裕度。当前阶段仅占位。",
    "inference_latency": "在线推理平均时延。当前阶段仅占位。",
    "effective_compute": "有效计算量或近似 MAC 口径。当前阶段仅占位。",
}


def print_metric_summary() -> None:
    print("=== VISUAL_METRICS ===")
    for k, v in VISUAL_METRICS.items():
        print(f"{k}: {v}")

    print("\n=== TRACKING_METRICS ===")
    for k, v in TRACKING_METRICS.items():
        print(f"{k}: {v}")

    print("\n=== PLANNING_METRICS_FUTURE ===")
    for k, v in PLANNING_METRICS_FUTURE.items():
        print(f"{k}: {v}")