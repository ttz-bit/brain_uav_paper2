# Paper 2 中文初稿一致性审查与改稿建议

审查对象：`C:\Users\24104\Desktop\论文2.docx`、`paper2.txt`、`论文.txt`、`实验.txt`、`phase3_vision_synops_report.json`、项目代码与现有报告。

## 1. 总体判断

当前中文初稿的核心方向基本正确：论文 2 应写成“固定 Paper 1 规划器 + 未知动态目标 + 视觉定位/状态估计前端”的闭环验证，而不是重新讲 Paper 1 的 SNN-TD3 规划算法。草稿中关于局部图像、像素到世界坐标、阶段化 GSD、SNN/CNN heatmap 对比、KF/raw 状态估计接口的主线，都能在代码中找到对应实现。

但是，当前 DOCX 仍有三个投稿风险：

1. 参考文献仍是 IEEE 模板占位文献，必须全部替换为真实文献。
2. 草稿摘要已经写入视觉定位数值，但正文“实验设计/结果/结论”尚未完成，且闭环结果不能支持“SNN 全面优于 CNN”的表述。
3. 文档属性显示当前约 4 页、3469 words，但缺少结果、结论和真实参考文献；若直接补全会超过 5 页，需要主动压缩引言和相关工作。

## 2. 与代码/结果一致的表述，可以保留

这些表述与当前代码和结果一致，可以作为论文主干：

- “下游 SNN-TD3 规划器固定，本文研究视觉观测及状态估计前端对闭环接近的影响。”对应 `scripts/run_phase3_vision_td3.py` 和 `scripts/run_phase3_snn_td3_oracle.py` 的固定 planner 接口与动态目标评估流程。
- “正式视觉定位评估使用同一 dataset/split/eval count/decode/water constraint。”对应 `outputs/reports/phase3_vision_eval_compare_formal/phase3_vision_eval_context.json`，其中 `same_*` 均为 true，`split=test`，`num_eval=11480`。
- “三阶段 GSD 和物理尺度一致。”对应 `configs/env_phase3_task_v1.0.2.yaml`：far/mid/terminal 的 GSD 分别为 0.020/0.010/0.005 km/px，目标长度 0.2 km、宽度 0.04 km。
- “主正式数据集为 no-port、no-distractor clean map 设置。”对应 `outputs/tmp_review_phase3_no_port_nodistractor/meta/generation_config.json`：`real_render_mode=map`、`reject_keywords=["port"]`、`distractor_count_min=max=0`。
- “SynOps/MACs 只能作为软件侧代理指标，不能写成实测能耗优势。”这一点与你的 `phase3_vision_synops_report.json` 中 caveat 完全一致。

## 3. 必须修改的关键表述

### 3.1 不能写“SNN 在所有场景全面优于 CNN”

当前正式视觉评估支持的结论是：

- SNN overall mean world error = 23.86 m，CNN = 148.45 m，SNN 显著更低。
- far 阶段 SNN = 41.62 m，CNN = 407.84 m，SNN 显著更低。
- coastal 背景 SNN pixel mean = 2.14 px，CNN = 20.61 px，SNN 显著更低。
- 但 terminal 阶段 CNN = 7.84 m，SNN = 11.83 m，CNN 更低。
- open_sea 和 island_complex 背景中 CNN pixel mean 也低于 SNN。
- overall P90 中 CNN = 44.36 m，SNN = 45.23 m，CNN 略低。

建议把摘要和结果中的结论改为：

> 实验结果表明，SNN 前端在总体平均世界坐标误差、远距阶段和海岸复杂背景下显著优于同结构 CNN，体现出对长尾困难样本的鲁棒性；但 CNN 在末端阶段、简单背景和第 90 百分位误差上保持竞争性甚至略优，因此本文不将 SNN 解释为所有场景的全面优势。

### 3.2 闭环实验不能支持“SNN 闭环优于 CNN”

当前闭环汇总表：

- Oracle-GT、SNN no-KF、SNN KF/raw、SNN full-KF、CNN KF/raw 都是 62/64 capture，valid capture 都是 62/62。
- Est. Err.：SNN KF/raw 在当前汇总表里是 37.48 m，CNN KF/raw = 26.40 m；但 `论文2.docx` 里写成了 37.84 m，终稿需要统一这一处数值。
- Vision Err.：SNN KF/raw = 42.88 m，CNN KF/raw = 29.93 m。
- Hard Viol. 全部为 0。

因此正文应改成：

> 闭环实验中，所有视觉增强组与真值上界组在当前 64 回合设置下均达到 62/64 的捕获结果，并保持硬约束违规为 0，说明视觉观测接入固定规划器后没有破坏闭环可达性和安全约束。CNN 在该闭环汇总指标上具有更低的平均估计误差，因此本文的闭环结论应聚焦于 SNN 前端的可行性和远距/复杂背景定位鲁棒性，而不是声称 SNN 闭环性能全面超过 CNN。

### 3.3 “干扰目标/云雾遮挡/运动模糊”要收紧

草稿中多处写到“干扰目标库、轻度云雾遮挡、运动模糊”。但正式 no-port/no-distractor 主实验里干扰船已关闭，当前审查到的正式 clean-map 配置不支持把干扰船作为主实验因素来写。

建议改为：

> 渲染框架预留干扰目标和成像扰动接口；为隔离视觉定位与闭环状态估计的基础影响，本文主实验采用 no-port、no-distractor 的 clean-map 设置，仅保留背景类别变化、目标姿态变化、阶段尺度变化和水域约束。干扰船和更强天气退化作为后续扩展或附加诊断，不作为本文主结论依据。

### 3.4 SeaDronesSee 不要写成主闭环证据

`scripts/eval_phase3_heatmap_on_seadronessee.py` 的 caveat 很清楚：SeaDronesSee crops 没有 Phase3 世界坐标或 GSD，只能报告 crop pixel metrics；如果 crop 已居中，center-baseline 要谨慎解释。因此 SeaDronesSee 只能作为外部视觉定位验证，不应写成闭环或世界坐标验证。

建议改为：

> 公开海上数据集用于验证视觉前端在真实图像裁剪上的外部定位行为；由于公开数据不提供本文闭环所需的目标世界坐标、飞行器状态、环境约束和在线规划接口，闭环结论仍来自任务型几何一致渲染与在线仿真。

## 4. 推荐 4-5 页结构

为了和论文 1 同时投稿且避免重复，建议 5 页内这样分配：

- 第 1 页：标题、摘要、关键词、引言前半。摘要控制 230-280 中文字，不放太多解释性背景。
- 第 1-2 页：引言后半 + 贡献。Paper 1 只一句话作为固定规划器来源，不展开 TD3 算法。
- 第 2 页：相关工作，三小段即可：无人机/海上视觉数据集，合成/Copy-Paste 数据构建，heatmap/SNN/状态估计。
- 第 2-3 页：方法。压缩成总体框架、数据集构建、SNN heatmap、坐标映射与阶段感知 KF/raw。
- 第 3-4 页：实验。一个数据集/协议表，一个视觉定位图或表，一个闭环表。
- 第 5 页：讨论、结论、参考文献。讨论必须写清楚限制：非真实相机链路、非实测能耗、主实验无干扰船、CNN 在部分指标更优。

## 5. 结果段落建议替换稿

可直接替换“结果与讨论”的核心段：

> 在 11480 张 held-out test 图像上，SNN-enhanced 的总体平均世界坐标误差为 23.86 m，低于 CNN-enhanced 的 148.45 m。该优势主要来自远距阶段和海岸复杂背景：远距阶段 SNN/CNN 平均世界坐标误差分别为 41.62 m 和 407.84 m，海岸背景平均像素误差分别为 2.14 px 和 20.61 px。与此同时，CNN 在末端阶段取得更低误差（7.84 m vs. 11.83 m），并在总体 P90 世界坐标误差上略低于 SNN（44.36 m vs. 45.23 m）。因此，本文将 SNN 的优势解释为远距和复杂背景下的平均误差与长尾鲁棒性改善，而不是所有阶段的全面优势。

> 在闭环动态目标接近实验中，Oracle-GT、SNN no-KF、SNN KF/raw、SNN full-KF 和 CNN KF/raw 均获得 62/64 的捕获结果，并且硬约束违规次数均为 0。SNN KF/raw 相比 no-KF 将平均目标估计误差由 39.22 m 降至 37.48 m，说明阶段感知估计接口对稳定视觉观测有一定帮助；但当前汇总表里的 CNN KF/raw 仍更低（26.40 m）。需要注意的是，`论文2.docx` 对 SNN KF/raw 写成了 37.84 m，因此在终稿或导出表格里要先统一这一处数字，再定最终版本。闭环结果更适合用于证明“视觉定位前端可以接入固定规划器并保持闭环可达性与安全约束”，而不是证明 SNN 在闭环层面绝对优于 CNN。

## 6. 相关工作必须替换的真实文献

建议最少引用以下真实文献，足够支撑 5 页短文：

1. UAVDT：D. Du et al., “The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking,” ECCV 2018. https://arxiv.org/abs/1804.00518
2. VisDrone：P. Zhu et al., “Vision Meets Drones: A Challenge,” arXiv/VisDrone challenge papers. https://arxiv.org/search/?query=VisDrone&searchtype=all
3. SeaDronesSee：Varga et al., “SeaDronesSee: A Maritime Benchmark for Detecting Humans in Open Water,” WACV 2022. https://openaccess.thecvf.com/WACV2022
4. Copy-Paste：G. Ghiasi et al., “Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation,” CVPR 2021. https://arxiv.org/abs/2012.07177
5. CenterNet：X. Zhou et al., “Objects as Points,” arXiv 2019. https://arxiv.org/abs/1904.07850
6. CenterTrack：X. Zhou et al., “Tracking Objects as Points,” ECCV 2020. https://arxiv.org/abs/2004.01177
7. DSNT / differentiable coordinate decoding：A. Nibali et al., “Numerical Coordinate Regression with Convolutional Neural Networks,” ECCV Workshops 2018. https://arxiv.org/abs/1801.07372
8. SNN surrogate gradient：E. O. Neftci, H. Mostafa, F. Zenke, “Surrogate Gradient Learning in Spiking Neural Networks,” IEEE Signal Processing Magazine, 2019. https://ieeexplore.ieee.org/document/8891809
9. SpikingJelly：W. Fang et al., “SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence,” Science Advances, 2023. https://www.science.org/doi/10.1126/sciadv.adi1480
10. TD3：S. Fujimoto, H. van Hoof, D. Meger, “Addressing Function Approximation Error in Actor-Critic Methods,” ICML 2018. https://arxiv.org/abs/1802.09477
11. Kalman filter：R. E. Kalman, “A New Approach to Linear Filtering and Prediction Problems,” 1960. https://doi.org/10.1115/1.3662552

## 7. 摘要建议压缩方向

当前摘要信息量很大，但 5 页稿中过长。建议压成四句：

1. 问题：固定在线规划器在未知动态海上目标下不能直接获得真值目标状态。
2. 方法：SNN heatmap 视觉定位 + 像素到世界坐标映射 + far/mid KF、terminal raw 的阶段感知状态估计。
3. 视觉结果：保留 23.86 vs 148.45、far 41.62 vs 407.84、coastal 2.14 vs 20.61。
4. 闭环与限制：各组 62/64 capture、0 hard violation；GPU 上 SNN 不更快，SynOps 仅为代理指标。

## 8. 最优投稿叙事

论文 2 不要和论文 1 抢“规划器创新”。建议题目和全文都围绕：

> 面向固定在线规划器的视觉状态前端：从真实目标状态输入转向视觉估计目标状态输入。

也就是：论文 1 解决“已知目标点下怎么规划”；论文 2 解决“目标未知且运动时，视觉如何把目标状态估出来并喂给固定规划器”。这样两篇同时投稿时，贡献边界清楚，重复最少。
