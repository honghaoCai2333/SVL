# HARP: Hierarchical Action-aware Reward Planning for Embodied VLM

## 一、项目概述

### 1.1 目标
训练一个基于强化学习的VLM模型，能够根据场景图像和任务描述生成可执行的规划序列（Navigate/Pick/Place）。

### 1.2 技术栈
- **基础模型**: Qwen2-VL-7B-Instruct
- **训练框架**: TRL (Transformer Reinforcement Learning)
- **核心创新**: 层次化过程奖励模型 (H-PRM)

### 1.3 数据流
```
Scene Image + Task Description
        ↓
    Qwen2-VL (Policy)
        ↓
Planning Sequence: [Navigate(X), Pick(Y), Navigate(Z), Place(Z)]
        ↓
    HTTP Call → VLA Service (其他同学负责)
```

---

## 二、核心创新点

### 2.1 创新点1: 层次化过程奖励模型 (H-PRM)

**动机**: 传统RL只在序列结束时给奖励，忽略了具身规划中动作之间的依赖关系。

**设计**: 三层奖励结构
```
R_total = α·R_action + β·R_transition + γ·R_plan

R_action:     单个动作的合理性（动作-场景匹配度）
R_transition: 动作转移的合法性（基于状态机约束）
R_plan:       整体规划的完整性和效率
```

**状态机约束**:
```
INIT ──Navigate──→ AT_LOCATION ──Pick──→ HOLDING
                        ↑                    │
                        └────Navigate────────┘
                                             │
                        Place────────────────┘
                          ↓
                        INIT
```

### 2.2 创新点2: 场景-规划对比预训练 (SPCL)

**动机**: 让模型学习场景表征与规划表征的对齐关系。

**正负样本构造**:
| 类型 | 样本 | 标签 |
|------|------|------|
| 正样本 | (scene_i, plan_i) | 1 |
| 随机负样本 | (scene_i, plan_j) | 0 |
| 顺序打乱负样本 | (scene_i, shuffle(plan_i)) | 0 |
| 不可执行负样本 | (scene_i, infeasible_plan) | 0 |

### 2.3 创新点3: Token级细粒度奖励分配 (FTCA)

**动机**: 规划序列中不同token的重要性不同。

**Token分类与权重**:
| Token类型 | 示例 | 权重 |
|-----------|------|------|
| 动作token | Navigate, Pick, Place | 1.0 |
| 目标token | Table, Apple, Basket | 0.8 |
| 结构token | (, ), , | 0.2 |

---

## 三、实现步骤

### Step 1: 数据准备与增强
**状态**: ✅ 已有基础 (data_build.py)

**任务清单**:
- [x] 基础数据生成脚本
- [ ] 负样本生成（用于对比学习）
- [ ] 数据格式统一化
- [ ] 划分训练/验证集

**输出**:
- src/data/dataset.py
- src/data/negative_sampler.py

---

### Step 2: 场景-规划对比预训练 (SPCL)
**状态**: 待开始

**任务清单**:
- [ ] 实现Scene Encoder（复用Qwen2-VL的Vision Encoder）
- [ ] 实现Plan Encoder（复用Qwen2-VL的Text Encoder）
- [ ] 实现InfoNCE对比学习Loss
- [ ] 预训练脚本

**输出**:
- src/models/spcl.py
- scripts/train_spcl.py
- configs/spcl_config.yaml

---

### Step 3: 监督微调 (SFT)
**状态**: 待开始

**任务清单**:
- [ ] 多模态数据加载器
- [ ] Qwen2-VL SFT配置
- [ ] 训练脚本
- [ ] 评估脚本

**输出**:
- scripts/train_sft.py
- configs/sft_config.yaml

---

### Step 4: H-PRM 奖励模型
**状态**: 待开始

**任务清单**:
- [ ] 动作奖励模型 (R_action) - 评估单个动作的场景匹配度
- [ ] 状态机转移检查 (R_transition) - 检查动作序列合法性
- [ ] 规划评估模型 (R_plan) - 评估整体规划完整性
- [ ] Token重要性权重 (FTCA)

**输出**:
- src/models/reward_model.py
- src/utils/state_machine.py

---

### Step 5: 强化学习训练 (PPO/GRPO)
**状态**: 待开始

**任务清单**:
- [ ] TRL PPO/GRPO配置
- [ ] H-PRM奖励函数集成
- [ ] 训练循环实现
- [ ] 检查点保存与评估

**输出**:
- scripts/train_rl.py
- configs/rl_config.yaml

---

### Step 6: 评估与消融实验
**状态**: 待开始

**任务清单**:
- [ ] 规划成功率评估
- [ ] 动作序列合法性检查
- [ ] 消融实验：移除各组件后的性能变化

**输出**:
- scripts/evaluate.py
- results/

---

## 四、目录结构

```
SVL/
├── data/
│   ├── raw/                    # 原始图像数据
│   ├── processed/              # 处理后的数据
│   ├── train.jsonl             # 训练数据
│   ├── val.jsonl               # 验证数据
│   └── negative_samples.jsonl  # 负样本数据
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # 数据集类
│   │   └── negative_sampler.py # 负样本生成
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── spcl.py             # 对比学习模块
│   │   └── reward_model.py     # H-PRM奖励模型
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py      # SFT训练
│   │   └── rl_trainer.py       # RL训练
│   │
│   └── utils/
│       ├── __init__.py
│       ├── state_machine.py    # 状态机
│       └── metrics.py          # 评估指标
│
├── scripts/
│   ├── train_spcl.py           # 对比预训练脚本
│   ├── train_sft.py            # SFT训练脚本
│   ├── train_rl.py             # RL训练脚本
│   └── evaluate.py             # 评估脚本
│
├── configs/
│   ├── spcl_config.yaml
│   ├── sft_config.yaml
│   └── rl_config.yaml
│
├── data_build.py               # 数据生成脚本（已有）
├── HARP_proposal.md            # 本文档
└── requirements.txt
```

---

## 五、实验设计

### 5.1 主实验对比
| 方法 | 规划成功率 | 动作合法率 | 平均步数 |
|------|-----------|-----------|---------|
| Qwen2-VL (Zero-shot) | - | - | - |
| Qwen2-VL + SFT | - | - | - |
| Qwen2-VL + SFT + RL (sparse reward) | - | - | - |
| **HARP (Ours)** | - | - | - |

### 5.2 消融实验
| 变体 | 规划成功率 | 说明 |
|------|-----------|------|
| HARP (full) | - | 完整方法 |
| w/o SPCL | - | 移除对比预训练 |
| w/o R_transition | - | 移除状态机约束奖励 |
| w/o FTCA | - | 移除Token级奖励分配 |
| w/o R_action | - | 移除动作合理性奖励 |

---

## 六、H-PRM奖励函数设计细节

### 6.1 R_action (动作合理性奖励)
- 输入: 场景图像 + 当前动作
- 输出: 该动作在当前场景下的合理性分数 [0, 1]
- 判断依据: 动作中涉及的物体/位置是否在图像中可见

### 6.2 R_transition (状态转移奖励)
- 基于有限状态机检查动作序列合法性
- 合法转移: +1.0
- 可行但冗余: +0.5 (如连续Navigate)
- 非法转移: -1.0 (如未Navigate就Pick)

### 6.3 R_plan (规划完整性奖励)
- 任务完成度: 规划是否能完成给定任务
- 效率性: 步骤数量是否最优
- 可执行性: 整体规划是否符合物理约束

### 6.4 FTCA (Token级奖励分配)
- 识别生成序列中的关键token
- 对不同类型token分配不同权重
- 使得模型更关注动作和目标物体的正确性

---

## 七、参考文献

1. Process Reward Model (PRM) - OpenAI
2. CLIP - Learning Transferable Visual Models From Natural Language Supervision
3. TRL - Transformer Reinforcement Learning Library
4. Qwen2-VL - Vision Language Model
5. PPO - Proximal Policy Optimization Algorithms
6. GRPO - Group Relative Policy Optimization

---

## 八、当前进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| Step 1 | 数据准备与增强 | 🔄 进行中 |
| Step 2 | SPCL对比预训练 | ⏳ 待开始 |
| Step 3 | SFT监督微调 | ⏳ 待开始 |
| Step 4 | H-PRM奖励模型 | ⏳ 待开始 |
| Step 5 | RL训练 | ⏳ 待开始 |
| Step 6 | 评估与消融 | ⏳ 待开始 |
