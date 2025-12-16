# HARP Training Guide

完整的训练流程指导，从数据生成到模型训练。

## 目录

1. [环境准备](#环境准备)
2. [数据生成](#数据生成)
3. [SFT训练](#sft训练)
4. [负样本生成](#负样本生成)
5. [GRPO训练](#grpo训练)
6. [模型评估](#模型评估)

---

## 环境准备

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n harp python=3.10
conda activate harp

# 安装依赖
pip install -r requirements.txt

# 如果需要最新的 Qwen2.5-VL 支持，从源码安装 transformers
pip install git+https://github.com/huggingface/transformers.git
```

### 2. 配置 API Key

```bash
# 创建 .env 文件
echo "OPENROUTER_API_KEY=your_api_key_here" > .env

# 或者直接设置环境变量
export OPENROUTER_API_KEY="your_api_key_here"
```

### 3. 准备目录结构

```bash
mkdir -p data outputs logs
mkdir -p bdv2  # 用于存放场景图像
```

---

## 数据生成

### Step 1: 准备场景图像

将你的场景图像放入 `./bdv2/` 目录：

```bash
# 图像命名规范：以 _0 结尾
# 例如：scene_001_0.png, scene_002_0.png
cp /path/to/your/images/*.png ./bdv2/
```

### Step 2: 运行数据生成脚本

```bash
# 生成任务描述和动作计划
python data_build.py
```

**输出文件**：
- `tasks.jsonl` - 任务描述
- `actions.jsonl` - 完整训练数据（图像 + 任务 + 计划）
- `processed/processed_images.txt` - 已处理图像列表

**数据格式示例**：
```json
{
  "image": "./bdv2/scene_001_0.png",
  "task": "Put the apple on the table into the basket",
  "plan": ["Navigate(Table)", "Pick(Apple)", "Navigate(Basket)", "Place(Basket)"]
}
```

### Step 3: 划分数据集

```bash
自动计算比例
python -c "
import json
import random
from pathlib import Path

# 读取数据
with open('actions.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# 打乱
random.seed(42)
random.shuffle(data)

# 划分
n = len(data)
train_end = int(0.8 * n)
val_end = int(0.9 * n)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# 保存
Path('data').mkdir(exist_ok=True)
for name, dataset in [('train', train_data), ('val', val_data), ('test', test_data)]:
    with open(f'data/sft_{name}.jsonl', 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')

print(f'Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')
"
```

---

## SFT训练

### Step 1: 配置训练参数

编辑 `configs/sft_config.yaml`：

```yaml
model:
  name: "Qwen/Qwen2.5-VL-7B-Instruct"  # 或其他 Qwen2.5-VL 变体
  freeze_vision_encoder: true
  use_lora: true

lora:
  rank: 64
  alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  dropout: 0.05

data:
  train_path: "data/sft_train.jsonl"
  val_path: "data/sft_val.jsonl"

training:
  output_dir: "outputs/sft"
  num_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 100
  max_length: 512
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
  bf16: true
  report_to: ["tensorboard", "wandb"]
  run_name: "sft_qwen2.5vl"
```

### Step 2: 运行 SFT 训练

```bash
# 单卡训练
python scripts/train_sft.py --config configs/sft_config.yaml

# 多卡训练（使用 DeepSpeed）
deepspeed --num_gpus=4 scripts/train_sft.py --config configs/sft_config.yaml
```

### Step 3: 监控训练

```bash
# 使用 TensorBoard
tensorboard --logdir outputs/sft/runs

# 或使用 Weights & Biases
# 访问 https://wandb.ai/your-project
```

**训练完成后**，模型保存在：
- `outputs/sft/final/` - 最终模型
- `outputs/sft/checkpoint-{step}/` - 中间检查点

---

## 负样本生成

SFT 训练完成后，使用训练好的模型生成负样本，用于 GRPO 训练。

### Step 1: 运行负样本生成

```bash
python scripts/generate_negatives.py \
    --model_path outputs/sft/final \
    --data_path data/sft_train.jsonl \
    --output_path data/negative_samples.jsonl \
    --num_samples_per_image 3 \
    --batch_size 4
```

**参数说明**：
- `--model_path`: SFT 训练好的模型路径
- `--data_path`: 训练数据（用于生成负样本）
- `--output_path`: 输出的负样本文件
- `--num_samples_per_image`: 每个图像生成多少个负样本
- `--batch_size`: 批次大小

**负样本类型**：
1. **格式错误**: 缺少逗号、括号不匹配
2. **动作错误**: 使用错误的动作类型
3. **状态转换错误**: 违反状态机规则（例如没有 Pick 就 Place）
4. **低效计划**: 包含冗余动作

**输出格式**：
```json
{
  "image": "./bdv2/scene_001_0.png",
  "task": "Put the apple on the table into the basket",
  "plan": ["Navigate(Table)", "Place(Basket)", "Pick(Apple)"],
  "error_type": "state_transition"
}
```

---

## GRPO训练

使用 H-PRM（分层过程奖励模型）和 GRPO 算法进行强化学习训练。

### Step 1: 配置 GRPO 参数

编辑 `configs/grpo_config.yaml`：

```yaml
model:
  base_model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
  sft_model_path: "outputs/sft/final"  # SFT 模型路径
  freeze_vision_encoder: true

data:
  train_path: "data/sft_train.jsonl"
  negative_path: "data/negative_samples.jsonl"

reward:
  w_format: 0.1      # 格式奖励权重
  w_action: 0.25     # 动作奖励权重
  w_transition: 0.25 # 状态转换奖励权重
  w_task: 0.25       # 任务完成奖励权重
  w_efficiency: 0.15 # 效率奖励权重
  use_ftca: true     # 使用细粒度 token 级别奖励分配

training:
  output_dir: "outputs/grpo"
  num_epochs: 2
  per_device_train_batch_size: 1
  num_candidates: 4           # 每个输入生成多少个候选
  temperature: 0.8            # 采样温度
  max_new_tokens: 128         # 最大生成长度
  learning_rate: 1e-6         # 比 SFT 更小的学习率
  kl_coef: 0.1                # KL 散度惩罚系数
  weight_decay: 0.01
  warmup_steps: 50
  logging_steps: 5
  save_steps: 200
  max_grad_norm: 1.0
  bf16: true
  report_to: ["tensorboard", "wandb"]
  run_name: "grpo_qwen2.5vl"
```

### Step 2: 运行 GRPO 训练

```bash
# 单卡训练
python scripts/train_grpo.py --config configs/grpo_config.yaml

# 多卡训练
deepspeed --num_gpus=4 scripts/train_grpo.py --config configs/grpo_config.yaml
```

### Step 3: 监控奖励和损失

```bash
# TensorBoard
tensorboard --logdir outputs/grpo/runs
```

**关键指标**：
- `train/loss` - GRPO 损失
- `train/reward` - 平均奖励
- `train/kl_div` - KL 散度（不要太大）
- `train/lr` - 学习率

**训练完成后**，模型保存在：
- `outputs/grpo/final/` - 最终模型
- `outputs/grpo/best/` - 最佳模型（最高奖励）
- `outputs/grpo/checkpoint-{step}/` - 中间检查点

---

## 模型评估

### Step 1: 运行评估脚本

```bash
python scripts/evaluate.py \
    --model_path outputs/grpo/final \
    --test_data data/sft_test.jsonl \
    --output_path results/evaluation.json \
    --batch_size 4
```

### Step 2: 查看评估结果

评估指标包括：

1. **格式正确率**：生成的计划格式正确的比例
2. **动作准确率**：动作类型和目标都正确的比例
3. **状态转换准确率**：符合状态机规则的比例
4. **任务完成率**：计划能够完成任务的比例
5. **平均步数**：计划的平均长度
6. **H-PRM 平均奖励**：所有测试样本的平均总奖励

**结果示例**：
```json
{
  "format_accuracy": 0.95,
  "action_accuracy": 0.87,
  "transition_accuracy": 0.92,
  "task_completion": 0.85,
  "avg_steps": 4.2,
  "avg_reward": 0.78
}
```

---

## 常见问题

### 1. OOM (Out of Memory)

**解决方案**：
- 减小 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用更小的模型（例如 Qwen2.5-VL-3B）
- 启用 DeepSpeed ZeRO-3

```yaml
# 添加到配置文件
training:
  deepspeed: "configs/ds_config_zero3.json"
```

### 2. 训练不稳定

**解决方案**：
- 降低学习率
- 增加 warmup steps
- 检查 KL 散度（如果太大，增加 `kl_coef`）
- 确保负样本质量

### 3. 奖励不增长

**解决方案**：
- 检查负样本是否足够多样化
- 调整奖励权重（增加 `w_task` 和 `w_efficiency`）
- 增加 `num_candidates`（更多探索）
- 降低 `temperature`（减少随机性）

### 4. 生成重复或无意义的计划

**解决方案**：
- 增加 SFT 训练的 epoch 数
- 检查训练数据质量
- 调整 GRPO 的 `temperature` 和 `kl_coef`
- 使用 nucleus sampling (`top_p`)

---

## 快速开始（完整流程）

```bash
# 1. 环境准备
conda create -n harp python=3.10
conda activate harp
pip install -r requirements.txt

# 2. 数据生成
export OPENROUTER_API_KEY="your_api_key"
python data_build.py

# 3. 数据划分
python -c "
import json, random
from pathlib import Path
with open('actions.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
random.seed(42)
random.shuffle(data)
n = len(data)
train_end, val_end = int(0.8*n), int(0.9*n)
Path('data').mkdir(exist_ok=True)
for name, d in [('train', data[:train_end]), ('val', data[train_end:val_end]), ('test', data[val_end:])]:
    with open(f'data/sft_{name}.jsonl', 'w') as f:
        for item in d: f.write(json.dumps(item) + '\n')
"

# 4. SFT 训练
python scripts/train_sft.py --config configs/sft_config.yaml

# 5. 负样本生成
python scripts/generate_negatives.py \
    --model_path outputs/sft/final \
    --data_path data/sft_train.jsonl \
    --output_path data/negative_samples.jsonl

# 6. GRPO 训练
python scripts/train_grpo.py --config configs/grpo_config.yaml

# 7. 评估
python scripts/evaluate.py \
    --model_path outputs/grpo/final \
    --test_data data/sft_test.jsonl
```
