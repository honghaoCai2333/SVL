# AI2-THOR 在线评估指南

本指南说明如何使用 `test_809.json` 测试集在 AI2-THOR 模拟器中评估模型。

## 概述

测试集 `test_809.json` 只包含任务描述和 ground truth 动作序列，**不包含图片**。评估需要：
1. 在 AI2-THOR 模拟器中加载对应场景
2. 模型根据实时观察图像预测动作
3. 在模拟器中执行动作并获取新图像
4. 与 ground truth 对比计算指标

## 环境准备

### 1. 安装 AI2-THOR

```bash
# 安装 AI2-THOR
pip install ai2thor

# 首次运行会自动下载模拟器资源（约 1GB）
python -c "from ai2thor.controller import Controller; c = Controller(); c.stop()"
```

### 2. 安装其他依赖

```bash
pip install torch transformers pillow numpy
```

### 3. 准备测试数据

确保以下文件存在：
- `test_809.json`: 测试任务（809个任务）
- `agent_positions.json`: 智能体初始位置（可选）

## 评估流程

### 整体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        评估循环                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                                               │
│   │ 加载测试任务 │ ← test_809.json                               │
│   └──────┬──────┘                                               │
│          ↓                                                      │
│   ┌─────────────────┐                                           │
│   │ 初始化 AI2-THOR │ ← 加载场景 (FloorPlan1, etc.)              │
│   └──────┬──────────┘                                           │
│          ↓                                                      │
│   ┌─────────────────┐                                           │
│   │ 获取初始观察图像 │ ← 第一人称视角 RGB 图像                    │
│   └──────┬──────────┘                                           │
│          ↓                                                      │
│   ┌──────────────────────────────────────┐                      │
│   │           模型推理循环                 │                      │
│   │  ┌────────────────────────────────┐  │                      │
│   │  │ 输入: 图像 + 任务 + 历史动作    │  │                      │
│   │  │        ↓                       │  │                      │
│   │  │ 模型预测下一步动作              │  │                      │
│   │  │        ↓                       │  │                      │
│   │  │ 在模拟器中执行动作              │  │                      │
│   │  │        ↓                       │  │                      │
│   │  │ 获取新的观察图像                │  │                      │
│   │  │        ↓                       │  │                      │
│   │  │ 检查: 任务完成 or 达到最大步数？ │  │                      │
│   │  └────────────────────────────────┘  │                      │
│   └──────────────────────────────────────┘                      │
│          ↓                                                      │
│   ┌─────────────────────┐                                       │
│   │ 与 GT 对比计算指标   │                                       │
│   └─────────────────────┘                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 详细步骤

#### Step 1: 加载测试任务

```python
import json

# 加载测试集
with open('test_809.json', 'r') as f:
    test_tasks = json.load(f)

# 每个任务包含:
task = test_tasks[0]
print(task['scene'])      # 'FloorPlan1' - 场景ID
print(task['taskname'])   # 任务描述
print(task['task_metadata']['actions'])  # Ground truth 动作序列
```

#### Step 2: 初始化 AI2-THOR 模拟器

```python
from ai2thor.controller import Controller

# 创建控制器
controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene="FloorPlan1",  # 初始场景
    gridSize=0.25,
    rotateStepDegrees=90,
    width=512,
    height=512,
    fieldOfView=90
)

# 切换到指定场景
controller.reset(task['scene'])
```

#### Step 3: 获取观察图像

```python
# 获取当前帧（RGB图像）
observation = controller.last_event.frame  # numpy array (H, W, 3)

# 保存图像（可选）
from PIL import Image
img = Image.fromarray(observation)
img.save('observation.png')
```

#### Step 4: 模型预测动作

```python
def predict_action(model, image, task_description, action_history):
    """
    模型推理
    
    Args:
        model: 训练好的 VLM 模型
        image: 观察图像 (numpy array)
        task_description: 任务描述字符串
        action_history: 历史动作列表
    
    Returns:
        predicted_action: 预测的动作字符串，如 'Navigate(CounterTop)'
    """
    # 构建输入 prompt
    prompt = f'''Task: "{task_description}"
Action History: {action_history}
Based on the current observation, what is the next action?'''
    
    # 模型推理（具体实现取决于你的模型）
    predicted_action = model.generate(image, prompt)
    
    return predicted_action
```

#### Step 5: 在模拟器中执行动作

```python
def execute_action(controller, action_str):
    """
    将模型输出转换为 AI2-THOR 动作并执行
    
    Args:
        controller: AI2-THOR controller
        action_str: 模型输出的动作，如 'Navigate(CounterTop)'
    
    Returns:
        success: 动作是否成功
        new_observation: 新的观察图像
    """
    # 解析动作
    action_type = action_str.split('(')[0]  # 'Navigate'
    target = action_str.split('(')[1].rstrip(')')  # 'CounterTop'
    
    if action_type == 'Navigate':
        # 导航到目标对象
        event = controller.step(
            action='ObjectNavExpertAction',
            objectType=target
        )
    
    elif action_type == 'Pick':
        # 找到目标对象并拾取
        obj = find_object(controller, target)
        event = controller.step(
            action='PickupObject',
            objectId=obj['objectId'],
            forceAction=True
        )
    
    elif action_type == 'Place':
        obj = find_object(controller, target)
        event = controller.step(
            action='PutObject',
            objectId=obj['objectId'],
            forceAction=True
        )
    
    elif action_type == 'Open':
        obj = find_object(controller, target)
        event = controller.step(
            action='OpenObject',
            objectId=obj['objectId'],
            forceAction=True
        )
    
    elif action_type == 'Close':
        obj = find_object(controller, target)
        event = controller.step(
            action='CloseObject',
            objectId=obj['objectId'],
            forceAction=True
        )
    
    elif action_type == 'Toggle':
        obj = find_object(controller, target)
        event = controller.step(
            action='ToggleObjectOn',
            objectId=obj['objectId'],
            forceAction=True
        )
    
    elif action_type == 'TaskCompleted':
        return True, controller.last_event.frame, True  # 任务完成
    
    success = event.metadata['lastActionSuccess']
    new_observation = controller.last_event.frame
    
    return success, new_observation, False


def find_object(controller, object_type):
    """根据类型找到可见的对象"""
    objects = controller.last_event.metadata['objects']
    for obj in objects:
        if obj['objectType'] == object_type and obj['visible']:
            return obj
    return None
```

#### Step 6: 完整评估循环

```python
def evaluate_task(controller, model, task, max_steps=50):
    """评估单个任务"""
    
    # 重置场景
    controller.reset(task['scene'])
    
    # 获取初始观察
    observation = controller.last_event.frame
    task_desc = task['taskname']
    
    # 评估循环
    action_history = []
    step = 0
    done = False
    
    while step < max_steps and not done:
        # 模型预测
        action = predict_action(model, observation, task_desc, action_history)
        
        # 执行动作
        success, observation, done = execute_action(controller, action)
        
        action_history.append(action)
        step += 1
        
        print(f"Step {step}: {action} (success: {success})")
    
    # 计算指标
    gt_actions = extract_gt_actions(task)
    accuracy = calculate_accuracy(gt_actions, action_history)
    
    return {
        'predicted_actions': action_history,
        'gt_actions': gt_actions,
        'accuracy': accuracy,
        'steps': step,
        'task_completed': done
    }
```

## 运行评估脚本

```bash
# 使用提供的评估脚本
python scripts/evaluate_in_thor.py \
    --test_json /path/to/test_809.json \
    --agent_positions /path/to/agent_positions.json \
    --model_path /path/to/your/trained/model \
    --output_dir results/ \
    --max_steps 50
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **Success Rate (SR)** | 成功完成任务的比例 |
| **Task Completion Rate** | 模型输出 TaskCompleted() 的比例 |
| **Action Accuracy** | 预测动作与 GT 动作的匹配率 |
| **Average Steps** | 平均执行步数 |
| **SPL** | Success weighted by Path Length |

## 注意事项

1. **GPU 要求**: AI2-THOR 需要 GPU 渲染，建议至少 4GB 显存
2. **首次运行**: 会下载约 1GB 的模拟器资源
3. **导航实现**: `ObjectNavExpertAction` 是简化的导航，实际可能需要更复杂的路径规划
4. **错误处理**: 某些动作可能因为物理约束失败，需要处理这些情况

## 相关资源

- [AI2-THOR 官方文档](https://ai2thor.allenai.org/ithor/documentation/)
- [Embodied-Reasoner 论文](https://arxiv.org/abs/2503.21696)
- [数据集 HuggingFace](https://huggingface.co/datasets/zwq2018/embodied_reasoner)
