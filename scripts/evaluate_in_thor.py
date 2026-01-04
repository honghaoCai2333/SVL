"""
AI2-THOR 在线评估脚本

使用 test_809.json 在 AI2-THOR 模拟器中评估模型

评估流程：
1. 加载测试任务
2. 初始化 AI2-THOR 场景
3. 获取初始观察图像
4. 模型预测下一步动作
5. 在模拟器中执行动作
6. 重复 3-5 直到任务完成或超时
7. 计算评估指标

Usage:
    python scripts/evaluate_in_thor.py \
        --test_json /path/to/test_809.json \
        --agent_positions /path/to/agent_positions.json \
        --model_path /path/to/your/model \
        --output_dir results/
"""

import json
import os
import argparse
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

# AI2-THOR 需要单独安装: pip install ai2thor
try:
    from ai2thor.controller import Controller
    THOR_AVAILABLE = True
except ImportError:
    THOR_AVAILABLE = False
    print("Warning: ai2thor not installed. Run: pip install ai2thor")


@dataclass
class EvalConfig:
    """评估配置"""
    max_steps: int = 50  # 每个任务最大步数
    action_timeout: float = 30.0  # 动作超时时间（秒）
    image_size: int = 512  # 观察图像大小
    field_of_view: int = 90  # 视野角度
    grid_size: float = 0.25  # 移动步长
    rotation_degrees: int = 90  # 旋转角度
    visibility_distance: float = 1.5  # 可见距离


class ActionMapper:
    """将模型输出的动作映射到 AI2-THOR 动作"""
    
    @staticmethod
    def parse_action(action_str: str) -> Tuple[str, Optional[str]]:
        """
        解析动作字符串
        'Navigate(CounterTop)' -> ('Navigate', 'CounterTop')
        'TaskCompleted()' -> ('TaskCompleted', None)
        """
        if '(' in action_str and ')' in action_str:
            action_type = action_str.split('(')[0]
            target = action_str.split('(')[1].rstrip(')')
            return action_type, target if target else None
        return action_str, None
    
    @staticmethod
    def to_thor_action(action_type: str, target: Optional[str], 
                       controller: 'Controller') -> Dict[str, Any]:
        """
        将解析后的动作转换为 AI2-THOR 可执行的动作
        
        Args:
            action_type: 动作类型 (Navigate, Pick, Place, Open, Close, Toggle)
            target: 目标对象类型
            controller: AI2-THOR controller
        
        Returns:
            AI2-THOR 动作字典
        """
        if action_type == 'Navigate':
            # 导航到目标对象
            # 需要找到目标对象的位置并移动过去
            return {
                'action': 'ObjectNavExpertAction',
                'objectType': target
            }
        
        elif action_type == 'Pick':
            # 拾取对象
            obj = ActionMapper._find_object_by_type(controller, target)
            if obj:
                return {
                    'action': 'PickupObject',
                    'objectId': obj['objectId'],
                    'forceAction': True
                }
        
        elif action_type == 'Place':
            # 放置对象
            obj = ActionMapper._find_object_by_type(controller, target)
            if obj:
                return {
                    'action': 'PutObject',
                    'objectId': obj['objectId'],
                    'forceAction': True
                }
        
        elif action_type == 'Open':
            # 打开对象（如抽屉、冰箱）
            obj = ActionMapper._find_object_by_type(controller, target)
            if obj:
                return {
                    'action': 'OpenObject',
                    'objectId': obj['objectId'],
                    'forceAction': True
                }
        
        elif action_type == 'Close':
            # 关闭对象
            obj = ActionMapper._find_object_by_type(controller, target)
            if obj:
                return {
                    'action': 'CloseObject',
                    'objectId': obj['objectId'],
                    'forceAction': True
                }
        
        elif action_type == 'Toggle':
            # 切换对象状态（如开关灯）
            obj = ActionMapper._find_object_by_type(controller, target)
            if obj:
                return {
                    'action': 'ToggleObjectOn' if not obj.get('isToggled') else 'ToggleObjectOff',
                    'objectId': obj['objectId'],
                    'forceAction': True
                }
        
        elif action_type == 'TaskCompleted':
            return {'action': 'Done'}
        
        return {'action': 'Pass'}  # 无效动作时跳过
    
    @staticmethod
    def _find_object_by_type(controller: 'Controller', object_type: str) -> Optional[Dict]:
        """根据对象类型找到最近的可见对象"""
        objects = controller.last_event.metadata['objects']
        candidates = [
            obj for obj in objects 
            if obj['objectType'] == object_type and obj['visible']
        ]
        
        if candidates:
            # 返回最近的对象
            agent_pos = controller.last_event.metadata['agent']['position']
            candidates.sort(key=lambda o: (
                (o['position']['x'] - agent_pos['x']) ** 2 +
                (o['position']['z'] - agent_pos['z']) ** 2
            ))
            return candidates[0]
        return None


class AI2THOREvaluator:
    """AI2-THOR 评估器"""
    
    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()
        self.controller = None
        self.action_mapper = ActionMapper()
    
    def initialize(self):
        """初始化 AI2-THOR 控制器"""
        if not THOR_AVAILABLE:
            raise RuntimeError("ai2thor not installed")
        
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=self.config.visibility_distance,
            scene="FloorPlan1",  # 初始场景，后续会切换
            gridSize=self.config.grid_size,
            rotateStepDegrees=self.config.rotation_degrees,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=self.config.image_size,
            height=self.config.image_size,
            fieldOfView=self.config.field_of_view
        )
        print("AI2-THOR initialized successfully")
    
    def reset_scene(self, scene: str, agent_position: Dict = None) -> Any:
        """
        重置场景
        
        Args:
            scene: 场景名称 (如 'FloorPlan1')
            agent_position: 智能体初始位置（可选）
        
        Returns:
            初始观察图像
        """
        self.controller.reset(scene)
        
        if agent_position:
            # 设置智能体位置
            self.controller.step(
                action="Teleport",
                position=agent_position.get('position', {}),
                rotation=agent_position.get('rotation', {}),
                horizon=agent_position.get('horizon', 0)
            )
        
        return self._get_observation()
    
    def step(self, action_str: str) -> Tuple[Any, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action_str: 模型输出的动作字符串 (如 'Navigate(CounterTop)')
        
        Returns:
            (observation, done, info)
        """
        action_type, target = self.action_mapper.parse_action(action_str)
        
        if action_type == 'TaskCompleted':
            return self._get_observation(), True, {'success': True}
        
        thor_action = self.action_mapper.to_thor_action(
            action_type, target, self.controller
        )
        
        event = self.controller.step(**thor_action)
        
        info = {
            'success': event.metadata['lastActionSuccess'],
            'error': event.metadata.get('errorMessage', ''),
            'action_type': action_type,
            'target': target
        }
        
        return self._get_observation(), False, info
    
    def _get_observation(self):
        """获取当前观察图像"""
        return self.controller.last_event.frame
    
    def close(self):
        """关闭控制器"""
        if self.controller:
            self.controller.stop()


class ModelInterface:
    """模型接口基类 - 需要根据实际模型实现"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型 - 根据实际模型框架实现"""
        # TODO: 实现模型加载
        # 例如使用 transformers:
        # from transformers import AutoModelForCausalLM, AutoProcessor
        # self.model = AutoModelForCausalLM.from_pretrained(model_path)
        # self.processor = AutoProcessor.from_pretrained(model_path)
        pass
    
    def predict(self, image, task: str, action_history: List[str]) -> str:
        """
        预测下一步动作
        
        Args:
            image: 当前观察图像 (numpy array)
            task: 任务描述
            action_history: 历史动作列表
        
        Returns:
            预测的动作字符串 (如 'Navigate(CounterTop)')
        """
        # TODO: 实现模型推理
        # 1. 构建 prompt
        # 2. 处理图像
        # 3. 模型推理
        # 4. 解析输出
        raise NotImplementedError("Please implement predict method")


def evaluate_task(evaluator: AI2THOREvaluator, 
                  model: ModelInterface,
                  task: Dict[str, Any],
                  agent_position: Dict = None,
                  max_steps: int = 50) -> Dict[str, Any]:
    """
    评估单个任务
    
    Args:
        evaluator: AI2-THOR 评估器
        model: 模型接口
        task: 任务数据
        agent_position: 智能体初始位置
        max_steps: 最大步数
    
    Returns:
        评估结果
    """
    scene = task['scene']
    task_desc = task.get('taskname', task.get('taskquery', ''))
    gt_actions = []
    
    # 提取 ground truth 动作
    if 'task_metadata' in task and 'actions' in task['task_metadata']:
        for act in task['task_metadata']['actions']:
            action_type = act.get('action', '')
            object_type = act.get('objectType', '')
            
            if action_type == 'navigate to':
                gt_actions.append(f"Navigate({object_type})")
            elif action_type == 'pickup':
                gt_actions.append(f"Pick({object_type})")
            elif action_type == 'put':
                gt_actions.append(f"Place({object_type})")
            elif action_type == 'open':
                gt_actions.append(f"Open({object_type})")
            elif action_type == 'close':
                gt_actions.append(f"Close({object_type})")
            elif action_type == 'toggle':
                gt_actions.append(f"Toggle({object_type})")
            elif action_type == 'end':
                gt_actions.append("TaskCompleted()")
    
    # 重置场景
    obs = evaluator.reset_scene(scene, agent_position)
    
    # 执行循环
    action_history = []
    step = 0
    done = False
    success = False
    
    while step < max_steps and not done:
        # 模型预测
        predicted_action = model.predict(obs, task_desc, action_history)
        
        # 执行动作
        obs, done, info = evaluator.step(predicted_action)
        
        action_history.append(predicted_action)
        step += 1
        
        if done:
            success = info.get('success', False)
    
    # 计算指标
    result = {
        'scene': scene,
        'task': task_desc,
        'gt_actions': gt_actions,
        'predicted_actions': action_history,
        'steps': step,
        'success': success,
        'action_accuracy': calculate_action_accuracy(gt_actions, action_history),
        'task_completed': 'TaskCompleted()' in action_history
    }
    
    return result


def calculate_action_accuracy(gt_actions: List[str], pred_actions: List[str]) -> float:
    """计算动作准确率"""
    if not gt_actions:
        return 0.0
    
    correct = 0
    for i, gt in enumerate(gt_actions):
        if i < len(pred_actions) and pred_actions[i] == gt:
            correct += 1
    
    return correct / len(gt_actions)


def run_evaluation(test_json_path: str,
                   agent_positions_path: str,
                   model_path: str,
                   output_dir: str,
                   config: EvalConfig = None):
    """
    运行完整评估
    
    Args:
        test_json_path: 测试集 JSON 文件路径
        agent_positions_path: 智能体位置文件路径
        model_path: 模型路径
        output_dir: 输出目录
        config: 评估配置
    """
    config = config or EvalConfig()
    
    # 加载测试数据
    print(f"Loading test data from {test_json_path}...")
    with open(test_json_path, 'r') as f:
        test_tasks = json.load(f)
    print(f"Loaded {len(test_tasks)} test tasks")
    
    # 加载智能体位置
    agent_positions = {}
    if os.path.exists(agent_positions_path):
        with open(agent_positions_path, 'r') as f:
            agent_positions = json.load(f)
        print(f"Loaded agent positions for {len(agent_positions)} scenes")
    
    # 初始化评估器和模型
    evaluator = AI2THOREvaluator(config)
    evaluator.initialize()
    
    model = ModelInterface(model_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行评估
    results = []
    for i, task in enumerate(test_tasks):
        print(f"\nEvaluating task {i+1}/{len(test_tasks)}: {task.get('taskname', '')[:50]}...")
        
        scene = task['scene']
        agent_pos = agent_positions.get(scene, None)
        
        try:
            result = evaluate_task(evaluator, model, task, agent_pos, config.max_steps)
            results.append(result)
            
            print(f"  Steps: {result['steps']}, Success: {result['success']}, "
                  f"Action Acc: {result['action_accuracy']:.2%}")
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'scene': scene,
                'task': task.get('taskname', ''),
                'error': str(e),
                'success': False
            })
    
    # 关闭评估器
    evaluator.close()
    
    # 计算总体指标
    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    completed = sum(1 for r in results if r.get('task_completed', False))
    avg_accuracy = sum(r.get('action_accuracy', 0) for r in results) / total if total > 0 else 0
    avg_steps = sum(r.get('steps', 0) for r in results) / total if total > 0 else 0
    
    summary = {
        'total_tasks': total,
        'successful_tasks': successful,
        'success_rate': successful / total if total > 0 else 0,
        'task_completion_rate': completed / total if total > 0 else 0,
        'average_action_accuracy': avg_accuracy,
        'average_steps': avg_steps
    }
    
    # 保存结果
    output_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'summary': summary,
            'results': results
        }, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Evaluation Summary:")
    print(f"  Total Tasks: {total}")
    print(f"  Success Rate: {summary['success_rate']:.2%}")
    print(f"  Task Completion Rate: {summary['task_completion_rate']:.2%}")
    print(f"  Average Action Accuracy: {avg_accuracy:.2%}")
    print(f"  Average Steps: {avg_steps:.1f}")
    print(f"\nResults saved to: {output_path}")
    
    return summary, results


def main():
    parser = argparse.ArgumentParser(description='Evaluate model in AI2-THOR')
    
    parser.add_argument('--test_json', type=str, required=True,
                        help='Path to test_809.json')
    parser.add_argument('--agent_positions', type=str, default='',
                        help='Path to agent_positions.json')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for results')
    
    # 评估配置
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Maximum steps per task')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Observation image size')
    
    args = parser.parse_args()
    
    config = EvalConfig(
        max_steps=args.max_steps,
        image_size=args.image_size
    )
    
    run_evaluation(
        args.test_json,
        args.agent_positions,
        args.model_path,
        args.output_dir,
        config
    )


if __name__ == '__main__':
    main()


# ============== 示例：如何实现模型接口 ==============
"""
以 Qwen2.5-VL 为例：

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

class Qwen2VLModel(ModelInterface):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
    
    def predict(self, image, task: str, action_history: List[str]) -> str:
        # 构建 prompt
        history_str = ", ".join(action_history) if action_history else "None"
        prompt = f'''You are a robot in a room. Complete the task step by step.

Task: "{task}"
Action History: [{history_str}]

Available Actions:
- Navigate(ObjectType): Move to the object
- Pick(ObjectType): Pick up the object
- Place(ObjectType): Place held object on/in target
- Open(ObjectType): Open a container
- Close(ObjectType): Close a container
- Toggle(ObjectType): Toggle object state
- TaskCompleted(): Mark task as done

Analyze the image and output your next action:'''

        # 处理输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(image)},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[Image.fromarray(image)],
            return_tensors="pt"
        ).to(self.model.device)
        
        # 生成
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # 解析动作
        action = self._parse_action(response)
        return action
    
    def _parse_action(self, response: str) -> str:
        # 从模型输出中提取动作
        import re
        patterns = [
            r'(Navigate|Pick|Place|Open|Close|Toggle)\([^)]+\)',
            r'TaskCompleted\(\)'
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(0)
        return "TaskCompleted()"  # 默认结束
"""
