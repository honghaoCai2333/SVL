"""
将模仿学习格式的数据转换为 Step-by-step SFT 训练格式

每一步作为一个独立的训练样本：
- 输入：当前图像 + 任务描述 + 历史动作
- 输出：思考过程 + 下一步动作

支持两种数据源格式：
1. 文件夹格式 (navigate1open1pickup0): 每个任务一个文件夹+JSON
2. 大JSON格式 (train_multiturn_9390.json): 一个JSON文件包含所有任务

输出格式:
{
    "image": "data/images/xxx/xxx.png",
    "task": "任务描述",
    "action_history": ["Navigate(CounterTop)", "Pick(Apple)"],
    "thinking": "思考过程",
    "next_action": "Navigate(Fridge)",
    "step": 2
}

Usage:
    # 文件夹格式
    python scripts/convert_to_sft.py \
        --input_dir /path/to/data \
        --output_path data/sft_train.jsonl \
        --copy_images
    
    # 大JSON格式
    python scripts/convert_to_sft.py \
        --input_json /path/to/train_multiturn_9390.json \
        --image_base_dir /path/to/embodied_reasoner \
        --output_path data/sft_train.jsonl \
        --copy_images
"""

import json
import os
import re
import shutil
import argparse
from typing import List, Dict, Any, Tuple
from glob import glob


# ============== 工具函数 ==============

def extract_action_from_decision(text: str) -> str:
    """从 <DecisionMaking>...</DecisionMaking> 中提取动作"""
    match = re.search(r'<DecisionMaking>(.*?)</DecisionMaking>', text)
    if match:
        return match.group(1).strip()
    return ""


def extract_thinking_from_content(content: str) -> str:
    """从 assistant 的 content 中提取 thinking 部分（<DecisionMaking> 之前的内容）"""
    match = re.search(r'<DecisionMaking>', content)
    if match:
        thinking = content[:match.start()].strip()
        # 清理一些常见的结尾词
        thinking = re.sub(r"(Okay,\s*I('ve| think|'ll).*?|Hmm.*?|So,.*?|Alright.*?)$", "", thinking, flags=re.IGNORECASE).strip()
        return thinking
    return content.strip()


def extract_thinking_from_trajectory(trajectory: List[str]) -> List[str]:
    """从 trajectory 中提取 <Thinking>...</Thinking> 内容"""
    thinkings = []
    for item in trajectory:
        thinking_match = re.search(r'<Thinking>(.*?)</Thinking>', item, re.DOTALL)
        if thinking_match:
            thinkings.append(thinking_match.group(1).strip())
    return thinkings


def normalize_action(action: str) -> str:
    """
    标准化动作格式
    'navigate to CounterTop' -> 'Navigate(CounterTop)'
    'pickup Apple' -> 'Pick(Apple)'
    'put in Fridge' -> 'Place(Fridge)'
    'open Fridge' -> 'Open(Fridge)'
    'close Fridge' -> 'Close(Fridge)'
    'end' -> 'TaskCompleted()'
    """
    action = action.strip()
    
    # 处理结束动作
    end_keywords = ['end', 'end task', 'task completed', 'task complete', 'done', 
                    'finish', 'finished', 'complete task', 'conclude task']
    if action.lower() in end_keywords:
        return "TaskCompleted()"
    if 'task' in action.lower() and any(w in action.lower() for w in ['complete', 'end', 'done', 'finish', 'conclude']):
        return "TaskCompleted()"
    
    # 处理各种动作格式
    if action.startswith('navigate to '):
        return f"Navigate({action.replace('navigate to ', '')})"
    elif action.startswith('pickup '):
        return f"Pick({action.replace('pickup ', '')})"
    elif action.startswith('put in '):
        return f"Place({action.replace('put in ', '')})"
    elif action.startswith('put '):
        return f"Place({action.replace('put ', '')})"
    elif action.startswith('open '):
        return f"Open({action.replace('open ', '')})"
    elif action.startswith('close '):
        return f"Close({action.replace('close ', '')})"
    elif action.startswith('toggle '):
        return f"Toggle({action.replace('toggle ', '')})"
    elif action == 'observe':
        return "Observe()"
    elif action == 'move forward':
        return "MoveForward()"
    
    return action


# ============== 常量 ==============

INITIAL_THINKING = "I will carefully observe the environment, analyze the current scene and the task requirements, then make the correct action to accomplish the goal step by step."
COMPLETION_THINKING = "I have successfully completed all the required subtasks. The task has been accomplished as instructed."


# ============== 大JSON格式转换 (train_multiturn_9390.json) ==============

def convert_multiturn_stepwise(data: Dict[str, Any], image_base_dir: str, 
                                copy_images: bool = False, image_output_dir: str = None) -> Tuple[List[Dict[str, Any]], set]:
    """
    将多轮对话格式的数据转换为 stepwise SFT 格式
    
    Args:
        data: 单个任务数据，包含 messages 和 images
        image_base_dir: 图片基础目录
        copy_images: 是否复制图片
        image_output_dir: 图片输出目录
    
    Returns:
        (samples 列表, 复制的图片集合)
    """
    samples = []
    copied_images = set()
    
    messages = data.get('messages', [])
    images = data.get('images', [])
    
    if not messages or not images:
        return samples, copied_images
    
    # 提取任务描述
    task = ""
    for msg in messages:
        if msg['role'] == 'user':
            task_match = re.search(r'Task:\s*"([^"]+)"', msg['content'])
            if task_match:
                task = task_match.group(1)
                break
    
    if not task:
        return samples, copied_images
    
    # 解析对话，提取动作和 thinking
    action_history = []
    image_idx = 0
    step = 0
    
    for msg in messages:
        if msg['role'] == 'assistant':
            content = msg['content']
            
            # 提取动作
            action = extract_action_from_decision(content)
            if not action:
                continue
            
            normalized_action = normalize_action(action)
            
            # 提取 thinking
            thinking = INITIAL_THINKING if step == 0 else extract_thinking_from_content(content)
            
            # 获取对应的图片
            relative_img_path = _process_image(
                images, image_idx, image_base_dir, copy_images, image_output_dir, copied_images
            )
            
            sample = {
                "image": relative_img_path,
                "task": task,
                "action_history": action_history.copy(),
                "thinking": thinking,
                "next_action": normalized_action,
                "step": step
            }
            samples.append(sample)
            
            action_history.append(normalized_action)
            image_idx += 1
            step += 1
    
    # 添加 TaskCompleted 结束样本（如果需要）
    if samples and samples[-1]['next_action'] != 'TaskCompleted()':
        relative_img_path = _process_image(
            images, image_idx, image_base_dir, copy_images, image_output_dir, copied_images
        )
        if relative_img_path is None and samples:
            relative_img_path = samples[-1]['image']
        
        samples.append({
            "image": relative_img_path,
            "task": task,
            "action_history": action_history.copy(),
            "thinking": COMPLETION_THINKING,
            "next_action": "TaskCompleted()",
            "step": step
        })
    
    return samples, copied_images


def _process_image(images: List[str], image_idx: int, image_base_dir: str,
                   copy_images: bool, image_output_dir: str, copied_images: set) -> str:
    """处理图片路径，可选复制图片"""
    if image_idx >= len(images):
        return None
    
    img_path = images[image_idx]
    
    # 处理相对路径
    if img_path.startswith('./'):
        img_path = img_path[2:]
    # 修正路径：data/images/xxx -> data/xxx
    if img_path.startswith('data/images/'):
        img_path = img_path.replace('data/images/', 'data/')
    
    full_img_path = os.path.join(image_base_dir, img_path)
    
    if copy_images and image_output_dir and os.path.exists(full_img_path):
        path_parts = img_path.split('/')
        if len(path_parts) >= 3:
            subfolder = path_parts[-2]
            img_filename = path_parts[-1]
            
            dst_subfolder = os.path.join(image_output_dir, subfolder)
            os.makedirs(dst_subfolder, exist_ok=True)
            dst_path = os.path.join(dst_subfolder, img_filename)
            
            if full_img_path not in copied_images:
                shutil.copy2(full_img_path, dst_path)
                copied_images.add(full_img_path)
            
            return os.path.join('data/images', subfolder, img_filename)
        return img_path
    
    return full_img_path if os.path.exists(full_img_path) else img_path


def process_multiturn_json(input_json: str, output_path: str, image_base_dir: str,
                           copy_images: bool = False, image_output_dir: str = None):
    """处理大JSON格式的数据文件 (train_multiturn_9390.json)"""
    print(f"Loading {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print(f"Found {len(data_list)} tasks")
    
    if copy_images and image_output_dir:
        os.makedirs(image_output_dir, exist_ok=True)
        print(f"Images will be copied to: {image_output_dir}")
    
    all_samples = []
    all_copied_images = set()
    
    for i, data in enumerate(data_list):
        if (i + 1) % 1000 == 0:
            print(f"Processing {i + 1}/{len(data_list)}...")
        
        try:
            samples, copied = convert_multiturn_stepwise(
                data, image_base_dir, copy_images, image_output_dir
            )
            all_samples.extend(samples)
            all_copied_images.update(copied)
        except Exception as e:
            print(f"Error processing task {i}: {e}")
            continue
    
    # 保存到 JSONL 文件
    _save_jsonl(all_samples, output_path)
    
    print(f"Converted {len(all_samples)} samples to {output_path}")
    if copy_images:
        print(f"Copied {len(all_copied_images)} unique images")
    
    return all_samples


# ============== 文件夹格式转换 (navigate1open1pickup0) ==============

def convert_stepwise_from_metadata(data: Dict[str, Any], base_dir: str, use_local_images: bool = True) -> List[Dict[str, Any]]:
    """
    将文件夹格式的数据转换为 stepwise SFT 格式
    使用 task_metadata 中的 ground truth 动作序列
    """
    samples = []
    task = data.get('taskname', '')
    images = data.get('images', [])
    trajectory = data.get('trajectory', [])
    
    # 从 trajectory 中提取 thinking
    thinkings = extract_thinking_from_trajectory(trajectory)
    
    # 从 task_metadata 获取 ground truth 动作序列
    if 'task_metadata' not in data or 'actions' not in data['task_metadata']:
        return samples
    
    # 标准化动作
    normalized_actions = []
    for act in data['task_metadata']['actions']:
        action_type = act.get('action', '')
        object_type = act.get('objectType', '')
        
        action_map = {
            'navigate to': f"Navigate({object_type})",
            'pickup': f"Pick({object_type})",
            'put': f"Place({object_type})",
            'open': f"Open({object_type})",
            'close': f"Close({object_type})",
            'toggle': f"Toggle({object_type})"
        }
        
        if action_type in action_map:
            normalized_actions.append(action_map[action_type])
    
    if not normalized_actions:
        return samples
    
    # 获取本地图像列表
    local_images = []
    if use_local_images:
        for f in sorted(os.listdir(base_dir)):
            if f.endswith('.png') or f.endswith('.jpg'):
                local_images.append(os.path.join(base_dir, f))
        # 按文件名中的数字排序
        local_images.sort(key=lambda x: int(re.search(r'_(\d+)_', os.path.basename(x)).group(1)) if re.search(r'_(\d+)_', os.path.basename(x)) else 0)
    
    err_action_num = data.get('err_action_num', 0)
    action_history = []
    
    # 生成每一步的样本
    for i, current_action in enumerate(normalized_actions):
        # 计算图像索引
        image_idx = 0 if i == 0 else err_action_num + i - 1
        
        # 获取图像
        current_image = _get_image_path(local_images, images, image_idx, base_dir, use_local_images)
        
        # 确定 thinking
        if i == 0:
            thinking = INITIAL_THINKING
        elif i - 1 < len(thinkings):
            thinking = thinkings[i - 1]
        else:
            thinking = ""
        
        samples.append({
            "image": current_image,
            "task": task,
            "action_history": action_history.copy(),
            "thinking": thinking,
            "next_action": current_action,
            "scene": data.get('scene', ''),
            "step": i
        })
        
        action_history.append(current_action)
    
    # 添加 TaskCompleted 结束样本
    last_image_idx = err_action_num + len(normalized_actions) - 1
    last_image = _get_image_path(local_images, images, last_image_idx, base_dir, use_local_images)
    final_thinking = thinkings[len(normalized_actions) - 1] if len(thinkings) >= len(normalized_actions) else COMPLETION_THINKING
    
    samples.append({
        "image": last_image,
        "task": task,
        "action_history": action_history.copy(),
        "thinking": final_thinking,
        "next_action": "TaskCompleted()",
        "scene": data.get('scene', ''),
        "step": len(normalized_actions)
    })
    
    return samples


def _get_image_path(local_images: List[str], images: List[str], image_idx: int, 
                    base_dir: str, use_local_images: bool) -> str:
    """获取图像路径"""
    if use_local_images and image_idx < len(local_images):
        return local_images[image_idx]
    elif image_idx < len(images):
        img_path = images[image_idx]
        if img_path.startswith('./'):
            img_path = os.path.basename(img_path)
        elif not os.path.isabs(img_path):
            img_path = os.path.basename(img_path)
        return os.path.join(base_dir, img_path)
    return None


def process_directory(input_dir: str, output_path: str,
                      copy_images: bool = False, image_output_dir: str = None):
    """处理目录下所有 JSON 文件"""
    json_files = glob(os.path.join(input_dir, '**/*.json'), recursive=True)
    print(f"Found {len(json_files)} JSON files")
    
    if copy_images and image_output_dir:
        os.makedirs(image_output_dir, exist_ok=True)
        print(f"Images will be copied to: {image_output_dir}")
    
    all_samples = []
    copied_images = set()
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            base_dir = os.path.dirname(json_file)
            subfolder_name = os.path.basename(base_dir)
            
            samples = convert_stepwise_from_metadata(data, base_dir)
            
            # 复制图片并修改路径
            if copy_images and image_output_dir:
                subfolder_output = os.path.join(image_output_dir, subfolder_name)
                os.makedirs(subfolder_output, exist_ok=True)
                
                for sample in samples:
                    if sample.get('image') and os.path.exists(sample['image']):
                        src_image = sample['image']
                        img_filename = os.path.basename(src_image)
                        dst_image = os.path.join(subfolder_output, img_filename)
                        
                        if src_image not in copied_images:
                            shutil.copy2(src_image, dst_image)
                            copied_images.add(src_image)
                        
                        sample['image'] = os.path.join('data/images', subfolder_name, img_filename)
            
            all_samples.extend(samples)
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    _save_jsonl(all_samples, output_path)
    
    print(f"Converted {len(all_samples)} samples to {output_path}")
    if copy_images:
        print(f"Copied {len(copied_images)} unique images")
    
    return all_samples


def _save_jsonl(samples: List[Dict], output_path: str):
    """保存为 JSONL 文件"""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


# ============== 主函数 ==============

def main():
    parser = argparse.ArgumentParser(description='Convert data to Step-by-step SFT format')
    
    # 输入源（二选一）
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory containing JSON files (folder format)')
    parser.add_argument('--input_json', type=str, default=None,
                        help='Input JSON file path (multiturn format like train_multiturn_9390.json)')
    parser.add_argument('--image_base_dir', type=str, default=None,
                        help='Base directory for images (required for --input_json)')
    
    # 输出
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output JSONL file path')
    
    # 图片处理
    parser.add_argument('--copy_images', action='store_true',
                        help='Copy images to project directory')
    parser.add_argument('--image_output_dir', type=str, default='data/images',
                        help='Output directory for images (default: data/images)')
    
    args = parser.parse_args()
    
    if args.input_json:
        # 大 JSON 格式
        if not args.image_base_dir:
            args.image_base_dir = os.path.dirname(args.input_json)
        
        process_multiturn_json(
            args.input_json,
            args.output_path,
            args.image_base_dir,
            args.copy_images,
            args.image_output_dir
        )
    elif args.input_dir:
        # 文件夹格式
        process_directory(
            args.input_dir,
            args.output_path,
            args.copy_images,
            args.image_output_dir
        )
    else:
        parser.error("Either --input_dir or --input_json is required")


if __name__ == '__main__':
    main()
