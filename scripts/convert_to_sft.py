"""
将模仿学习格式的数据转换为SFT训练格式

支持两种数据源格式：
1. 文件夹格式 (navigate1open1pickup0): 每个任务一个文件夹+JSON
2. 大JSON格式 (train_multiturn_9390.json): 一个JSON文件包含所有任务

支持两种转换方式：
1. One-shot Planning: 一次性生成完整规划
2. Step-by-step: 逐步决策，每步一个样本

Usage:
    # 文件夹格式
    python scripts/convert_to_sft.py \
        --input_dir /path/to/data \
        --output_path data/sft_train.jsonl \
        --mode stepwise
    
    # 大JSON格式
    python scripts/convert_to_sft.py \
        --input_json /path/to/train_multiturn_9390.json \
        --image_base_dir /path/to/embodied_reasoner \
        --output_path data/sft_train.jsonl \
        --mode stepwise
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from glob import glob


def extract_action_from_decision(text: str) -> str:
    """从 <DecisionMaking>...</DecisionMaking> 中提取动作"""
    match = re.search(r'<DecisionMaking>(.*?)</DecisionMaking>', text)
    if match:
        return match.group(1).strip()
    return ""


def extract_thinking_from_content(content: str) -> str:
    """
    从assistant的content中提取thinking部分
    thinking是<DecisionMaking>之前的所有内容
    """
    # 找到DecisionMaking标签
    match = re.search(r'<DecisionMaking>', content)
    if match:
        thinking = content[:match.start()].strip()
        # 清理一些常见的结尾词
        thinking = re.sub(r"(Okay,\s*I('ve| think|'ll).*?|Hmm.*?|So,.*?|Alright.*?)$", "", thinking, flags=re.IGNORECASE).strip()
        return thinking
    return content.strip()


def normalize_action(action: str) -> str:
    """
    标准化动作格式
    'navigate to CounterTop' -> 'Navigate(CounterTop)'
    'pickup Apple' -> 'Pick(Apple)'
    'put in Fridge' -> 'Place(Fridge)'
    'open Fridge' -> 'Open(Fridge)'
    'close Fridge' -> 'Close(Fridge)'
    """
    action = action.strip()
    
    # 处理不同的动作格式
    if action.startswith('navigate to '):
        target = action.replace('navigate to ', '')
        return f"Navigate({target})"
    elif action.startswith('pickup '):
        target = action.replace('pickup ', '')
        return f"Pick({target})"
    elif action.startswith('put in ') or action.startswith('put '):
        target = action.replace('put in ', '').replace('put ', '')
        return f"Place({target})"
    elif action.startswith('open '):
        target = action.replace('open ', '')
        return f"Open({target})"
    elif action.startswith('close '):
        target = action.replace('close ', '')
        return f"Close({target})"
    elif action.startswith('toggle '):
        target = action.replace('toggle ', '')
        return f"Toggle({target})"
    elif action == 'observe':
        return "Observe()"
    elif action == 'move forward':
        return "MoveForward()"
    elif action == 'end' or 'task' in action.lower() and any(w in action.lower() for w in ['complete', 'end', 'done', 'finish', 'conclude']):
        return "TaskCompleted()"
    elif action.lower() in ['end task', 'task completed', 'task complete', 'done', 'finish', 'finished', 'complete task', 'conclude task']:
        return "TaskCompleted()"
    else:
        return action


# ============== 大JSON格式转换 (train_multiturn_9390.json) ==============

def convert_multiturn_stepwise(data: Dict[str, Any], image_base_dir: str, 
                                copy_images: bool = False, image_output_dir: str = None) -> Tuple[List[Dict[str, Any]], set]:
    """
    将多轮对话格式的数据转换为stepwise SFT格式
    
    Args:
        data: 单个任务数据，包含messages和images
        image_base_dir: 图片基础目录
        copy_images: 是否复制图片
        image_output_dir: 图片输出目录
    
    Returns:
        (samples列表, 复制的图片集合)
    """
    import shutil
    
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
            content = msg['content']
            # 从第一个user消息中提取Task
            task_match = re.search(r'Task:\s*"([^"]+)"', content)
            if task_match:
                task = task_match.group(1)
                break
    
    if not task:
        return samples, copied_images
    
    # 第一步的固定thinking
    INITIAL_THINKING = "I will carefully observe the environment, analyze the current scene and the task requirements, then make the correct action to accomplish the goal step by step."
    
    # 解析对话，提取动作和thinking
    action_history = []
    image_idx = 0  # 图片索引
    step = 0
    
    for i, msg in enumerate(messages):
        if msg['role'] == 'assistant':
            content = msg['content']
            
            # 提取动作
            action = extract_action_from_decision(content)
            if not action:
                continue
            
            normalized_action = normalize_action(action)
            
            # 提取thinking
            if step == 0:
                thinking = INITIAL_THINKING
            else:
                thinking = extract_thinking_from_content(content)
            
            # 获取对应的图片
            if image_idx < len(images):
                img_path = images[image_idx]
                # 处理相对路径
                if img_path.startswith('./'):
                    img_path = img_path[2:]
                # 修正路径：data/images/xxx -> data/xxx
                if img_path.startswith('data/images/'):
                    img_path = img_path.replace('data/images/', 'data/')
                full_img_path = os.path.join(image_base_dir, img_path)
                
                # 如果需要复制图片
                if copy_images and image_output_dir and os.path.exists(full_img_path):
                    # 提取子文件夹名
                    path_parts = img_path.split('/')
                    if len(path_parts) >= 3:
                        subfolder = path_parts[-2]  # 如 FloorPlan204_pickup_and_put_in_closerep_1_c
                        img_filename = path_parts[-1]
                        
                        dst_subfolder = os.path.join(image_output_dir, subfolder)
                        os.makedirs(dst_subfolder, exist_ok=True)
                        dst_path = os.path.join(dst_subfolder, img_filename)
                        
                        if full_img_path not in copied_images:
                            shutil.copy2(full_img_path, dst_path)
                            copied_images.add(full_img_path)
                        
                        # 使用相对路径
                        relative_img_path = os.path.join('data/images', subfolder, img_filename)
                    else:
                        relative_img_path = img_path
                else:
                    relative_img_path = full_img_path if os.path.exists(full_img_path) else img_path
            else:
                relative_img_path = None
            
            sample = {
                "image": relative_img_path,
                "task": task,
                "action_history": action_history.copy(),
                "thinking": thinking,
                "next_action": normalized_action,
                "step": step
            }
            samples.append(sample)
            
            # 更新历史和索引
            action_history.append(normalized_action)
            image_idx += 1
            step += 1
    
    # 检查是否需要添加TaskCompleted
    if samples and samples[-1]['next_action'] != 'TaskCompleted()':
        # 添加一个结束样本
        if image_idx < len(images):
            img_path = images[image_idx]
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
                    relative_img_path = os.path.join('data/images', subfolder, img_filename)
                else:
                    relative_img_path = img_path
            else:
                relative_img_path = full_img_path if os.path.exists(full_img_path) else None
        else:
            relative_img_path = samples[-1]['image'] if samples else None
        
        completion_sample = {
            "image": relative_img_path,
            "task": task,
            "action_history": action_history.copy(),
            "thinking": "I have successfully completed all the required subtasks. The task has been accomplished as instructed.",
            "next_action": "TaskCompleted()",
            "step": step
        }
        samples.append(completion_sample)
    
    return samples, copied_images


def process_multiturn_json(input_json: str, output_path: str, image_base_dir: str,
                           copy_images: bool = False, image_output_dir: str = None):
    """
    处理大JSON格式的数据文件 (train_multiturn_9390.json)
    
    Args:
        input_json: 输入的JSON文件路径
        output_path: 输出的JSONL文件路径
        image_base_dir: 图片基础目录
        copy_images: 是否复制图片
        image_output_dir: 图片输出目录
    """
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
    
    # 保存到JSONL文件
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(all_samples)} samples to {output_path}")
    if copy_images:
        print(f"Copied {len(all_copied_images)} unique images")
    
    return all_samples


# ============== 文件夹格式转换 (navigate1open1pickup0) ==============

def convert_oneshot(data: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    """
    方式A: One-shot Planning
    
    输入：初始图像 + 任务描述
    输出：完整的动作序列
    
    这种方式类似于HARP项目的设计，模型一次性生成整个规划
    """
    # 获取初始图像（第一张图像）
    if data.get('images') and len(data['images']) > 0:
        init_image = data['images'][0]
        # 转换为绝对路径
        if not os.path.isabs(init_image):
            init_image = os.path.join(base_dir, init_image)
    else:
        init_image = None
    
    # 获取任务描述
    task = data.get('taskname', '')
    
    # 从 task_metadata 获取 ground truth 动作序列
    actions = []
    if 'task_metadata' in data and 'actions' in data['task_metadata']:
        for act in data['task_metadata']['actions']:
            action_type = act.get('action', '')
            object_type = act.get('objectType', '')
            
            if action_type == 'navigate to':
                actions.append(f"Navigate({object_type})")
            elif action_type == 'pickup':
                actions.append(f"Pick({object_type})")
            elif action_type == 'put':
                actions.append(f"Place({object_type})")
            elif action_type == 'open':
                actions.append(f"Open({object_type})")
            elif action_type == 'close':
                actions.append(f"Close({object_type})")
            elif action_type == 'toggle':
                actions.append(f"Toggle({object_type})")
            elif action_type == 'end':
                # 通常不需要包含 end
                pass
    
    return {
        "image": init_image,
        "task": task,
        "plan": actions,
        "scene": data.get('scene', ''),
        "tasktype": data.get('tasktype', '')
    }


def convert_stepwise(data: Dict[str, Any], base_dir: str) -> List[Dict[str, Any]]:
    """
    方式B: Step-by-step Decision
    
    每一步作为一个独立的训练样本
    输入：当前图像 + 任务 + 历史动作
    输出：下一步动作
    
    这种方式更适合需要实时感知反馈的场景
    """
    samples = []
    task = data.get('taskname', '')
    images = data.get('images', [])
    
    # 从 trajectory 中提取正确的动作序列
    trajectory = data.get('trajectory', [])
    
    action_history = []
    image_idx = 0  # 跟踪对应的图像索引
    
    for i, item in enumerate(trajectory):
        # trajectory 中的格式是交替的：Reflection/Thinking -> DecisionMaking
        if '<DecisionMaking>' in item:
            action = extract_action_from_decision(item)
            normalized_action = normalize_action(action)
            
            # 获取当前图像
            # 注意：图像索引需要根据实际数据调整
            # 这里使用 err_trajectory 的长度来估算正确轨迹开始的图像索引
            err_action_num = data.get('err_action_num', 0)
            current_image_idx = err_action_num + (len(action_history))
            
            if current_image_idx < len(images):
                current_image = images[current_image_idx]
                if not os.path.isabs(current_image):
                    current_image = os.path.join(base_dir, current_image)
            else:
                current_image = None
            
            sample = {
                "image": current_image,
                "task": task,
                "action_history": action_history.copy(),
                "next_action": normalized_action,
                "scene": data.get('scene', ''),
                "step": len(action_history)
            }
            samples.append(sample)
            
            action_history.append(normalized_action)
    
    return samples


def extract_thinking_from_trajectory(trajectory: List[str]) -> List[str]:
    """
    从trajectory中提取thinking内容
    
    trajectory格式: [Reflection/Thinking, DecisionMaking, Thinking, DecisionMaking, ...]
    我们需要提取所有的Thinking内容（跳过第一个Reflection）
    """
    import re
    thinkings = []
    
    for item in trajectory:
        # 提取 <Thinking>...</Thinking> 内容
        thinking_match = re.search(r'<Thinking>(.*?)</Thinking>', item, re.DOTALL)
        if thinking_match:
            thinkings.append(thinking_match.group(1).strip())
    
    return thinkings


def convert_stepwise_from_metadata(data: Dict[str, Any], base_dir: str, use_local_images: bool = True) -> List[Dict[str, Any]]:
    """
    方式B变体: 使用 task_metadata 中的 ground truth 动作
    
    这种方式更准确，因为使用的是标注的正确动作序列
    
    关键理解：
    - 图像 FloorPlan4_12_navigate_CounterTop.png 是执行 Navigate(CounterTop) 后的结果
    - 所以看到这个图像时，应该预测的是下一步动作 Pick(Apple)
    - 训练数据格式：给定当前观察图像，预测下一步应该执行的动作
    - 每一步都包含thinking（思考过程）
    
    Args:
        data: JSON数据
        base_dir: JSON文件所在目录
        use_local_images: 是否使用本地目录下的图像文件（而不是JSON中的路径）
    """
    samples = []
    task = data.get('taskname', '')
    images = data.get('images', [])
    
    # 从trajectory中提取thinking
    trajectory = data.get('trajectory', [])
    thinkings = extract_thinking_from_trajectory(trajectory)
    
    # 从 task_metadata 获取 ground truth 动作序列
    if 'task_metadata' not in data or 'actions' not in data['task_metadata']:
        return samples
    
    gt_actions = data['task_metadata']['actions']
    
    # 过滤并标准化动作
    normalized_actions = []
    for act in gt_actions:
        action_type = act.get('action', '')
        object_type = act.get('objectType', '')
        
        if action_type == 'navigate to':
            normalized_actions.append(f"Navigate({object_type})")
        elif action_type == 'pickup':
            normalized_actions.append(f"Pick({object_type})")
        elif action_type == 'put':
            normalized_actions.append(f"Place({object_type})")
        elif action_type == 'open':
            normalized_actions.append(f"Open({object_type})")
        elif action_type == 'close':
            normalized_actions.append(f"Close({object_type})")
        elif action_type == 'toggle':
            normalized_actions.append(f"Toggle({object_type})")
        elif action_type == 'end':
            pass  # 跳过 end 动作
    
    if not normalized_actions:
        return samples
    
    # 计算正确轨迹开始的图像索引
    err_action_num = data.get('err_action_num', 0)
    
    # 如果使用本地图像，扫描目录下的所有图像文件
    local_images = []
    if use_local_images:
        import re
        for f in sorted(os.listdir(base_dir)):
            if f.endswith('.png') or f.endswith('.jpg'):
                local_images.append(os.path.join(base_dir, f))
        # 按文件名中的数字排序
        local_images.sort(key=lambda x: int(re.search(r'_(\d+)_', os.path.basename(x)).group(1)) if re.search(r'_(\d+)_', os.path.basename(x)) else 0)
    
    # 生成训练样本（SFT模式）
    # 对于SFT，我们只关注正确的示范轨迹，忽略错误轨迹
    # 
    # 正确的对应关系：
    # - Step 0: 输入初始图像 (images[0])，预测第一个动作
    # - Step i (i>0): 输入执行动作i-1后的图像，预测动作i
    #
    # 图像索引：
    # - Step 0: images[0] (初始图像)
    # - Step 1: images[err_action_num] (执行第一个正确动作后)
    # - Step i: images[err_action_num + i - 1]
    #
    # Thinking:
    # - Step 0: 固定的初始thinking
    # - Step i (i>0): 从trajectory中提取的thinking
    # - 最后一步: 添加task completed
    
    # 第一步的固定thinking
    INITIAL_THINKING = "I will carefully observe the environment, analyze the current scene and the task requirements, then make the correct action to accomplish the goal step by step."
    
    # 任务完成的thinking模板
    COMPLETION_THINKING = "I have successfully completed all the required subtasks. The task has been accomplished as instructed."
    
    action_history = []
    
    for i in range(len(normalized_actions)):
        current_action = normalized_actions[i]
        
        # 计算图像索引
        if i == 0:
            # 第一步：使用初始图像
            image_idx = 0
        else:
            # 后续步骤：使用执行前一个正确动作后的图像
            # 正确轨迹的图像从 err_action_num 开始
            image_idx = err_action_num + i - 1
        
        if use_local_images and image_idx < len(local_images):
            current_image = local_images[image_idx]
        elif image_idx < len(images):
            current_image = images[image_idx]
            # 只取文件名，与base_dir拼接
            if current_image.startswith('./'):
                img_filename = os.path.basename(current_image)
                current_image = os.path.join(base_dir, img_filename)
            elif not os.path.isabs(current_image):
                img_filename = os.path.basename(current_image)
                current_image = os.path.join(base_dir, img_filename)
        else:
            current_image = None
        
        # 确定thinking
        if i == 0:
            # 第一步使用固定的初始thinking
            thinking = INITIAL_THINKING
        elif i - 1 < len(thinkings):
            # 后续步骤使用trajectory中的thinking
            # thinkings[0]对应step 1, thinkings[1]对应step 2, ...
            thinking = thinkings[i - 1]
        else:
            thinking = ""
        
        sample = {
            "image": current_image,
            "task": task,
            "action_history": action_history.copy(),
            "thinking": thinking,
            "next_action": current_action,
            "scene": data.get('scene', ''),
            "step": i
        }
        samples.append(sample)
        
        # 更新历史
        action_history.append(current_action)
    
    # 添加最后一步：Task Completed
    # 使用最后一个正确动作执行后的图像
    last_image_idx = err_action_num + len(normalized_actions) - 1
    if use_local_images and last_image_idx < len(local_images):
        last_image = local_images[last_image_idx]
    elif last_image_idx < len(images):
        last_image = images[last_image_idx]
        if last_image.startswith('./'):
            img_filename = os.path.basename(last_image)
            last_image = os.path.join(base_dir, img_filename)
        elif not os.path.isabs(last_image):
            img_filename = os.path.basename(last_image)
            last_image = os.path.join(base_dir, img_filename)
    else:
        last_image = None
    
    # 使用trajectory中最后一个thinking（如果有的话）
    if len(thinkings) >= len(normalized_actions):
        final_thinking = thinkings[len(normalized_actions) - 1]
    else:
        final_thinking = COMPLETION_THINKING
    
    completion_sample = {
        "image": last_image,
        "task": task,
        "action_history": action_history.copy(),
        "thinking": final_thinking,
        "next_action": "TaskCompleted()",
        "scene": data.get('scene', ''),
        "step": len(normalized_actions)
    }
    samples.append(completion_sample)
    
    return samples


def process_directory(input_dir: str, output_path: str, mode: str = 'oneshot', 
                      copy_images: bool = False, image_output_dir: str = None):
    """
    处理目录下所有JSON文件
    
    Args:
        input_dir: 输入目录
        output_path: 输出JSONL文件路径
        mode: 'oneshot' 或 'stepwise'
        copy_images: 是否复制图片到项目目录
        image_output_dir: 图片输出目录（相对于项目根目录）
    """
    import shutil
    
    # 查找所有JSON文件
    json_files = glob(os.path.join(input_dir, '**/*.json'), recursive=True)
    print(f"Found {len(json_files)} JSON files")
    
    # 确定图片输出目录
    if copy_images and image_output_dir:
        os.makedirs(image_output_dir, exist_ok=True)
        print(f"Images will be copied to: {image_output_dir}")
    
    all_samples = []
    copied_images = set()  # 避免重复复制
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            base_dir = os.path.dirname(json_file)
            # 获取子文件夹名称（如 FloorPlan4_ordered_pickup_two_object_and_put_73_fix）
            subfolder_name = os.path.basename(base_dir)
            
            if mode == 'oneshot':
                sample = convert_oneshot(data, base_dir)
                if sample['plan']:  # 只添加有效样本
                    all_samples.append(sample)
            elif mode == 'stepwise':
                samples = convert_stepwise_from_metadata(data, base_dir)
                
                # 如果需要复制图片并修改路径
                if copy_images and image_output_dir:
                    # 创建子目录
                    subfolder_output = os.path.join(image_output_dir, subfolder_name)
                    os.makedirs(subfolder_output, exist_ok=True)
                    
                    for sample in samples:
                        if sample.get('image') and os.path.exists(sample['image']):
                            src_image = sample['image']
                            img_filename = os.path.basename(src_image)
                            dst_image = os.path.join(subfolder_output, img_filename)
                            
                            # 复制图片（如果还没复制过）
                            if src_image not in copied_images:
                                shutil.copy2(src_image, dst_image)
                                copied_images.add(src_image)
                            
                            # 更新为相对路径
                            sample['image'] = os.path.join('data/images', subfolder_name, img_filename)
                
                all_samples.extend(samples)
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存到JSONL文件
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(all_samples)} samples to {output_path}")
    if copy_images:
        print(f"Copied {len(copied_images)} unique images")
    return all_samples


def main():
    parser = argparse.ArgumentParser(description='Convert imitation learning data to SFT format')
    
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
    
    # 模式
    parser.add_argument('--mode', type=str, choices=['oneshot', 'stepwise'],
                        default='stepwise',
                        help='Conversion mode: oneshot (one-shot planning) or stepwise (step-by-step)')
    
    # 图片处理
    parser.add_argument('--copy_images', action='store_true',
                        help='Copy images to project directory')
    parser.add_argument('--image_output_dir', type=str, default='data/images',
                        help='Output directory for images (default: data/images)')
    
    args = parser.parse_args()
    
    # 验证输入
    if args.input_json:
        # 使用大JSON格式
        if not args.image_base_dir:
            # 默认使用JSON文件所在目录
            args.image_base_dir = os.path.dirname(args.input_json)
        
        process_multiturn_json(
            args.input_json,
            args.output_path,
            args.image_base_dir,
            args.copy_images,
            args.image_output_dir
        )
    elif args.input_dir:
        # 使用文件夹格式
        process_directory(
            args.input_dir,
            args.output_path,
            args.mode,
            args.copy_images,
            args.image_output_dir
        )
    else:
        parser.error("Either --input_dir or --input_json is required")


if __name__ == '__main__':
    main()


# ============== 示例用法 ==============
"""
# 方式A: One-shot Planning (一次性规划)
# 输出格式:
{
    "image": "/path/to/init_image.png",
    "task": "First, place Apple on Fridge, then, place Fork on SinkBasin.",
    "plan": ["Navigate(CounterTop)", "Pick(Apple)", "Navigate(Fridge)", "Open(Fridge)", "Place(Fridge)", "Close(Fridge)", "Navigate(DiningTable)", "Pick(Fork)", "Navigate(SinkBasin)", "Place(SinkBasin)"],
    "scene": "FloorPlan4",
    "tasktype": "ordered_pickup_two_object_and_put"
}

# 方式B: Step-by-step (逐步决策)
# 输出格式:
{
    "image": "/path/to/current_image.png",
    "task": "First, place Apple on Fridge, then, place Fork on SinkBasin.",
    "action_history": ["Navigate(CounterTop)", "Pick(Apple)"],
    "next_action": "Navigate(Fridge)",
    "scene": "FloorPlan4",
    "step": 2
}

# 运行示例:
python scripts/convert_to_sft.py \
    --input_dir /Users/yuyuan/Downloads/navigate1open1pickup0 \
    --output_path data/sft_oneshot.jsonl \
    --mode oneshot

python scripts/convert_to_sft.py \
    --input_dir /Users/yuyuan/Downloads/navigate1open1pickup0 \
    --output_path data/sft_stepwise.jsonl \
    --mode stepwise
"""
