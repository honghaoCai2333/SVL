"""
测试集数据转换脚本

将 test_809.json 格式的测试数据转换为 stepwise SFT 格式用于评估

测试集格式：
- trajectory: 包含 Observation, Thinking, Planning, DecisionMaking, Reflection 的列表
- images: 图片路径列表
- task_metadata.actions: ground truth 动作序列

输出格式（用于评估）:
{
    "image": "path/to/image.png",
    "task": "任务描述",
    "action_history": ["历史动作1", "历史动作2"],
    "thinking": "思考过程",
    "next_action": "下一步动作",
    "gt_action": "ground truth 动作（如果不同）",
    "scene": "FloorPlan1",
    "tasktype": "single_search",
    "step": 0
}

Usage:
    python scripts/convert_test_set.py \
        --input_json /path/to/test_809.json \
        --image_base_dir /path/to/embodied_reasoner \
        --output_path data/test_stepwise.jsonl \
        --copy_images
"""

import json
import os
import re
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple


def extract_action_from_decision(text: str) -> str:
    """从 <DecisionMaking>...</DecisionMaking> 中提取动作"""
    match = re.search(r'<DecisionMaking>(.*?)</DecisionMaking>', text)
    if match:
        return match.group(1).strip()
    return ""


def extract_tag_content(text: str, tag: str) -> str:
    """从文本中提取指定标签的内容"""
    pattern = rf'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


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
    if action.lower() in ['end', 'end task', 'task completed', 'task complete', 'done', 'finish', 
                          'finished', 'complete task', 'conclude task', 'the task is complete.']:
        return "TaskCompleted()"
    
    if 'task' in action.lower() and any(w in action.lower() for w in ['complete', 'end', 'done', 'finish', 'conclude']):
        return "TaskCompleted()"
    
    # 处理导航动作
    if action.startswith('navigate to '):
        target = action.replace('navigate to ', '')
        return f"Navigate({target})"
    
    # 处理拾取动作
    if action.startswith('pickup '):
        target = action.replace('pickup ', '')
        return f"Pick({target})"
    
    # 处理放置动作
    if action.startswith('put in ') or action.startswith('put '):
        target = action.replace('put in ', '').replace('put ', '')
        return f"Place({target})"
    
    # 处理开门动作
    if action.startswith('open '):
        target = action.replace('open ', '')
        return f"Open({target})"
    
    # 处理关门动作
    if action.startswith('close '):
        target = action.replace('close ', '')
        return f"Close({target})"
    
    # 处理切换动作
    if action.startswith('toggle '):
        target = action.replace('toggle ', '')
        return f"Toggle({target})"
    
    # 处理观察动作
    if action == 'observe':
        return "Observe()"
    
    # 处理移动动作
    if action == 'move forward':
        return "MoveForward()"
    
    return action


def fix_image_path(img_path: str) -> str:
    """
    修正图片路径格式
    
    问题1: 测试集 JSON 中的路径格式: data_single_search/xxx/xxx.png
           实际目录结构: data/single_search/xxx/xxx.png
           将 data_xxx/ 转换为 data/xxx/
    
    问题2: 文件名中的坐标使用竖线: 1_Cabinet|-01.69|+02.02|-02.46.png
           实际文件名使用下划线: 1_Cabinet_-01.69_+02.02_-02.46.png
           将 | 替换为 _
    """
    import re
    
    # 修正1: 匹配 data_xxx/ 格式并替换为 data/xxx/
    pattern = r'^data_([^/]+)/'
    match = re.match(pattern, img_path)
    if match:
        folder_name = match.group(1)
        img_path = re.sub(pattern, f'data/{folder_name}/', img_path)
    
    # 修正2: 将文件名中的 | 替换为 _
    # 只替换文件名部分（最后一个 / 之后的内容）
    parts = img_path.rsplit('/', 1)
    if len(parts) == 2:
        dir_part, filename = parts
        filename = filename.replace('|', '_')
        img_path = f'{dir_part}/{filename}'
    else:
        img_path = img_path.replace('|', '_')
    
    return img_path


def normalize_action_from_metadata(act: Dict[str, Any]) -> str:
    """从 task_metadata.actions 中的动作字典标准化动作"""
    action_type = act.get('action', '')
    object_type = act.get('objectType', '')
    
    if action_type == 'navigate to':
        return f"Navigate({object_type})"
    elif action_type == 'pickup':
        return f"Pick({object_type})"
    elif action_type == 'put':
        return f"Place({object_type})"
    elif action_type == 'open':
        return f"Open({object_type})"
    elif action_type == 'close':
        return f"Close({object_type})"
    elif action_type == 'toggle':
        return f"Toggle({object_type})"
    elif action_type == 'end':
        return "TaskCompleted()"
    else:
        return action_type


def convert_test_task(data: Dict[str, Any], image_base_dir: str, 
                      copy_images: bool = False, image_output_dir: str = None) -> Tuple[List[Dict[str, Any]], set]:
    """
    将单个测试任务转换为 stepwise 格式
    
    Args:
        data: 单个任务数据
        image_base_dir: 图片基础目录
        copy_images: 是否复制图片
        image_output_dir: 图片输出目录
    
    Returns:
        (samples列表, 复制的图片集合)
    """
    samples = []
    copied_images = set()
    
    trajectory = data.get('trajectory', [])
    images = data.get('images', [])
    task = data.get('taskname', data.get('taskquery', ''))
    scene = data.get('scene', '')
    tasktype = data.get('tasktype', '')
    
    if not trajectory or not images:
        return samples, copied_images
    
    # 提取 ground truth 动作序列
    gt_actions = []
    if 'task_metadata' in data and 'actions' in data['task_metadata']:
        for act in data['task_metadata']['actions']:
            normalized = normalize_action_from_metadata(act)
            if normalized:
                gt_actions.append(normalized)
    
    # 第一步的固定 thinking
    INITIAL_THINKING = "I will carefully observe the environment, analyze the current scene and the task requirements, then make the correct action to accomplish the goal step by step."
    
    # 解析 trajectory，提取动作和思考过程
    action_history = []
    image_idx = 0
    step = 0
    current_thinking = ""
    
    for i, item in enumerate(trajectory):
        # 提取不同类型的内容
        if '<Observation>' in item:
            # 观察通常在开头
            if step == 0:
                current_thinking = INITIAL_THINKING
        elif '<Thinking>' in item:
            current_thinking = extract_tag_content(item, 'Thinking')
        elif '<Planning>' in item:
            # 可以将 Planning 作为 thinking 的一部分
            planning = extract_tag_content(item, 'Planning')
            if planning and not current_thinking:
                current_thinking = planning
        elif '<Reflection>' in item:
            # Reflection 是执行动作后的反思，用于下一步的思考
            reflection = extract_tag_content(item, 'Reflection')
            if reflection:
                current_thinking = reflection
        elif '<DecisionMaking>' in item:
            action = extract_action_from_decision(item)
            if not action:
                continue
            
            normalized_action = normalize_action(action)
            
            # 如果没有提取到 thinking，使用默认值
            if not current_thinking:
                if step == 0:
                    current_thinking = INITIAL_THINKING
                else:
                    current_thinking = "Based on the current observation, I will take the next action."
            
            # 获取对应的图片
            relative_img_path = None
            if image_idx < len(images):
                img_path = images[image_idx]
                # 修正路径格式: data_xxx/ -> data/xxx/
                img_path = fix_image_path(img_path)
                full_img_path = os.path.join(image_base_dir, img_path)
                
                if copy_images and image_output_dir and os.path.exists(full_img_path):
                    # 提取子文件夹名
                    path_parts = img_path.split('/')
                    if len(path_parts) >= 2:
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
                    relative_img_path = full_img_path if os.path.exists(full_img_path) else img_path
            
            # 获取对应的 ground truth 动作（如果有）
            gt_action = gt_actions[step] if step < len(gt_actions) else None
            
            sample = {
                "image": relative_img_path,
                "task": task,
                "action_history": action_history.copy(),
                "thinking": current_thinking,
                "next_action": normalized_action,  # 模型实际执行的动作
                "scene": scene,
                "tasktype": tasktype,
                "step": step
            }
            
            # 如果 ground truth 不同，添加字段
            if gt_action and gt_action != normalized_action:
                sample["gt_action"] = gt_action
            
            samples.append(sample)
            
            # 更新状态
            action_history.append(normalized_action)
            image_idx += 1
            step += 1
            current_thinking = ""
    
    # 检查最后一个动作是否是 TaskCompleted
    if samples and samples[-1]['next_action'] != 'TaskCompleted()':
        # 添加一个结束样本
        relative_img_path = None
        if image_idx < len(images):
            img_path = images[image_idx]
            # 修正路径格式: data_xxx/ -> data/xxx/
            img_path = fix_image_path(img_path)
            full_img_path = os.path.join(image_base_dir, img_path)
            
            if copy_images and image_output_dir and os.path.exists(full_img_path):
                path_parts = img_path.split('/')
                if len(path_parts) >= 2:
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
                relative_img_path = full_img_path if os.path.exists(full_img_path) else samples[-1]['image']
        else:
            relative_img_path = samples[-1]['image'] if samples else None
        
        completion_sample = {
            "image": relative_img_path,
            "task": task,
            "action_history": action_history.copy(),
            "thinking": "I have successfully completed all the required subtasks. The task has been accomplished as instructed.",
            "next_action": "TaskCompleted()",
            "scene": scene,
            "tasktype": tasktype,
            "step": step
        }
        samples.append(completion_sample)
    
    return samples, copied_images


def convert_test_task_from_gt(data: Dict[str, Any], image_base_dir: str, 
                               copy_images: bool = False, image_output_dir: str = None) -> Tuple[List[Dict[str, Any]], set]:
    """
    使用 ground truth 动作序列转换测试任务
    
    这种方式只使用 task_metadata.actions 中的 ground truth 动作，
    适合用于生成标准评估数据
    
    Args:
        data: 单个任务数据
        image_base_dir: 图片基础目录
        copy_images: 是否复制图片
        image_output_dir: 图片输出目录
    
    Returns:
        (samples列表, 复制的图片集合)
    """
    samples = []
    copied_images = set()
    
    images = data.get('images', [])
    task = data.get('taskname', data.get('taskquery', ''))
    scene = data.get('scene', '')
    tasktype = data.get('tasktype', '')
    trajectory = data.get('trajectory', [])
    
    # 提取 ground truth 动作序列
    gt_actions = []
    if 'task_metadata' in data and 'actions' in data['task_metadata']:
        for act in data['task_metadata']['actions']:
            normalized = normalize_action_from_metadata(act)
            if normalized:
                gt_actions.append(normalized)
    
    if not gt_actions or not images:
        return samples, copied_images
    
    # 第一步的固定 thinking
    INITIAL_THINKING = "I will carefully observe the environment, analyze the current scene and the task requirements, then make the correct action to accomplish the goal step by step."
    
    # 从 trajectory 中提取 thinking 内容
    thinkings = []
    for item in trajectory:
        if '<Thinking>' in item:
            thinking = extract_tag_content(item, 'Thinking')
            if thinking:
                thinkings.append(thinking)
        elif '<Reflection>' in item:
            reflection = extract_tag_content(item, 'Reflection')
            if reflection:
                thinkings.append(reflection)
    
    action_history = []
    
    for step, gt_action in enumerate(gt_actions):
        # 获取对应的图片（初始图像是 step 0，执行动作后的图像是 step+1）
        image_idx = step
        
        relative_img_path = None
        if image_idx < len(images):
            img_path = images[image_idx]
            # 修正路径格式: data_xxx/ -> data/xxx/
            img_path = fix_image_path(img_path)
            full_img_path = os.path.join(image_base_dir, img_path)
            
            if copy_images and image_output_dir and os.path.exists(full_img_path):
                path_parts = img_path.split('/')
                if len(path_parts) >= 2:
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
                # 使用修正后的完整路径
                relative_img_path = full_img_path if os.path.exists(full_img_path) else img_path
        
        # 确定 thinking
        if step == 0:
            thinking = INITIAL_THINKING
        elif step - 1 < len(thinkings):
            thinking = thinkings[step - 1]
        else:
            thinking = "Based on the current observation, I will take the next action."
        
        sample = {
            "image": relative_img_path,
            "task": task,
            "action_history": action_history.copy(),
            "thinking": thinking,
            "next_action": gt_action,
            "scene": scene,
            "tasktype": tasktype,
            "step": step
        }
        samples.append(sample)
        
        action_history.append(gt_action)
    
    return samples, copied_images


def process_test_json(input_json: str, output_path: str, image_base_dir: str,
                      copy_images: bool = False, image_output_dir: str = None,
                      use_gt_actions: bool = True):
    """
    处理测试集 JSON 文件
    
    Args:
        input_json: 输入的 JSON 文件路径
        output_path: 输出的 JSONL 文件路径
        image_base_dir: 图片基础目录
        copy_images: 是否复制图片
        image_output_dir: 图片输出目录
        use_gt_actions: 是否使用 ground truth 动作（True）或模型实际执行的动作（False）
    """
    print(f"Loading {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print(f"Found {len(data_list)} test tasks")
    print(f"Mode: {'Ground Truth Actions' if use_gt_actions else 'Model Executed Actions'}")
    
    if copy_images and image_output_dir:
        os.makedirs(image_output_dir, exist_ok=True)
        print(f"Images will be copied to: {image_output_dir}")
    
    all_samples = []
    all_copied_images = set()
    
    convert_func = convert_test_task_from_gt if use_gt_actions else convert_test_task
    
    for i, data in enumerate(data_list):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(data_list)}...")
        
        try:
            samples, copied = convert_func(
                data, image_base_dir, copy_images, image_output_dir
            )
            all_samples.extend(samples)
            all_copied_images.update(copied)
        except Exception as e:
            print(f"Error processing task {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存到 JSONL 文件
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n=== 转换完成 ===")
    print(f"测试任务数: {len(data_list)}")
    print(f"总样本数: {len(all_samples)}")
    print(f"输出文件: {output_path}")
    if copy_images:
        print(f"复制图片数: {len(all_copied_images)}")
    
    # 统计任务类型分布
    tasktype_counts = {}
    for sample in all_samples:
        tt = sample.get('tasktype', 'unknown')
        tasktype_counts[tt] = tasktype_counts.get(tt, 0) + 1
    
    print(f"\n任务类型分布:")
    for tt, count in sorted(tasktype_counts.items(), key=lambda x: -x[1]):
        print(f"  {tt}: {count}")
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(description='Convert test set data to SFT format for evaluation')
    
    parser.add_argument('--input_json', type=str, required=True,
                        help='Input test JSON file path (e.g., test_809.json)')
    parser.add_argument('--image_base_dir', type=str, default=None,
                        help='Base directory for images (default: directory of input_json)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output JSONL file path')
    
    # 图片处理
    parser.add_argument('--copy_images', action='store_true',
                        help='Copy images to project directory')
    parser.add_argument('--image_output_dir', type=str, default='data/images',
                        help='Output directory for images (default: data/images)')
    
    # 动作来源
    parser.add_argument('--use_model_actions', action='store_true',
                        help='Use model executed actions instead of ground truth actions')
    
    args = parser.parse_args()
    
    # 默认使用 JSON 文件所在目录作为图片基础目录
    if args.image_base_dir is None:
        args.image_base_dir = os.path.dirname(args.input_json)
    
    process_test_json(
        args.input_json,
        args.output_path,
        args.image_base_dir,
        args.copy_images,
        args.image_output_dir,
        use_gt_actions=not args.use_model_actions
    )


if __name__ == '__main__':
    main()


# ============== 示例用法 ==============
"""
# 基本用法（使用 ground truth 动作，不复制图片）
python scripts/convert_test_set.py \
    --input_json /Users/yuyuan/Desktop/embodied_reasoner/test_809.json \
    --output_path data/test_stepwise.jsonl

# 使用 ground truth 动作并复制图片
python scripts/convert_test_set.py \
    --input_json /Users/yuyuan/Desktop/embodied_reasoner/test_809.json \
    --image_base_dir /Users/yuyuan/Desktop/embodied_reasoner \
    --output_path data/test_stepwise.jsonl \
    --copy_images \
    --image_output_dir data/images

# 使用模型实际执行的动作（用于分析模型行为）
python scripts/convert_test_set.py \
    --input_json /Users/yuyuan/Desktop/embodied_reasoner/test_809.json \
    --output_path data/test_model_actions.jsonl \
    --use_model_actions
"""
