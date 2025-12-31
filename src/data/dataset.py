"""
Dataset classes for HARP training

Supports:
1. SFT Dataset: (image, task) -> plan
2. Contrastive Dataset: (image, plan_positive, plan_negative)
3. RL Dataset: (image, task) for policy rollout
"""

import json
import random
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any
from torch.utils.data import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)



class HARPDataset(Dataset):
    """Base dataset for HARP training"""

    def __init__(self,
                 jsonl_path: Path,
                 image_processor=None,
                 tokenizer=None,
                 max_length: int = 512):
        self.data = self.load_jsonl(jsonl_path)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_jsonl(self, jsonl_path: Path) -> List[Dict[str, Any]]:
        """Load data from jsonl file"""
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def load_image(self, image_path: str) -> Image.Image:
        """Load image from path"""
        return Image.open(image_path).convert('RGB')

    def format_plan(self, plan: List[str]) -> str:
        """Format plan list to string"""
        return ", ".join(plan)

    def __len__(self) -> int:
        return len(self.data)


class SFTDataset(HARPDataset):
    """
    Supervised Fine-Tuning Dataset

    Returns:
        - image: processed image tensor
        - input_text: "Task: {task}"
        - target_text: "{plan}"
    """

    def __init__(self,
                 jsonl_path: Path,
                 image_processor=None,
                 tokenizer=None,
                 max_length: int = 512):
        super().__init__(jsonl_path, image_processor, tokenizer, max_length)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Load image
        image = self.load_image(item['image'])

        # Format text
        task = item['task']
        plan = self.format_plan(item['plan'])

        # Prepare input
        input_text = f"Task: {task}"
        target_text = plan

        result = {
            'image': image,
            'input_text': input_text,
            'target_text': target_text,
            'image_path': item['image']
        }

        # Process with tokenizer if provided
        if self.image_processor is not None:
            result['pixel_values'] = self.image_processor(image, return_tensors='pt').pixel_values[0]

        if self.tokenizer is not None:
            # For Qwen2-VL, we'll handle tokenization in the collator
            pass

        return result


class ContrastiveDataset(HARPDataset):
    """
    Contrastive Learning Dataset for SPCL

    Returns:
        - image: processed image tensor
        - plan_positive: correct plan for this scene
        - plan_negative: incorrect plan (from negative_samples.jsonl)
    """

    def __init__(self,
                 positive_jsonl: Path,
                 negative_jsonl: Path,
                 image_processor=None,
                 tokenizer=None,
                 max_length: int = 512):
        # Load positive samples
        super().__init__(positive_jsonl, image_processor, tokenizer, max_length)
        self.positive_data = self.data

        # Load negative samples
        self.negative_data = self.load_jsonl(negative_jsonl)

        # Build negative lookup: image_path -> list of negative plans
        self.negative_lookup = {}
        for neg in self.negative_data:
            img_path = neg['image']
            if img_path not in self.negative_lookup:
                self.negative_lookup[img_path] = []
            self.negative_lookup[img_path].append(neg['plan'])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pos_item = self.positive_data[idx]

        # Load image
        image = self.load_image(pos_item['image'])

        # Get positive plan
        plan_positive = self.format_plan(pos_item['plan'])

        # Get negative plan
        img_path = pos_item['image']
        if img_path in self.negative_lookup and len(self.negative_lookup[img_path]) > 0:
            neg_plan_list = random.choice(self.negative_lookup[img_path])
            plan_negative = self.format_plan(neg_plan_list)
        else:
            # Fallback: use another positive sample as negative
            other_idx = (idx + 1) % len(self.positive_data)
            plan_negative = self.format_plan(self.positive_data[other_idx]['plan'])

        result = {
            'image': image,
            'plan_positive': plan_positive,
            'plan_negative': plan_negative,
            'image_path': pos_item['image']
        }

        # Process with image processor if provided
        if self.image_processor is not None:
            result['pixel_values'] = self.image_processor(image, return_tensors='pt').pixel_values[0]

        return result


class RLDataset(HARPDataset):
    """
    RL Dataset for policy rollout

    Returns:
        - image: processed image tensor
        - task: task description
        - reference_plan: ground truth plan (for reward computation)
    """

    def __init__(self,
                 jsonl_path: Path,
                 image_processor=None,
                 tokenizer=None,
                 max_length: int = 512):
        super().__init__(jsonl_path, image_processor, tokenizer, max_length)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Load image
        image = self.load_image(item['image'])

        # Get task and reference plan
        task = item['task']
        reference_plan = item['plan']

        result = {
            'image': image,
            'task': task,
            'reference_plan': reference_plan,
            'image_path': item['image']
        }

        # Process with image processor if provided
        if self.image_processor is not None:
            result['pixel_values'] = self.image_processor(image, return_tensors='pt').pixel_values[0]

        return result


class TrajectoryDataset(HARPDataset):
    """
    轨迹数据集 - 逐步预测模式

    每个样本是轨迹中的一个step，模型需要根据当前图像、任务和历史动作
    来预测下一个动作（包含思考过程）

    数据格式:
    {
        "image": "path/to/step_image.png",
        "task": "First, place Apple on Fridge, then place Fork on SinkBasin.",
        "action_history": ["Navigate(CounterTop)", "Pick(Apple)"],
        "thinking": "I observe... Reflecting on the task...",
        "next_action": "Navigate(Fridge)",
        "scene": "FloorPlan4",
        "step": 2
    }

    Returns:
        - image: PIL Image of current view
        - input_text: formatted input with task and action history
        - target_text: thinking + next_action
        - step: step number in trajectory
        - scene: scene identifier
    """

    def __init__(self,
                 jsonl_path: Path,
                 image_processor=None,
                 tokenizer=None,
                 max_length: int = 1024,
                 include_thinking: bool = True):
        """
        Args:
            jsonl_path: Path to JSONL file with trajectory data
            image_processor: Image processor for the VLM
            tokenizer: Tokenizer for the VLM
            max_length: Maximum sequence length
            include_thinking: Whether to include thinking in target (for CoT training)
        """
        super().__init__(jsonl_path, image_processor, tokenizer, max_length)
        self.include_thinking = include_thinking
        logger.info(f"Loaded TrajectoryDataset with {len(self.data)} steps, include_thinking={include_thinking}")

    def format_action_history(self, action_history: List[str]) -> str:
        """Format action history list to string"""
        if not action_history:
            return "None"
        return " -> ".join(action_history)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Load image
        image = self.load_image(item['image'])

        # Extract fields
        task = item['task']
        action_history = item.get('action_history', [])
        thinking = item.get('thinking', '')
        next_action = item['next_action']
        step = item.get('step', 0)
        scene = item.get('scene', 'unknown')

        # Format input text
        history_str = self.format_action_history(action_history)
        if step == 0:
            input_text = f"Task: {task}\n\nThis is the initial state. What action should I take first?"
        else:
            input_text = f"Task: {task}\n\nCompleted actions: {history_str}\n\nBased on the current view, what action should I take next?"

        # Format target text
        if self.include_thinking:
            target_text = f"Thinking: {thinking}\n\nNext Action: {next_action}"
        else:
            target_text = next_action

        result = {
            'image': image,
            'input_text': input_text,
            'target_text': target_text,
            'step': step,
            'scene': scene,
            'image_path': item['image'],
            'action_history': action_history,
            'next_action': next_action
        }

        # Process with image processor if provided
        if self.image_processor is not None:
            result['pixel_values'] = self.image_processor(image, return_tensors='pt').pixel_values[0]

        return result


if __name__ == "__main__":
    # Test dataset loading
    logger.info("Testing SFTDataset...")
    sft_dataset = SFTDataset(jsonl_path=Path("./actions.jsonl"))
    logger.info(f"Loaded {len(sft_dataset)} samples")

    if len(sft_dataset) > 0:
        sample = sft_dataset[0]
        logger.info(f"Sample keys: {sample.keys()}")
        logger.info(f"Input: {sample['input_text']}")
        logger.info(f"Target: {sample['target_text']}")

    # Test TrajectoryDataset
    logger.info("\n" + "="*50)
    logger.info("Testing TrajectoryDataset...")

    # Create test data
    test_trajectory_data = [
        {
            "image": "test_image_0.png",
            "task": "Place Apple on Fridge",
            "action_history": [],
            "thinking": "I see a kitchen with a countertop and fridge. The Apple is on the countertop.",
            "next_action": "Navigate(CounterTop)",
            "scene": "FloorPlan4",
            "step": 0
        },
        {
            "image": "test_image_1.png",
            "task": "Place Apple on Fridge",
            "action_history": ["Navigate(CounterTop)"],
            "thinking": "I am now at the countertop and can see the Apple. I should pick it up.",
            "next_action": "Pick(Apple)",
            "scene": "FloorPlan4",
            "step": 1
        }
    ]

    # Write test data
    test_jsonl_path = Path("/tmp/test_trajectory.jsonl")
    with open(test_jsonl_path, 'w') as f:
        for item in test_trajectory_data:
            import json
            f.write(json.dumps(item) + '\n')

    # Load and test
    traj_dataset = TrajectoryDataset(jsonl_path=test_jsonl_path, include_thinking=True)
    logger.info(f"Loaded {len(traj_dataset)} trajectory steps")

    for i in range(len(traj_dataset)):
        sample = traj_dataset[i]
        logger.info(f"\n--- Step {sample['step']} ---")
        logger.info(f"Input: {sample['input_text'][:100]}...")
        logger.info(f"Target: {sample['target_text'][:100]}...")

    # Clean up
    test_jsonl_path.unlink()
