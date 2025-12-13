"""
Dataset classes for HARP training

Supports:
1. SFT Dataset: (image, task) -> plan
2. Contrastive Dataset: (image, plan_positive, plan_negative)
3. RL Dataset: (image, task) for policy rollout
"""

import json
import torch
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset


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
            import random
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


def collate_fn_sft(batch: List[Dict[str, Any]], processor, tokenizer) -> Dict[str, torch.Tensor]:
    """
    Collate function for SFT training with Qwen2-VL

    Args:
        batch: List of samples from SFTDataset
        processor: Qwen2VLProcessor
        tokenizer: Qwen2VLTokenizer

    Returns:
        Batched tensors ready for model input
    """
    images = [item['image'] for item in batch]
    input_texts = [item['input_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]

    # Prepare messages for Qwen2-VL format
    messages_list = []
    for input_text in input_texts:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": input_text}
                ]
            }
        ]
        messages_list.append(messages)

    # Process inputs
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
             for msg in messages_list]

    # Tokenize
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Tokenize targets (labels)
    labels = tokenizer(
        target_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs['labels'] = labels['input_ids']

    return inputs


if __name__ == "__main__":
    # Test dataset loading
    print("Testing SFTDataset...")
    sft_dataset = SFTDataset(jsonl_path=Path("./actions.jsonl"))
    print(f"Loaded {len(sft_dataset)} samples")

    if len(sft_dataset) > 0:
        sample = sft_dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input: {sample['input_text']}")
        print(f"Target: {sample['target_text']}")
