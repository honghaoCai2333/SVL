"""
Data preparation script for SFT training

Convert Step2 generation results to training JSONL format
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_step2_data(step2_path: str) -> List[Dict]:
    """
    Load Step2 generation results

    Args:
        step2_path: Path to step2.json file

    Returns:
        List of successful generation results
    """
    with open(step2_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter successful generations
    valid_data = [item for item in data if item.get('success', False)]

    logger.info(f"Loaded {len(data)} total samples, {len(valid_data)} successful")

    return valid_data


def convert_to_training_format(step2_data: List[Dict]) -> List[Dict]:
    """
    Convert Step2 format to training JSONL format

    Step2 format:
    {
        "success": true,
        "image_path": "path/to/image.jpg",
        "task": "Put apple in basket",
        "plan": ["Navigate(Table)", "Pick(Apple)", ...],
        "reasoning": "..."
    }

    Training format:
    {
        "image": "path/to/image.jpg",
        "task": "Put apple in basket",
        "plan": ["Navigate(Table)", "Pick(Apple)", ...]
    }

    Args:
        step2_data: List of Step2 results

    Returns:
        List of training samples
    """
    training_samples = []

    for item in step2_data:
        # Skip if plan is empty
        if not item.get('plan') or len(item['plan']) == 0:
            continue

        training_sample = {
            "image": item['image_path'],
            "task": item['task'],
            "plan": item['plan']
        }

        training_samples.append(training_sample)

    logger.info(f"Converted {len(training_samples)} training samples")

    return training_samples


def split_train_val(
        data: List[Dict],
        val_ratio: float = 0.1,
        seed: int = 42
) -> tuple[List[Dict], List[Dict]]:
    """
    Split data into train and validation sets

    Args:
        data: List of training samples
        val_ratio: Ratio of validation set
        seed: Random seed

    Returns:
        Tuple of (train_data, val_data)
    """
    random.seed(seed)

    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Split
    val_size = int(len(shuffled_data) * val_ratio)
    val_data = shuffled_data[:val_size]
    train_data = shuffled_data[val_size:]

    logger.info(f"Split into {len(train_data)} train and {len(val_data)} val samples")

    return train_data, val_data


def save_jsonl(data: List[Dict], output_path: str):
    """
    Save data to JSONL format (one JSON per line)

    Args:
        data: List of samples
        output_path: Output JSONL file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training data")
    parser.add_argument(
        "--step2_data",
        type=str,
        required=True,
        help="Path to step2.json file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for train/val JSONL files"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting"
    )

    args = parser.parse_args()

    # Load Step2 data
    logger.info(f"Loading data from {args.step2_data}")
    step2_data = load_step2_data(args.step2_data)

    # Convert to training format
    training_samples = convert_to_training_format(step2_data)

    # Split train/val
    train_data, val_data = split_train_val(
        training_samples,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Save to JSONL
    output_dir = Path(args.output_dir)
    train_path = output_dir / "sft_train.jsonl"
    val_path = output_dir / "sft_val.jsonl"

    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)

    logger.info("Data preparation complete!")
    logger.info(f"  Train: {train_path} ({len(train_data)} samples)")
    logger.info(f"  Val: {val_path} ({len(val_data)} samples)")


if __name__ == "__main__":
    main()