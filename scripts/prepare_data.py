"""
Split data into train and validation sets
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import argparse

from src.utils.logger import get_logger

logger = get_logger(__name__)



def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from jsonl file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: Path):
    """Save data to jsonl file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_data(input_path: Path,
               train_path: Path,
               val_path: Path,
               val_ratio: float = 0.1,
               seed: int = 42):
    """
    Split data into train and validation sets

    Args:
        input_path: Path to input jsonl file
        train_path: Path to output train jsonl file
        val_path: Path to output validation jsonl file
        val_ratio: Ratio of validation set (default: 0.1 = 10%)
        seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(seed)

    # Load data
    logger.info(f"Loading data from {input_path}...")
    data = load_jsonl(input_path)
    logger.info(f"Total samples: {len(data)}")

    # Shuffle data
    random.shuffle(data)

    # Split
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size

    train_data = data[:train_size]
    val_data = data[train_size:]

    # Save
    logger.info(f"Saving train set ({len(train_data)} samples) to {train_path}...")
    save_jsonl(train_data, train_path)

    logger.info(f"Saving validation set ({len(val_data)} samples) to {val_path}...")
    save_jsonl(val_data, val_path)

    logger.info("Data split complete!")
    logger.info(f"Train: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
    logger.info(f"Val: {len(val_data)} samples ({len(val_data)/len(data)*100:.1f}%)")

    return len(train_data), len(val_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train and validation sets")
    parser.add_argument("--input", type=str, default="./actions.jsonl",
                        help="Input jsonl file path")
    parser.add_argument("--train_output", type=str, default="./data/train.jsonl",
                        help="Output train jsonl file path")
    parser.add_argument("--val_output", type=str, default="./data/val.jsonl",
                        help="Output validation jsonl file path")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    input_path = Path(args.input)
    train_path = Path(args.train_output)
    val_path = Path(args.val_output)

    if not input_path.exists():
        logger.info(f"Error: Input file {input_path} does not exist!")
        logger.info("Please run data_build.py first to generate training data.")
        exit(1)

    split_data(
        input_path=input_path,
        train_path=train_path,
        val_path=val_path,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
