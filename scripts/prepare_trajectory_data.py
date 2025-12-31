"""
Convert trajectory data to JSONL format for SFT training

This script processes trajectory directories and converts step-by-step data
into a flat JSONL format suitable for training.

Usage:
    python scripts/prepare_trajectory_data.py \
        --input_dir /path/to/trajectories \
        --output_dir data/trajectory \
        --val_ratio 0.1
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_trajectory_from_json_files(trajectory_dir: Path) -> List[Dict[str, Any]]:
    """
    Load trajectory data from individual JSON files in a directory

    Expected structure:
        trajectory_dir/
            step_0.json
            step_1.json
            ...

    Or individual JSON files with step data.
    """
    steps = []

    # Try to find JSON files
    json_files = list(trajectory_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                steps.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    # Sort by step number
    steps.sort(key=lambda x: x.get('step', 0))

    return steps


def load_trajectory_from_single_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load trajectory data from a single JSON file containing all steps

    Expected format:
    [
        {"image": "...", "task": "...", "action_history": [], "thinking": "...", "next_action": "...", "step": 0},
        {"image": "...", "task": "...", "action_history": [...], "thinking": "...", "next_action": "...", "step": 1},
        ...
    ]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        return sorted(data, key=lambda x: x.get('step', 0))
    else:
        # Single step
        return [data]


def process_trajectory_directory(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Process a directory containing multiple trajectory folders/files

    Expected structure option 1 (folders):
        input_dir/
            FloorPlan1_task_1/
                step_0.json, step_1.json, ...
            FloorPlan1_task_2/
                step_0.json, step_1.json, ...

    Expected structure option 2 (single files per trajectory):
        input_dir/
            trajectory_1.json
            trajectory_2.json
    """
    all_samples = []

    # Check if input_dir contains subdirectories (trajectory folders)
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if subdirs:
        # Process each trajectory folder
        for traj_dir in subdirs:
            logger.info(f"Processing trajectory: {traj_dir.name}")
            steps = load_trajectory_from_json_files(traj_dir)
            all_samples.extend(steps)
    else:
        # Process JSON files directly
        json_files = list(input_dir.glob("*.json"))
        for json_file in json_files:
            logger.info(f"Processing file: {json_file.name}")
            steps = load_trajectory_from_single_json(json_file)
            all_samples.extend(steps)

    return all_samples


def convert_raw_step_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw step data to standardized format

    Handles variations in input format and ensures consistent output.
    """
    # Required fields
    converted = {
        "image": raw_data.get("image", ""),
        "task": raw_data.get("task", ""),
        "action_history": raw_data.get("action_history", []),
        "thinking": raw_data.get("thinking", ""),
        "next_action": raw_data.get("next_action", ""),
        "step": raw_data.get("step", 0),
        "scene": raw_data.get("scene", "unknown")
    }

    # Validate required fields
    if not converted["image"]:
        logger.warning(f"Missing image path in step {converted['step']}")
    if not converted["next_action"]:
        logger.warning(f"Missing next_action in step {converted['step']}")

    return converted


def split_by_trajectory(samples: List[Dict[str, Any]], val_ratio: float = 0.1) -> tuple:
    """
    Split samples into train/val sets, keeping trajectories together

    This ensures that all steps from a trajectory are in the same split,
    preventing data leakage.
    """
    # Group samples by trajectory (scene + task)
    trajectories = defaultdict(list)

    for sample in samples:
        # Create trajectory key from scene and task
        traj_key = f"{sample.get('scene', 'unknown')}_{sample.get('task', '')[:50]}"
        trajectories[traj_key].append(sample)

    # Split trajectories
    traj_keys = list(trajectories.keys())
    random.shuffle(traj_keys)

    val_count = max(1, int(len(traj_keys) * val_ratio))
    val_keys = set(traj_keys[:val_count])
    train_keys = set(traj_keys[val_count:])

    train_samples = []
    val_samples = []

    for key in train_keys:
        train_samples.extend(trajectories[key])
    for key in val_keys:
        val_samples.extend(trajectories[key])

    logger.info(f"Split: {len(train_keys)} train trajectories, {len(val_keys)} val trajectories")
    logger.info(f"Split: {len(train_samples)} train steps, {len(val_samples)} val steps")

    return train_samples, val_samples


def save_to_jsonl(samples: List[Dict[str, Any]], output_path: Path):
    """Save samples to JSONL file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(samples)} samples to {output_path}")


def validate_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and filter samples"""
    valid_samples = []

    for sample in samples:
        # Check required fields
        if not sample.get('image'):
            logger.warning(f"Skipping sample: missing image")
            continue
        if not sample.get('next_action'):
            logger.warning(f"Skipping sample: missing next_action")
            continue
        if not sample.get('task'):
            logger.warning(f"Skipping sample: missing task")
            continue

        # Check if image file exists (optional, can be slow)
        # if not Path(sample['image']).exists():
        #     logger.warning(f"Skipping sample: image not found: {sample['image']}")
        #     continue

        valid_samples.append(sample)

    logger.info(f"Validated: {len(valid_samples)}/{len(samples)} samples are valid")
    return valid_samples


def main():
    parser = argparse.ArgumentParser(description="Prepare trajectory data for SFT training")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing trajectory data")
    parser.add_argument("--output_dir", type=str, default="data/trajectory",
                        help="Output directory for processed data")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Skip sample validation")
    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # Process trajectory data
    logger.info(f"Processing trajectories from: {input_dir}")
    raw_samples = process_trajectory_directory(input_dir)
    logger.info(f"Loaded {len(raw_samples)} raw samples")

    # Convert to standardized format
    samples = [convert_raw_step_data(s) for s in raw_samples]

    # Validate samples
    if not args.skip_validation:
        samples = validate_samples(samples)

    if len(samples) == 0:
        logger.error("No valid samples found!")
        return

    # Split into train/val
    train_samples, val_samples = split_by_trajectory(samples, args.val_ratio)

    # Save to JSONL
    save_to_jsonl(train_samples, output_dir / "trajectory_train.jsonl")
    save_to_jsonl(val_samples, output_dir / "trajectory_val.jsonl")

    # Print statistics
    logger.info("\n=== Statistics ===")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")

    # Action distribution
    action_counts = defaultdict(int)
    for sample in samples:
        action = sample['next_action'].split('(')[0] if sample['next_action'] else 'Unknown'
        action_counts[action] += 1

    logger.info("\nAction distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {action}: {count} ({count/len(samples)*100:.1f}%)")


if __name__ == "__main__":
    main()
