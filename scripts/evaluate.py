"""
SFT 模型评估脚本

在测试集上评估模型性能，计算准确率
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from scripts.inference import SFTInference

logger = get_logger(__name__)


def load_test_data(test_jsonl: str):
    """加载测试数据"""
    data = []
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_single_step(inferencer, test_data, max_samples=None):
    """
    评估单步预测准确率

    对于每个样本，给定 (image, task, history)，预测 next_action
    """
    if max_samples:
        test_data = test_data[:max_samples]

    correct = 0
    total = 0
    results = []

    logger.info(f"开始评估 {len(test_data)} 个样本...")

    for item in tqdm(test_data, desc="评估中"):
        try:
            # 预测
            pred_action = inferencer.predict_next_action(
                image_path=item['image'],
                task=item['task'],
                action_history=item.get('action_history', [])
            )

            # 真实值
            gt_action = item['next_action']

            # 判断是否正确（简单字符串匹配）
            is_correct = pred_action.strip().lower() == gt_action.strip().lower()

            if is_correct:
                correct += 1

            total += 1

            # 记录结果
            results.append({
                'image': item['image'],
                'task': item['task'],
                'history': item.get('action_history', []),
                'ground_truth': gt_action,
                'prediction': pred_action,
                'correct': is_correct
            })

        except Exception as e:
            logger.error(f"处理样本失败: {e}")
            results.append({
                'image': item['image'],
                'task': item['task'],
                'error': str(e)
            })

    # 计算准确率
    accuracy = correct / total if total > 0 else 0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"评估结果:")
    logger.info(f"  总样本数: {total}")
    logger.info(f"  正确数: {correct}")
    logger.info(f"  准确率: {accuracy * 100:.2f}%")
    logger.info(f"{'=' * 60}\n")

    return results, accuracy


def evaluate_trajectory_completion(inferencer, test_data, max_samples=None):
    """
    评估完整轨迹完成率

    给定初始图像和任务，生成完整计划，与ground truth对比
    """
    if max_samples:
        test_data = test_data[:max_samples]

    # 按 (image, task) 分组
    trajectories = {}
    for item in test_data:
        key = (item['image'], item['task'])
        if key not in trajectories:
            trajectories[key] = []
        trajectories[key].append(item)

    logger.info(f"找到 {len(trajectories)} 个完整轨迹")

    results = []
    correct_plans = 0
    total_plans = 0

    for (image_path, task), traj_items in tqdm(trajectories.items(), desc="评估轨迹"):
        # 获取 ground truth plan
        gt_plan = [item['next_action'] for item in sorted(traj_items, key=lambda x: x.get('step', 0))]

        try:
            # 预测完整计划
            pred_plan = inferencer.predict_complete_plan(
                image_path=image_path,
                task=task,
                max_steps=len(gt_plan) + 5  # 给一些余量
            )

            # 比较计划
            is_correct = (
                    len(pred_plan) == len(gt_plan) and
                    all(p.strip().lower() == g.strip().lower()
                        for p, g in zip(pred_plan, gt_plan))
            )

            if is_correct:
                correct_plans += 1

            total_plans += 1

            results.append({
                'image': image_path,
                'task': task,
                'ground_truth_plan': gt_plan,
                'predicted_plan': pred_plan,
                'correct': is_correct
            })

        except Exception as e:
            logger.error(f"处理轨迹失败: {e}")
            results.append({
                'image': image_path,
                'task': task,
                'error': str(e)
            })

    # 计算准确率
    plan_accuracy = correct_plans / total_plans if total_plans > 0 else 0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"轨迹完成评估结果:")
    logger.info(f"  总轨迹数: {total_plans}")
    logger.info(f"  完全正确: {correct_plans}")
    logger.info(f"  准确率: {plan_accuracy * 100:.2f}%")
    logger.info(f"{'=' * 60}\n")

    return results, plan_accuracy


def save_results(results, output_path: str):
    """保存评估结果"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="评估SFT模型")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/sft/final",
        help="训练好的模型路径"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="基础模型路径"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/test_stepwise.jsonl",
        help="测试数据路径（JSONL格式）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="评估结果输出路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single_step",
        choices=["single_step", "trajectory"],
        help="评估模式: single_step=单步准确率, trajectory=完整轨迹"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大评估样本数（用于快速测试）"
    )

    args = parser.parse_args()

    # 加载模型
    inferencer = SFTInference(
        model_path=args.model_path,
        base_model_path=args.base_model_path
    )

    # 加载测试数据
    logger.info(f"加载测试数据: {args.test_data}")
    test_data = load_test_data(args.test_data)
    logger.info(f"总样本数: {len(test_data)}")

    # 评估
    if args.mode == "single_step":
        results, accuracy = evaluate_single_step(
            inferencer,
            test_data,
            max_samples=args.max_samples
        )
    else:  # trajectory
        results, accuracy = evaluate_trajectory_completion(
            inferencer,
            test_data,
            max_samples=args.max_samples
        )

    # 保存结果
    save_results(results, args.output)

    logger.info(f"✓ 评估完成! 准确率: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()