"""
Negative Sampler for Contrastive Learning

Generate negative samples for Scene-Plan Contrastive Learning (SPCL):
1. Random negative: pair scene_i with plan_j (j != i)
2. Shuffled negative: shuffle the order of actions in the plan
3. Infeasible negative: create plans that violate state machine constraints
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import copy


class NegativeSampler:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.action_types = ["Navigate", "Pick", "Place"]

    def load_data(self, jsonl_path: Path) -> List[Dict[str, Any]]:
        """Load data from jsonl file"""
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def parse_plan(self, plan: List[str]) -> List[Dict[str, str]]:
        """
        Parse plan from string format to structured format
        Example: "Navigate(Table)" -> {"action": "Navigate", "target": "Table"}
        """
        parsed = []
        for step in plan:
            action = step.split('(')[0]
            target = step.split('(')[1].rstrip(')')
            parsed.append({"action": action, "target": target})
        return parsed

    def unparse_plan(self, parsed_plan: List[Dict[str, str]]) -> List[str]:
        """Convert structured plan back to string format"""
        return [f"{step['action']}({step['target']})" for step in parsed_plan]

    def generate_random_negative(self, data: List[Dict[str, Any]], num_negatives: int = 1) -> List[Dict[str, Any]]:
        """
        Generate random negative samples by pairing scene_i with plan_j (j != i)
        """
        negatives = []
        n = len(data)

        if n < 2:
            return negatives  # 需要至少2个样本才能生成random negative

        for item in data:
            for _ in range(num_negatives):
                # Randomly select a different plan
                candidate_indices = [i for i in range(n) if data[i]['image'] != item['image']]

                # 如果没有不同图片的样本，选择不同的plan（即使图片相同）
                if not candidate_indices:
                    candidate_indices = [i for i in range(n) if data[i]['plan'] != item['plan']]

                if not candidate_indices:
                    continue  # 如果仍然没有候选，跳过

                neg_idx = random.choice(candidate_indices)
                negative = {
                    'image': item['image'],
                    'task': item['task'],
                    'plan': data[neg_idx]['plan'],
                    'label': 'random_negative'
                }
                negatives.append(negative)

        return negatives

    def generate_shuffled_negative(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate shuffled negative samples by randomly shuffling the action order
        This creates infeasible plans (e.g., Pick before Navigate)
        """
        negatives = []

        for item in data:
            plan = item['plan']
            if len(plan) <= 1:
                continue

            # Shuffle the plan
            shuffled_plan = plan.copy()
            random.shuffle(shuffled_plan)

            # Make sure it's actually different
            if shuffled_plan == plan:
                # Swap first two elements if shuffle didn't change anything
                shuffled_plan[0], shuffled_plan[1] = shuffled_plan[1], shuffled_plan[0]

            negative = {
                'image': item['image'],
                'task': item['task'],
                'plan': shuffled_plan,
                'label': 'shuffled_negative'
            }
            negatives.append(negative)

        return negatives

    def generate_infeasible_negative(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate infeasible negative samples that violate state machine constraints
        Examples:
        - Pick without prior Navigate
        - Place without prior Pick
        - Duplicate actions
        """
        negatives = []

        for item in data:
            parsed_plan = self.parse_plan(item['plan'])
            if len(parsed_plan) < 2:
                continue

            # Strategy 1: Remove Navigate before Pick
            infeasible_1 = [step for step in parsed_plan if step['action'] != 'Navigate']
            if len(infeasible_1) > 0 and infeasible_1 != parsed_plan:
                negatives.append({
                    'image': item['image'],
                    'task': item['task'],
                    'plan': self.unparse_plan(infeasible_1),
                    'label': 'infeasible_no_navigate'
                })

            # Strategy 2: Insert Place before Pick
            if any(step['action'] == 'Pick' for step in parsed_plan):
                infeasible_2 = copy.deepcopy(parsed_plan)
                pick_idx = next(i for i, step in enumerate(infeasible_2) if step['action'] == 'Pick')
                if pick_idx > 0:
                    # Insert Place before Pick
                    place_step = {'action': 'Place', 'target': parsed_plan[-1]['target'] if parsed_plan[-1]['action'] == 'Place' else 'Unknown'}
                    infeasible_2.insert(pick_idx, place_step)
                    negatives.append({
                        'image': item['image'],
                        'task': item['task'],
                        'plan': self.unparse_plan(infeasible_2),
                        'label': 'infeasible_place_before_pick'
                    })

            # Strategy 3: Duplicate actions
            if len(parsed_plan) >= 2:
                infeasible_3 = copy.deepcopy(parsed_plan)
                dup_idx = random.randint(0, len(infeasible_3) - 1)
                infeasible_3.insert(dup_idx, infeasible_3[dup_idx])
                negatives.append({
                    'image': item['image'],
                    'task': item['task'],
                    'plan': self.unparse_plan(infeasible_3),
                    'label': 'infeasible_duplicate'
                })

        return negatives

    def generate_all_negatives(self,
                               jsonl_path: Path,
                               output_path: Path,
                               num_random: int = 1) -> Dict[str, int]:
        """
        Generate all types of negative samples

        Args:
            jsonl_path: Path to input jsonl file with positive samples
            output_path: Path to output jsonl file with negative samples
            num_random: Number of random negatives per positive sample

        Returns:
            Statistics dictionary
        """
        # Load positive samples
        data = self.load_data(jsonl_path)

        # Generate negatives
        random_negs = self.generate_random_negative(data, num_random)
        shuffled_negs = self.generate_shuffled_negative(data)
        infeasible_negs = self.generate_infeasible_negative(data)

        # Combine all negatives
        all_negatives = random_negs + shuffled_negs + infeasible_negs

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for neg in all_negatives:
                f.write(json.dumps(neg, ensure_ascii=False) + '\n')

        stats = {
            'num_positives': len(data),
            'num_random_negatives': len(random_negs),
            'num_shuffled_negatives': len(shuffled_negs),
            'num_infeasible_negatives': len(infeasible_negs),
            'total_negatives': len(all_negatives)
        }

        return stats


if __name__ == "__main__":
    sampler = NegativeSampler(seed=42)

    # Generate negative samples from actions.jsonl
    input_path = Path("./actions.jsonl")
    output_path = Path("./data/negative_samples.jsonl")

    if input_path.exists():
        stats = sampler.generate_all_negatives(
            jsonl_path=input_path,
            output_path=output_path,
            num_random=2
        )

        print("Negative Sample Generation Complete!")
        print(f"Positive samples: {stats['num_positives']}")
        print(f"Random negatives: {stats['num_random_negatives']}")
        print(f"Shuffled negatives: {stats['num_shuffled_negatives']}")
        print(f"Infeasible negatives: {stats['num_infeasible_negatives']}")
        print(f"Total negatives: {stats['total_negatives']}")
        print(f"Output saved to: {output_path}")
    else:
        print(f"Error: {input_path} not found!")
        print("Please run data_build.py first to generate training data.")
