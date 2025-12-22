"""
Hierarchical Process Reward Model (H-PRM)

Provides multi-level reward signals for embodied planning:
1. Format Reward: Checks if output format is correct
2. Action Reward: Checks if all actions are valid
3. Transition Reward: Checks if action sequence satisfies state machine constraints
4. Task Reward: Checks if plan matches the task description
5. Efficiency Reward: Rewards shorter, more efficient plans

Also includes FTCA (Fine-grained Token-level Credit Assignment) for token-level rewards.
"""

import os
import json
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TokenLevelRewardAssigner:
    """
    FTCA: Fine-grained Token-level Credit Assignment

    为规划序列中的不同token类型分配不同的重要性权重
    - 动作token (Navigate/Pick/Place): 最高权重，决定规划的正确性
    - 目标token (Table/Apple等): 高权重，决定规划的语义正确性
    - 结构token (括号/逗号): 低权重，只影响格式
    """

    def __init__(self,
                 action_weight: float = 1.0,
                 target_weight: float = 0.8,
                 structure_weight: float = 0.2):
        self.action_weight = action_weight
        self.target_weight = target_weight
        self.structure_weight = structure_weight
        self.action_names = {'Navigate', 'Pick', 'Place'}

    def tokenize_plan(self, plan_text: str) -> List[Tuple[str, str]]:
        """
        将规划文本分解为带类型的token列表

        Args:
            plan_text: e.g., "Navigate(Table), Pick(Apple)"

        Returns:
            List of (token, type) tuples
            e.g., [("Navigate", "action"), ("(", "structure"), ("Table", "target"), ...]
        """
        tokens = []
        current_token = ""
        plan_text = plan_text.strip()

        i = 0
        while i < len(plan_text):
            char = plan_text[i]

            if char in '(),':
                # 先保存之前积累的token
                if current_token:
                    # 判断是动作还是目标
                    if current_token in self.action_names:
                        tokens.append((current_token, "action"))
                    else:
                        tokens.append((current_token, "target"))
                    current_token = ""

                # 结构token
                tokens.append((char, "structure"))

            elif char == ' ':
                # 空格跳过，但先保存之前的token
                if current_token:
                    if current_token in self.action_names:
                        tokens.append((current_token, "action"))
                    else:
                        tokens.append((current_token, "target"))
                    current_token = ""
            else:
                current_token += char

            i += 1

        # 保存最后一个token
        if current_token:
            if current_token in self.action_names:
                tokens.append((current_token, "action"))
            else:
                tokens.append((current_token, "target"))

        return tokens

    def get_token_weights(self, plan_text: str) -> List[float]:
        """
        获取每个token的权重

        Args:
            plan_text: 规划文本

        Returns:
            List of weights for each token
        """
        tokens = self.tokenize_plan(plan_text)
        weights = []

        for token, token_type in tokens:
            if token_type == "action":
                weights.append(self.action_weight)
            elif token_type == "target":
                weights.append(self.target_weight)
            else:
                weights.append(self.structure_weight)

        return weights

    def compute_weighted_reward(self, plan_text: str, base_reward: float) -> Tuple[float, List[float]]:
        """
        计算加权的token级奖励

        Args:
            plan_text: 规划文本
            base_reward: 基础奖励（序列级）

        Returns:
            (weighted_reward, per_token_rewards)
        """
        tokens = self.tokenize_plan(plan_text)
        weights = self.get_token_weights(plan_text)

        if not weights:
            return base_reward, []

        # 将基础奖励按权重分配到每个token
        total_weight = sum(weights)
        per_token_rewards = [(w / total_weight) * base_reward * len(weights) for w in weights]

        # 加权平均
        weighted_reward = sum(r * w for r, w in zip(per_token_rewards, weights)) / total_weight

        return weighted_reward, per_token_rewards


class HierarchicalPRM:
    """
    Hierarchical Process Reward Model (Simplified)

    Computes rewards at multiple levels:
    - R_format: Format correctness
    - R_action: Action validity (Navigate/Pick/Place)
    - R_task: Task-plan alignment
    - R_efficiency: Plan efficiency (shorter is better)

    Note: State machine validation is removed. The model learns constraints from prompt rules during training.

    Total reward: R_total = w1*R_format + w2*R_action + w3*R_task + w4*R_efficiency
    """

    def __init__(self,
                 w_format: float = 0.15,
                 w_action: float = 0.25,
                 w_task: float = 0.40,
                 w_efficiency: float = 0.20,
                 use_ftca: bool = True,
                 openai_api_key: str = None,
                 openai_base_url: str = None,
                 llm_model: str = "gpt-4o-mini"):
        """
        Args:
            w_format: Weight for format reward (default: 0.15)
            w_action: Weight for action validity reward (default: 0.25)
            w_task: Weight for task reward (default: 0.40, increased from 0.25)
            w_efficiency: Weight for efficiency reward (default: 0.20, increased from 0.15)
            use_ftca: Whether to use token-level credit assignment
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            openai_base_url: OpenAI base URL (optional, for compatible APIs)
            llm_model: LLM model name for task parsing
        """
        self.w_format = w_format
        self.w_action = w_action
        self.w_task = w_task
        self.w_efficiency = w_efficiency
        self.use_ftca = use_ftca
        self.llm_model = llm_model

        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            client_kwargs = {"api_key": api_key}
            if openai_base_url:
                client_kwargs["base_url"] = openai_base_url
            self.llm_client = OpenAI(**client_kwargs)
        else:
            self.llm_client = None
            logger.warning("No OpenAI API key provided, task parsing will return empty results")

        self.valid_actions = {"Navigate", "Pick", "Place"}
        self.ftca = TokenLevelRewardAssigner()

    def parse_plan(self, plan_text: str) -> Tuple[bool, List[str]]:
        """
        Parse plan text into action list

        Args:
            plan_text: e.g., "Navigate(Table), Pick(Apple), Place(Basket)"

        Returns:
            (success, action_list)
        """
        # Remove extra whitespace
        plan_text = plan_text.strip()

        # Split by comma
        actions = [a.strip() for a in plan_text.split(',')]

        # Filter empty strings
        actions = [a for a in actions if a]

        if not actions:
            return False, []

        # Check basic format: Action(Target)
        for action in actions:
            if '(' not in action or ')' not in action:
                return False, []

        return True, actions

    def extract_task_info(self, task: str) -> Dict[str, List[str]]:
        """
        从任务描述中提取结构化信息（使用 LLM）

        Args:
            task: 任务描述，如 "Put the apple on the table into the basket"

        Returns:
            Dict with 'objects', 'source', 'destination', 'action_verb'
        """
        if not self.llm_client:
            logger.warning("No LLM client available, returning empty result")
            return {'objects': [], 'source': [], 'destination': [], 'action_verb': None}

        prompt = f"""Extract structured information from this embodied AI task description.

Task: "{task}"

Extract and return a JSON object with:
- "objects": list of objects to manipulate (e.g., ["apple", "book"])
- "source": list of source locations where objects are located (e.g., ["table", "shelf"])
- "destination": list of target locations where objects should be placed (e.g., ["basket", "drawer"])
- "action_verb": the main action verb (e.g., "put", "move", "take", "bring")

Rules:
1. Only include explicitly mentioned items
2. Use lowercase for all values
3. If not mentioned, use empty list []
4. For action_verb, use null if not clear

Return ONLY valid JSON, no explanation."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured information from task descriptions. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200
            )

            result_text = response.choices[0].message.content.strip()
            # 清理可能的 markdown 代码块
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            result = json.loads(result_text)

            return {
                'objects': result.get('objects', []),
                'source': result.get('source', []),
                'destination': result.get('destination', []),
                'action_verb': result.get('action_verb')
            }
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            return {'objects': [], 'source': [], 'destination': [], 'action_verb': None}

    def compute_format_reward(self, plan_text: str) -> float:
        """
        R_format: Reward for format correctness

        Returns:
            +1.0: Correct format
            +0.5: Parseable but has minor issues
            -1.0: Unparseable
        """
        success, actions = self.parse_plan(plan_text)

        if not success:
            return -1.0

        # Check if all actions follow Action(Target) pattern
        perfect_format = True
        for action in actions:
            # Check for correct parentheses
            if action.count('(') != 1 or action.count(')') != 1:
                perfect_format = False
                break

            # Check that Target is not empty
            target = action.split('(')[1].rstrip(')')
            if not target:
                perfect_format = False
                break

        if perfect_format:
            return 1.0
        else:
            return 0.5

    def compute_action_reward(self, plan_text: str) -> float:
        """
        R_action: Reward for action validity (simplified)

        Checks if all actions are Navigate/Pick/Place

        Returns:
            +1.0: All actions are valid
            Scaled penalty for invalid actions [-1, 1]
        """
        success, actions = self.parse_plan(plan_text)

        if not success:
            return -1.0

        if len(actions) == 0:
            return -1.0

        valid_count = 0
        for action_str in actions:
            # Simple parsing: extract action name
            if '(' in action_str:
                action_name = action_str.split('(')[0].strip()
                if action_name in self.valid_actions:
                    valid_count += 1

        # 返回有效动作的比例，范围 [-1, 1]
        valid_ratio = valid_count / len(actions)
        return valid_ratio * 2 - 1  # 映射到 [-1, 1]

    def compute_task_reward(self, plan_text: str, task: str) -> float:
        """
        R_task: Reward for task-plan alignment (改进版)

        使用结构化信息匹配：
        1. 检查Pick的目标是否与任务中的object匹配
        2. 检查Place的目标是否与任务中的destination匹配
        3. 检查Navigate的目标是否与source/destination匹配

        Args:
            plan_text: Generated plan
            task: Task description

        Returns:
            Reward in range [-1.0, 1.0]
        """
        success, actions = self.parse_plan(plan_text)

        if not success:
            return -1.0

        # 提取任务信息
        task_info = self.extract_task_info(task)

        # 提取规划中的动作和目标
        pick_targets = []
        place_targets = []
        navigate_targets = []

        for action_str in actions:
            # Simple parsing: Action(Target)
            if '(' not in action_str or ')' not in action_str:
                continue

            try:
                action_name = action_str.split('(')[0].strip()
                target = action_str.split('(')[1].rstrip(')').strip()

                if target:
                    target_lower = target.lower()
                    if action_name == "Pick":
                        pick_targets.append(target_lower)
                    elif action_name == "Place":
                        place_targets.append(target_lower)
                    elif action_name == "Navigate":
                        navigate_targets.append(target_lower)
            except:
                continue

        # 计算匹配分数
        scores = []

        # 1. Pick目标应该匹配task中的object
        if task_info['objects']:
            pick_match = any(
                obj in pt or pt in obj
                for obj in task_info['objects']
                for pt in pick_targets
            )
            scores.append(1.0 if pick_match else -0.5)

        # 2. Place目标应该匹配task中的destination
        if task_info['destination']:
            place_match = any(
                dest in plt or plt in dest
                for dest in task_info['destination']
                for plt in place_targets
            )
            scores.append(1.0 if place_match else -0.5)

        # 3. Navigate应该包含source和destination
        if task_info['source'] or task_info['destination']:
            all_locations = task_info['source'] + task_info['destination']
            nav_match_count = sum(
                1 for loc in all_locations
                if any(loc in nt or nt in loc for nt in navigate_targets)
            )
            nav_score = nav_match_count / len(all_locations) if all_locations else 0
            scores.append(nav_score * 2 - 1)  # 映射到 [-1, 1]

        # 计算平均分数
        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def compute_efficiency_reward(self, plan_text: str) -> float:
        """
        R_efficiency: Reward for plan efficiency

        鼓励更短、更高效的规划
        - 标准的pick-and-place任务通常需要4步: Navigate, Pick, Navigate, Place
        - 更短的规划得到奖励
        - 过长的规划得到惩罚

        Returns:
            Reward in range [-1.0, 1.0]
        """
        success, actions = self.parse_plan(plan_text)

        if not success:
            return -1.0

        num_actions = len(actions)

        # 基准：4步是标准
        optimal_length = 4

        if num_actions == 0:
            return -1.0
        elif num_actions <= optimal_length:
            # 短于或等于最优长度，给予正奖励
            return 1.0
        elif num_actions <= optimal_length + 2:
            # 稍长，给予中等奖励
            return 0.5
        elif num_actions <= optimal_length + 4:
            # 较长，给予小惩罚
            return 0.0
        else:
            # 过长，给予惩罚
            return -0.5

    def compute_reward(self, plan_text: str, task: str, image_path: str = None) -> Dict[str, float]:
        """
        Compute total hierarchical reward (simplified)

        Args:
            plan_text: Generated plan text
            task: Task description
            image_path: (Optional) Path to image, for future vision-based rewards

        Returns:
            Dictionary with reward components and total
        """
        r_format = self.compute_format_reward(plan_text)
        r_action = self.compute_action_reward(plan_text)
        r_task = self.compute_task_reward(plan_text, task)
        r_efficiency = self.compute_efficiency_reward(plan_text)

        r_total = (
            self.w_format * r_format +
            self.w_action * r_action +
            self.w_task * r_task +
            self.w_efficiency * r_efficiency
        )

        result = {
            'r_format': r_format,
            'r_action': r_action,
            'r_task': r_task,
            'r_efficiency': r_efficiency,
            'r_total': r_total
        }

        # FTCA: Token级奖励
        if self.use_ftca:
            weighted_reward, token_rewards = self.ftca.compute_weighted_reward(plan_text, r_total)
            result['r_total_ftca'] = weighted_reward
            result['token_rewards'] = token_rewards

        return result

    def compute_token_level_rewards(self, plan_text: str, task: str) -> List[Tuple[str, float]]:
        """
        获取每个token的奖励（用于细粒度分析）

        Returns:
            List of (token, reward) tuples
        """
        base_reward = self.compute_reward(plan_text, task)['r_total']
        tokens = self.ftca.tokenize_plan(plan_text)
        weights = self.ftca.get_token_weights(plan_text)

        if not tokens or not weights:
            return []

        total_weight = sum(weights)
        token_rewards = []

        for (token, token_type), weight in zip(tokens, weights):
            # 按权重分配基础奖励
            token_reward = (weight / total_weight) * base_reward
            token_rewards.append((token, token_reward))

        return token_rewards

    def __call__(self, plan_text: str, task: str, image_path: str = None) -> float:
        """Shortcut to get total reward"""
        rewards = self.compute_reward(plan_text, task, image_path)
        return rewards['r_total']


def test_reward_model():
    """Test cases for H-PRM"""
    prm = HierarchicalPRM()

    task = "Put the apple on the table into the basket"

    logger.info("=" * 60)
    logger.info("Testing Hierarchical Process Reward Model (H-PRM)")
    logger.info("=" * 60)

    # Test 1: Perfect plan
    plan1 = "Navigate(Table), Pick(Apple), Navigate(Basket), Place(Basket)"
    rewards1 = prm.compute_reward(plan1, task)
    logger.info("\nTest 1 - Perfect plan:")
    logger.info(f"  Plan: {plan1}")
    logger.info(f"  Task: {task}")
    logger.info(f"  Rewards: {rewards1}")

    # Test 2: Invalid action sequence (Pick before Navigate)
    plan2 = "Pick(Apple), Navigate(Table), Place(Basket)"
    rewards2 = prm.compute_reward(plan2, task)
    logger.info("\nTest 2 - Pick before Navigate:")
    logger.info(f"  Plan: {plan2}")
    logger.info(f"  Rewards: {rewards2}")

    # Test 3: Unknown action
    plan3 = "Navigate(Table), Grab(Apple), Navigate(Basket), Place(Basket)"
    rewards3 = prm.compute_reward(plan3, task)
    logger.info("\nTest 3 - Unknown action (Grab):")
    logger.info(f"  Plan: {plan3}")
    logger.info(f"  Rewards: {rewards3}")

    # Test 4: Wrong format
    plan4 = "Navigate Table, Pick Apple"
    rewards4 = prm.compute_reward(plan4, task)
    logger.info("\nTest 4 - Wrong format (missing parentheses):")
    logger.info(f"  Plan: {plan4}")
    logger.info(f"  Rewards: {rewards4}")

    # Test 5: Task mismatch
    plan5 = "Navigate(Chair), Pick(Book), Navigate(Shelf), Place(Shelf)"
    rewards5 = prm.compute_reward(plan5, task)
    logger.info("\nTest 5 - Task mismatch (different objects):")
    logger.info(f"  Plan: {plan5}")
    logger.info(f"  Rewards: {rewards5}")

    # Test 6: Inefficient plan (too many steps)
    plan6 = "Navigate(Table), Navigate(Table), Pick(Apple), Navigate(Basket), Navigate(Basket), Place(Basket)"
    rewards6 = prm.compute_reward(plan6, task)
    logger.info("\nTest 6 - Inefficient plan (redundant navigates):")
    logger.info(f"  Plan: {plan6}")
    logger.info(f"  Rewards: {rewards6}")

    # Test 7: Token-level rewards (FTCA)
    logger.info("\n" + "=" * 60)
    logger.info("Testing FTCA (Token-level Credit Assignment)")
    logger.info("=" * 60)
    token_rewards = prm.compute_token_level_rewards(plan1, task)
    logger.info(f"\nPlan: {plan1}")
    logger.info("Token-level rewards:")
    for token, reward in token_rewards:
        logger.info(f"  '{token}': {reward:.4f}")


if __name__ == "__main__":
    test_reward_model()
