"""
Prompt templates for Qwen2.5-VL training

Provides consistent prompt formatting for:
1. SFT training
2. RL inference
3. Evaluation
"""

from typing import List, Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)



class PromptTemplate:
    """Prompt template manager for embodied planning tasks"""

    def __init__(self):
        # 原有的完整计划生成 system prompt
        self.system_prompt = """You are an embodied AI planning assistant. Given a scene image and a task, generate a step-by-step action plan.

## Available Actions
- Navigate(location): Move to a target location
- Pick(object): Pick up an object (must be at the object's location first)
- Place(location): Place the held object at a location

## Action Rules (IMPORTANT)
1. You must Navigate to an object's location before you can Pick it
2. You can only Pick one object at a time
3. You must be holding an object before you can Place it
4. After Place, your hands are empty and you can Pick another object

## Valid Action Patterns
- Navigate → Pick → Navigate → Place (standard pick-and-place)
- Navigate → Pick → Place (place at current location)
- Navigate → Navigate → Pick → ... (multiple navigation is allowed)

## Invalid Patterns (NEVER do these)
- Pick without Navigate first (you're not at the object!)
- Place without Pick first (you're not holding anything!)
- Pick while already holding something (you only have two hands!)

## Output Format
Action1(Target1), Action2(Target2), Action3(Target3), ...

## Example
Task: Put the apple on the table into the basket
Plan: Navigate(Table), Pick(Apple), Navigate(Basket), Place(Basket)"""

        # 轨迹模式：逐步决策的 system prompt
        self.system_prompt_trajectory = """You are an embodied AI agent that makes decisions step by step in a household environment.

## Available Actions
- Navigate(location): Move to a target location (e.g., Navigate(CounterTop), Navigate(Fridge))
- Pick(object): Pick up an object (must be at the object's location first)
- Place(receptacle): Place the held object on/in a receptacle
- Open(object): Open a container or appliance (e.g., Open(Fridge), Open(Cabinet))
- Close(object): Close a container or appliance
- TaskCompleted(): Indicate the task is finished

## Action Rules
1. You must Navigate to an object's location before you can Pick it
2. You can only hold one object at a time
3. You must be holding an object before you can Place it
4. Some receptacles (Fridge, Cabinet) need to be Open before Place
5. Call TaskCompleted() when all subtasks are done

## Your Task
Given:
- An image of your current view
- The task description
- Actions you have already completed

You must:
1. Carefully observe the current scene in the image
2. Think about what you see and what the task requires
3. Decide the single next action to take

## Output Format
Thinking: [Describe what you observe and your reasoning]

Next Action: [Single action to execute]

## Example
Task: Place the apple on the table
Completed actions: Navigate(CounterTop) -> Pick(Apple)

Thinking: I have picked up the Apple from the CounterTop. Now I can see the Table in my view. To complete the task, I need to navigate to the Table and place the Apple there.

Next Action: Navigate(Table)"""

    def format_sft_prompt(self, task: str) -> str:
        """
        Format prompt for SFT training

        Args:
            task: Task description

        Returns:
            Formatted prompt string
        """
        return f"Task: {task}\nPlan:"

    def format_trajectory_prompt(self, task: str, action_history: List[str], step: int = 0) -> str:
        """
        Format prompt for trajectory-based step-by-step prediction

        Args:
            task: Task description
            action_history: List of completed actions
            step: Current step number

        Returns:
            Formatted prompt string
        """
        if step == 0 or not action_history:
            return f"Task: {task}\n\nThis is the initial state. What action should I take first?"
        else:
            history_str = " -> ".join(action_history)
            return f"Task: {task}\n\nCompleted actions: {history_str}\n\nBased on the current view, what action should I take next?"

    def format_messages_for_trajectory(self, task: str, action_history: List[str] = None, step: int = 0) -> List[Dict[str, Any]]:
        """
        Format messages in Qwen2.5-VL chat format for trajectory mode

        Args:
            task: Task description
            action_history: List of completed actions
            step: Current step number

        Returns:
            List of message dicts
        """
        input_text = self.format_trajectory_prompt(task, action_history or [], step)

        messages = [
            {
                "role": "system",
                "content": self.system_prompt_trajectory
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": input_text}
                ]
            }
        ]
        return messages

    def parse_trajectory_response(self, response: str) -> Dict[str, str]:
        """
        Parse model response from trajectory mode

        Args:
            response: Model output with Thinking and Next Action

        Returns:
            Dict with 'thinking' and 'next_action'
        """
        thinking = ""
        next_action = ""

        # Try to parse structured format
        if "Thinking:" in response:
            parts = response.split("Next Action:")
            if len(parts) == 2:
                thinking = parts[0].replace("Thinking:", "").strip()
                next_action = parts[1].strip()
            else:
                # Fallback: try to extract thinking only
                thinking = response.split("Thinking:")[-1].strip()

        # If no structured format, treat whole response as action
        if not next_action:
            # Try to find action pattern like Navigate(X), Pick(Y), etc.
            import re
            action_pattern = r'(Navigate|Pick|Place|Open|Close|TaskCompleted)\([^)]*\)'
            matches = re.findall(action_pattern, response)
            if matches:
                # Find the full action string
                full_matches = re.findall(r'(Navigate|Pick|Place|Open|Close|TaskCompleted)\([^)]*\)', response)
                if full_matches:
                    next_action = full_matches[-1] if isinstance(full_matches[-1], str) else full_matches[-1][0]

        return {
            "thinking": thinking,
            "next_action": next_action
        }

    def format_messages_for_qwen(self, task: str, image_path: str = None) -> List[Dict[str, Any]]:
        """
        Format messages in Qwen2.5-VL chat format

        Args:
            task: Task description
            image_path: Path to image (optional, handled by processor)

        Returns:
            List of message dicts
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Image will be provided separately
                    {"type": "text", "text": f"Task: {task}"}
                ]
            }
        ]
        return messages

    def format_for_training(self, task: str, plan: List[str]) -> Dict[str, str]:
        """
        Format data for training (input + target)

        Args:
            task: Task description
            plan: List of actions

        Returns:
            Dict with 'prompt' and 'completion'
        """
        prompt = self.format_sft_prompt(task)
        completion = ", ".join(plan)

        return {
            "prompt": prompt,
            "completion": completion
        }

    def parse_response(self, response: str) -> List[str]:
        """
        Parse model response into action list

        Args:
            response: Model output, e.g., "Navigate(Table), Pick(Apple), Place(Basket)"

        Returns:
            List of actions
        """
        # Clean up response
        response = response.strip()

        # Remove any additional explanation after the plan
        # (in case model generates extra text)
        if '\n' in response:
            response = response.split('\n')[0]

        # Split by comma
        actions = [a.strip() for a in response.split(',')]

        # Filter empty strings
        actions = [a for a in actions if a]

        return actions

    def format_few_shot_examples(self) -> str:
        """
        Generate few-shot examples for in-context learning

        Returns:
            Formatted few-shot examples
        """
        examples = [
            {
                "task": "Put the apple on the table into the basket",
                "plan": "Navigate(Table), Pick(Apple), Navigate(Basket), Place(Basket)"
            },
            {
                "task": "Move the book from the shelf to the desk",
                "plan": "Navigate(Shelf), Pick(Book), Navigate(Desk), Place(Desk)"
            },
            {
                "task": "Place the cup from the counter into the sink",
                "plan": "Navigate(Counter), Pick(Cup), Navigate(Sink), Place(Sink)"
            }
        ]

        formatted = "Here are some examples:\n\n"
        for i, ex in enumerate(examples, 1):
            formatted += f"Example {i}:\n"
            formatted += f"Task: {ex['task']}\n"
            formatted += f"Plan: {ex['plan']}\n\n"

        return formatted


class Qwen25VLProcessor:
    """
    Wrapper for Qwen2.5-VL processor with prompt formatting

    Handles image processing and text tokenization

    注意：此类仅用于推理，不用于训练。训练时labels的处理在collate_fn中完成。
    """

    def __init__(self, processor, tokenizer):
        """
        Args:
            processor: Qwen2.5-VL AutoProcessor instance
            tokenizer: Qwen2.5-VL AutoTokenizer instance
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.prompt_template = PromptTemplate()

    def prepare_inputs_for_inference(self, image, task: str) -> Dict[str, Any]:
        """
        Prepare inputs for model inference (not training)

        Args:
            image: PIL Image
            task: Task description

        Returns:
            Dict with input_ids, pixel_values, attention_mask
        """
        # Format messages
        messages = self.prompt_template.format_messages_for_qwen(task)

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process image and text
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        return inputs


def test_prompt_template():
    """Test prompt template formatting"""
    template = PromptTemplate()

    task = "Put the apple on the table into the basket"
    plan = ["Navigate(Table)", "Pick(Apple)", "Navigate(Basket)", "Place(Basket)"]

    logger.info("=== SFT Prompt ===")
    sft_prompt = template.format_sft_prompt(task)
    logger.info(sft_prompt)
    logger.info()

    logger.info("=== Qwen Messages Format ===")
    messages = template.format_messages_for_qwen(task)
    import json
    logger.info(json.dumps(messages, indent=2))
    logger.info()

    logger.info("=== Training Format ===")
    training_data = template.format_for_training(task, plan)
    logger.info(f"Prompt: {training_data['prompt']}")
    logger.info(f"Completion: {training_data['completion']}")
    logger.info()

    logger.info("=== Parse Response ===")
    response = "Navigate(Table), Pick(Apple), Navigate(Basket), Place(Basket)"
    parsed = template.parse_response(response)
    logger.info(f"Response: {response}")
    logger.info(f"Parsed: {parsed}")
    logger.info()

    logger.info("=== Few-shot Examples ===")
    few_shot = template.format_few_shot_examples()
    logger.info(few_shot)


if __name__ == "__main__":
    test_prompt_template()
