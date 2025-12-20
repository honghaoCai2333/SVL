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
        self.system_prompt = """You are an embodied AI planning assistant. Given a scene image and a task, generate a step-by-step action plan using only these actions:
- Navigate(target): Move to a target location
- Pick(object): Pick up an object
- Place(receptacle): Place the held object

Output format: Action1(Target1), Action2(Target2), ...
Example: Navigate(Table), Pick(Apple), Navigate(Basket), Place(Basket)"""

    def format_sft_prompt(self, task: str) -> str:
        """
        Format prompt for SFT training

        Args:
            task: Task description

        Returns:
            Formatted prompt string
        """
        return f"Task: {task}\nPlan:"

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
