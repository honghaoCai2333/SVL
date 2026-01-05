"""
SFT 模型推理脚本

用法:
    python scripts/inference.py \
        --model_path outputs/sft/final \
        --image_path data/images/xxx.png \
        --task "Fetch a ToiletPaper from the room."
"""

import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)
from peft import PeftModel

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.prompt_template import PromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SFTInference:
    """SFT 模型推理器（单步预测模式）"""

    def __init__(self, model_path: str, base_model_path: str = None):
        """
        Args:
            model_path: 训练好的模型路径（LoRA adapter）
            base_model_path: 基础模型路径（如果None，从model_path读取）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # 加载模型
        self.load_model(model_path, base_model_path)

        # 初始化 prompt template
        self.prompt_template = PromptTemplate()

    def load_model(self, model_path: str, base_model_path: str = None):
        """加载训练好的模型"""
        logger.info(f"Loading model from: {model_path}")

        # 确定基础模型路径
        if base_model_path is None:
            # 尝试从 adapter_config.json 读取
            import json
            adapter_config_path = Path(model_path) / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                    base_model_path = adapter_config.get('base_model_name_or_path')

        if base_model_path is None:
            raise ValueError("无法确定基础模型路径，请通过 --base_model_path 指定")

        logger.info(f"Base model: {base_model_path}")

        # 加载 processor 和 tokenizer
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        # 加载基础模型
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )

        # 加载 LoRA adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            model_path,
            device_map="auto"
        )

        self.model.eval()
        logger.info("✓ 模型加载完成")

    def predict_next_action(
            self,
            image_path: str,
            task: str,
            action_history: list = None,
            max_new_tokens: int = 50
    ) -> str:
        """
        预测下一步动作

        Args:
            image_path: 图像路径
            task: 任务描述
            action_history: 已完成的动作历史（可选）
            max_new_tokens: 最大生成token数

        Returns:
            预测的下一步动作
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 构建输入文本
        action_history = action_history or []
        if action_history:
            history_str = " -> ".join(action_history)
            input_text = f"Task: {task}\nCompleted actions: {history_str}\n\nBased on the current view, what action should I take next?"
        else:
            input_text = f"Task: {task}\n\nThis is the initial state. What action should I take first?"

        # 使用和训练时一致的 simple system prompt
        simple_system_prompt = """You are an embodied AI agent. Given a scene image, task description, and action history, predict the NEXT ACTION.

Available Actions:
- Navigate(location): Move to a location (e.g., Navigate(CounterTop), Navigate(Fridge))
- Pick(object): Pick up an object (e.g., Pick(Apple), Pick(ToiletPaper))
- Place(receptacle): Place held object (e.g., Place(GarbageCan), Place(SinkBasin))
- Open(object): Open container/appliance (e.g., Open(Fridge), Open(Drawer))
- Close(object): Close container/appliance (e.g., Close(Cabinet))
- TaskCompleted(): Task is finished

Rules:
1. Navigate before Pick
2. Hold object before Place
3. Output ONLY the action, nothing else

Output: Action(Target)"""

        # 构建消息
        messages = [
            {
                "role": "system",
                "content": simple_system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": input_text}
                ]
            }
        ]

        # 应用 chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 处理输入
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        )

        # 移动到 GPU
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 贪心解码
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],  # 只解码生成的部分
            skip_special_tokens=True
        )

        # 解析输出：提取 "Next Action: XXX" 部分
        parsed = self._parse_action_from_response(generated_text)

        return parsed

    def _parse_action_from_response(self, response: str) -> str:
        """
        从模型输出中提取动作

        输出格式可能是:
        1. "Thinking: ... Next Action: Navigate(Table)"
        2. "Navigate(Table)"
        3. "Next Action: Navigate(Table)"

        Returns:
            提取的动作字符串
        """
        import re

        response = response.strip()

        # 方法1: 查找 "Next Action:" 后面的内容
        if "Next Action:" in response:
            parts = response.split("Next Action:")
            if len(parts) > 1:
                action_part = parts[-1].strip()
                # 提取第一行（避免后续的thinking）
                action = action_part.split('\n')[0].strip()
                return action

        # 方法2: 使用正则提取动作模式
        # 匹配 Navigate(...), Pick(...), Place(...), Open(...), Close(...), Done 等
        action_pattern = r'(Navigate|Pick|Place|Open|Close|TaskCompleted|Done)\s*\([^)]*\)|Done|TaskCompleted\(\)'
        matches = re.findall(action_pattern, response)
        if matches:
            # 返回第一个匹配的完整动作
            full_match = re.search(action_pattern, response)
            if full_match:
                return full_match.group(0)

        # 方法3: 如果都失败，返回原始内容的第一行
        return response.split('\n')[0].strip()

    def predict_complete_plan(
            self,
            image_path: str,
            task: str,
            max_steps: int = 20
    ) -> list:
        """
        自回归生成完整计划（逐步预测直到Done）

        Args:
            image_path: 初始图像路径
            task: 任务描述
            max_steps: 最大步数

        Returns:
            完整的动作序列
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Task: {task}")
        logger.info(f"Image: {image_path}")
        logger.info(f"{'=' * 60}\n")

        action_history = []

        for step in range(max_steps):
            # 预测下一步
            next_action = self.predict_next_action(
                image_path=image_path,
                task=task,
                action_history=action_history
            )

            logger.info(f"Step {step + 1}: {next_action}")

            # 检查是否完成
            if next_action.strip().lower() in ['done', 'taskcompleted()', 'task completed']:
                logger.info("✓ 任务完成")
                break

            # 添加到历史
            action_history.append(next_action)

        logger.info(f"\n完整计划: {' -> '.join(action_history)}\n")

        return action_history


def main():
    parser = argparse.ArgumentParser(description="SFT 模型推理")
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
        help="基础模型路径（可选，会从adapter_config.json读取）"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="图像路径"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="任务描述"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="complete",
        choices=["single", "complete"],
        help="推理模式: single=单步, complete=完整计划"
    )
    parser.add_argument(
        "--history",
        type=str,
        nargs="*",
        default=[],
        help="动作历史（仅在single模式下使用）"
    )

    args = parser.parse_args()

    # 创建推理器
    inferencer = SFTInference(
        model_path=args.model_path,
        base_model_path=args.base_model_path
    )

    # 执行推理
    if args.mode == "single":
        # 单步预测
        next_action = inferencer.predict_next_action(
            image_path=args.image_path,
            task=args.task,
            action_history=args.history
        )
        print(f"\n预测的下一步动作: {next_action}\n")
    else:
        # 完整计划
        plan = inferencer.predict_complete_plan(
            image_path=args.image_path,
            task=args.task
        )
        print(f"\n完整计划:")
        for i, action in enumerate(plan, 1):
            print(f"  {i}. {action}")
        print()


if __name__ == "__main__":
    main()