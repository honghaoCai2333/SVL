"""
Simplified SFT训练脚本 - 使用Transformers原生Trainer
不依赖TRL，避免兼容性问题
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import SFTDataset
from src.utils.prompt_template import PromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SFTTrainer:
    """Simplified SFT Trainer using Transformers Trainer"""

    def __init__(self, config: dict):
        self.config = config
        self.prompt_template = PromptTemplate()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model, processor, tokenizer
        self.load_model()

    def load_model(self):
        """Load Qwen2.5-VL model with LoRA"""
        model_name = self.config['model']['name']
        logger.info(f"Loading model: {model_name}")

        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

        # 确保pad_token存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.config['training'].get('bf16', True) else torch.float16,
            device_map=None,  # DeepSpeed会处理设备分配
            trust_remote_code=True,
            local_files_only=True
        )

        # Enable gradient checkpointing before LoRA (节省显存)
        if self.config['training'].get('gradient_checkpointing', False):
            logger.info("Enabling gradient checkpointing...")
            self.model.gradient_checkpointing_enable()

        # Freeze vision encoder if specified
        if self.config['model'].get('freeze_vision_encoder', True):
            logger.info("Freezing vision encoder...")
            for param in self.model.visual.parameters():
                param.requires_grad = False

        # Setup LoRA
        if self.config['model'].get('use_lora', True):
            logger.info("Setting up LoRA...")
            lora_config = LoraConfig(
                r=self.config['lora']['rank'],
                lora_alpha=self.config['lora']['alpha'],
                target_modules=self.config['lora']['target_modules'],
                lora_dropout=self.config['lora']['dropout'],
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

    def load_data(self):
        """Load and prepare dataset"""
        train_path = Path(self.config['data']['train_path'])
        val_path = Path(self.config['data']['val_path'])

        logger.info(f"Loading training data from: {train_path}")
        logger.info(f"Loading validation data from: {val_path}")

        # Load datasets
        self.train_dataset = SFTDataset(
            jsonl_path=train_path,
            image_processor=self.processor.image_processor,
            tokenizer=self.tokenizer
        )

        self.val_dataset = SFTDataset(
            jsonl_path=val_path,
            image_processor=self.processor.image_processor,
            tokenizer=self.tokenizer
        )

        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")

    def collate_fn(self, batch):
        """Custom collate function for batching"""
        images = [item['image'] for item in batch]
        tasks = [item['input_text'] for item in batch]
        target_texts = [item['target_text'] for item in batch]

        # 构建完整的对话格式（包含assistant回复）
        # 使用简单的 system prompt，只输出动作，不要 Thinking
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

        full_texts = []
        for task, target in zip(tasks, target_texts):
            messages = [
                {"role": "system", "content": simple_system_prompt},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": task}
                ]},
                {"role": "assistant", "content": target}
            ]
            full_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            full_texts.append(full_text)

        # 构建只有prompt的文本（用于计算prompt长度）
        prompt_texts = []
        for task in tasks:
            messages = [
                {"role": "system", "content": simple_system_prompt},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": task}
                ]}
            ]
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_texts.append(prompt_text)

        # 处理完整序列（包含图像）
        inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,  # 关闭truncation，避免截断图像token
        )

        # 计算每个样本的prompt长度，用于设置labels
        labels = inputs['input_ids'].clone()

        # 逐个处理每个样本，获取实际的prompt token长度（包含图像token）
        for i in range(len(batch)):
            # 单独处理每个样本的prompt，获取真实长度
            prompt_input = self.processor(
                text=[prompt_texts[i]],
                images=[images[i]],
                return_tensors="pt",
                padding=False,
                truncation=False,  # 关闭truncation
            )
            prompt_len = prompt_input['input_ids'].shape[1]

            # 将prompt部分的labels设为-100
            labels[i, :prompt_len] = -100

        # padding token也设为-100
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        inputs['labels'] = labels

        return inputs

    def train(self):
        """Run training with Transformers Trainer"""
        # Load data
        self.load_data()

        # Setup TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            eval_strategy="steps",  # 新版transformers改名了
            save_strategy="steps",
            bf16=self.config['training'].get('bf16', True),
            fp16=not self.config['training'].get('bf16', True),
            dataloader_num_workers=self.config['training'].get('num_workers', 4),
            remove_unused_columns=False,
            report_to=self.config['training'].get('report_to', []),
            run_name=self.config['training'].get('run_name', 'sft_qwen2.5vl'),
            deepspeed=self.config['training'].get('deepspeed', None),
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            gradient_checkpointing=self.config['training'].get('gradient_checkpointing', False),
        )

        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collate_fn,
        )

        # Train
        logger.info("Starting SFT training...")
        trainer.train()

        # Save final model
        final_output_dir = os.path.join(self.config['training']['output_dir'], "final")
        logger.info(f"Saving final model to: {final_output_dir}")
        trainer.save_model(final_output_dir)
        self.processor.save_pretrained(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)

        logger.info("SFT training complete!")


def main():
    parser = argparse.ArgumentParser(description="Simplified SFT training for Qwen2.5-VL")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml",
                        help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer and train
    trainer = SFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()