"""
Supervised Fine-Tuning (SFT) script for Qwen2.5-VL on embodied planning task

Usage:
    python scripts/train_sft.py --config configs/sft_config.yaml
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
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer as TRLSFTTrainer

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import SFTDataset
from src.utils.prompt_template import PromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SFTTrainer:
    """Trainer for Supervised Fine-Tuning"""

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
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 确保pad_token存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 根据是否使用 DeepSpeed 决定 device_map
        # DeepSpeed 需要 Trainer 自己处理设备分配
        use_deepspeed = self.config['training'].get('deepspeed', None) is not None
        device_map = None if use_deepspeed else "auto"

        # Load model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.config['training'].get('bf16', True) else torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )

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

            # Prepare model for k-bit training if using quantization
            if self.config['model'].get('load_in_8bit', False):
                self.model = prepare_model_for_kbit_training(self.model)

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
        """
        Custom collate function for batching

        正确处理VLM的因果语言建模：
        - 将完整序列 (prompt + response) tokenize
        - prompt部分的labels设为-100（不计算loss）
        - 只在response部分计算loss

        关键：对于VLM，需要通过processor同时处理图像和文本，
        图像会被转换为特殊的vision tokens插入到序列中
        """
        images = [item['image'] for item in batch]
        tasks = [item['input_text'] for item in batch]
        target_texts = [item['target_text'] for item in batch]

        # 构建完整的对话格式（包含assistant回复）
        full_texts = []
        for task, target in zip(tasks, target_texts):
            messages = [
                {"role": "system", "content": self.prompt_template.system_prompt},
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
                {"role": "system", "content": self.prompt_template.system_prompt},
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
            truncation=True,
            max_length=self.config['training']['max_length']
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
                truncation=True,
                max_length=self.config['training']['max_length']
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
        """Run training with TRL SFTTrainer"""
        # Load data
        self.load_data()

        # Get bf16 setting with default
        use_bf16 = self.config['training'].get('bf16', True)

        # DeepSpeed config path (optional)
        deepspeed_config = self.config['training'].get('deepspeed', None)

        # Setup SFT config (TRL's SFTConfig)
        # 注意：只使用 SFTConfig 明确支持的参数
        sft_config = SFTConfig(
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
            # Note: eval_strategy, save_strategy 等参数在某些 TRL 版本中可能不支持
            # 如果报错，请移除下面的可选参数
            bf16=use_bf16,
            fp16=not use_bf16,
            dataloader_num_workers=self.config['training'].get('num_workers', 4),
            remove_unused_columns=False,
            report_to=self.config['training'].get('report_to', ['tensorboard']),
            run_name=self.config['training'].get('run_name', 'sft_qwen2.5vl'),
            max_seq_length=self.config['training']['max_length'],
            dataset_text_field="text",  # Will be handled by formatting function
            packing=False,  # Don't pack samples for multimodal
            deepspeed=deepspeed_config,  # DeepSpeed config
        )

        # Create TRL SFTTrainer
        trainer = TRLSFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collate_fn,
            tokenizer=self.tokenizer,
        )

        # Train
        logger.info("Starting SFT training with TRL...")
        trainer.train()

        # Save final model
        final_output_dir = os.path.join(self.config['training']['output_dir'], "final")
        logger.info(f"Saving final model to: {final_output_dir}")
        trainer.save_model(final_output_dir)
        self.processor.save_pretrained(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)

        logger.info("SFT training complete!")


def main():
    parser = argparse.ArgumentParser(description="SFT training for Qwen2.5-VL")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer and train
    trainer = SFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
