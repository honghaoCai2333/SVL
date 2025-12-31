"""
Supervised Fine-Tuning (SFT) script for trajectory-based step-by-step prediction

This script trains Qwen2.5-VL to predict actions step by step with chain-of-thought reasoning.

Usage:
    python scripts/train_sft_trajectory.py --config configs/sft_trajectory_config.yaml
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

from src.data.dataset import TrajectoryDataset
from src.utils.prompt_template import PromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrajectorySFTTrainer:
    """Trainer for trajectory-based Supervised Fine-Tuning"""

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

        # Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # DeepSpeed needs Trainer to handle device mapping
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
        """Load and prepare trajectory dataset"""
        train_path = Path(self.config['data']['train_path'])
        val_path = Path(self.config['data']['val_path'])

        logger.info(f"Loading training data from: {train_path}")
        logger.info(f"Loading validation data from: {val_path}")

        # Get trajectory-specific settings
        include_thinking = self.config['data'].get('include_thinking', True)

        # Load datasets
        self.train_dataset = TrajectoryDataset(
            jsonl_path=train_path,
            image_processor=self.processor.image_processor,
            tokenizer=self.tokenizer,
            max_length=self.config['training'].get('max_length', 1024),
            include_thinking=include_thinking
        )

        self.val_dataset = TrajectoryDataset(
            jsonl_path=val_path,
            image_processor=self.processor.image_processor,
            tokenizer=self.tokenizer,
            max_length=self.config['training'].get('max_length', 1024),
            include_thinking=include_thinking
        )

        logger.info(f"Train samples (steps): {len(self.train_dataset)}")
        logger.info(f"Val samples (steps): {len(self.val_dataset)}")

    def collate_fn(self, batch):
        """
        Custom collate function for trajectory batching

        Handles:
        - Current view image
        - Task + action history as input
        - Thinking + next action as target
        """
        images = [item['image'] for item in batch]
        input_texts = [item['input_text'] for item in batch]
        target_texts = [item['target_text'] for item in batch]

        # Build full conversation (including assistant response)
        full_texts = []
        for input_text, target in zip(input_texts, target_texts):
            messages = [
                {"role": "system", "content": self.prompt_template.system_prompt_trajectory},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": input_text}
                ]},
                {"role": "assistant", "content": target}
            ]
            full_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            full_texts.append(full_text)

        # Build prompt-only texts (for computing prompt length)
        prompt_texts = []
        for input_text in input_texts:
            messages = [
                {"role": "system", "content": self.prompt_template.system_prompt_trajectory},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": input_text}
                ]}
            ]
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt_texts.append(prompt_text)

        # Process full sequences (with images)
        inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['training']['max_length']
        )

        # Compute labels: mask prompt tokens with -100
        labels = inputs['input_ids'].clone()

        # Process each sample to get actual prompt length (including image tokens)
        for i in range(len(batch)):
            prompt_input = self.processor(
                text=[prompt_texts[i]],
                images=[images[i]],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.config['training']['max_length']
            )
            prompt_len = prompt_input['input_ids'].shape[1]

            # Mask prompt portion of labels
            labels[i, :prompt_len] = -100

        # Also mask padding tokens
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

        # Setup SFT config
        sft_config = SFTConfig(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training'].get('per_device_eval_batch_size', 1),
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01),
            warmup_steps=self.config['training'].get('warmup_steps', 100),
            logging_steps=self.config['training'].get('logging_steps', 10),
            save_steps=self.config['training'].get('save_steps', 500),
            eval_steps=self.config['training'].get('eval_steps', 500),
            bf16=use_bf16,
            fp16=not use_bf16,
            dataloader_num_workers=self.config['training'].get('num_workers', 4),
            remove_unused_columns=False,
            report_to=self.config['training'].get('report_to', ['tensorboard']),
            run_name=self.config['training'].get('run_name', 'sft_trajectory_qwen2.5vl'),
            max_seq_length=self.config['training']['max_length'],
            dataset_text_field="text",
            packing=False,
            deepspeed=deepspeed_config,
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
        logger.info("Starting trajectory-based SFT training...")
        trainer.train()

        # Save final model
        final_output_dir = os.path.join(self.config['training']['output_dir'], "final")
        logger.info(f"Saving final model to: {final_output_dir}")
        trainer.save_model(final_output_dir)
        self.processor.save_pretrained(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)

        logger.info("Trajectory SFT training complete!")


def main():
    parser = argparse.ArgumentParser(description="Trajectory-based SFT training for Qwen2.5-VL")
    parser.add_argument("--config", type=str, default="configs/sft_trajectory_config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer and train
    trainer = TrajectorySFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
