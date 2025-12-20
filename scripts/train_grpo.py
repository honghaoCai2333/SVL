"""
GRPO (Group Relative Policy Optimization) training script for Qwen2.5-VL

由于TRL的GRPOTrainer对VLM支持有限，这里实现自定义的GRPO训练循环。

GRPO核心思想：
1. 对每个prompt生成多个候选
2. 用奖励函数对候选打分
3. 计算组内相对优势 A_i = (r_i - mean(r)) / std(r)
4. 用优势加权的策略梯度更新模型

Usage:
    python scripts/train_grpo.py --config configs/grpo_config.yaml
"""

import sys
import argparse
import yaml
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from peft import PeftModel
from tqdm import tqdm
import json
import numpy as np
from typing import List, Dict, Tuple
import wandb

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import RLDataset
from src.models.reward_model import HierarchicalPRM
from src.utils.prompt_template import PromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GRPOTrainerForVLM:
    """
    Custom GRPO Trainer for Vision-Language Models

    实现GRPO算法用于VLM的规划任务训练
    """

    def __init__(self, config: dict):
        self.config = config
        self.prompt_template = PromptTemplate()

        # 初始化奖励模型
        self.reward_model = HierarchicalPRM(
            w_format=config['reward']['w_format'],
            w_action=config['reward']['w_action'],
            w_transition=config['reward']['w_transition'],
            w_task=config['reward']['w_task'],
            w_efficiency=config['reward'].get('w_efficiency', 0.15),
            use_ftca=config['reward'].get('use_ftca', True)
        )

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load models
        self.load_models()

        # Setup optimizer
        self.setup_optimizer()

    def load_models(self):
        """Load policy model and reference model"""
        sft_model_path = self.config['model']['sft_model_path']
        base_model_name = self.config['model']['base_model_name']

        logger.info(f"Loading base model: {base_model_name}")
        logger.info(f"Loading LoRA weights from: {sft_model_path}")

        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(sft_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load policy model (trainable)
        self.policy_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if self.config['training'].get('bf16', True) else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.policy_model = PeftModel.from_pretrained(
            self.policy_model,
            sft_model_path,
            is_trainable=True
        )

        # Freeze vision encoder
        if self.config['model'].get('freeze_vision_encoder', True):
            logger.info("Freezing vision encoder...")
            for param in self.policy_model.base_model.model.visual.parameters():
                param.requires_grad = False

        # Load reference model (frozen, for KL penalty)
        logger.info("Loading reference model (frozen)...")
        self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if self.config['training'].get('bf16', True) else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.ref_model = PeftModel.from_pretrained(
            self.ref_model,
            sft_model_path,
            is_trainable=False
        )
        self.ref_model.eval()

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.policy_model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.policy_model.parameters()),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )

    def load_data(self):
        """Load training dataset"""
        train_path = Path(self.config['data']['train_path'])
        logger.info(f"Loading training data from: {train_path}")

        self.train_dataset = RLDataset(
            jsonl_path=train_path,
            image_processor=None,  # We'll process in generate
            tokenizer=self.tokenizer
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['per_device_train_batch_size'],
            shuffle=True,
            collate_fn=self.collate_fn
        )

        # Load negative samples for injection
        self.negative_samples = {}
        negative_path = self.config['data'].get('negative_path')
        if negative_path and Path(negative_path).exists():
            logger.info(f"Loading negative samples from: {negative_path}")
            with open(negative_path, 'r', encoding='utf-8') as f:
                for line in f:
                    neg = json.loads(line)
                    img_path = neg['image']
                    if img_path not in self.negative_samples:
                        self.negative_samples[img_path] = []
                    self.negative_samples[img_path].append(neg['plan'])

        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Negative samples loaded for {len(self.negative_samples)} images")

    def collate_fn(self, batch):
        """Collate function for dataloader"""
        return {
            'images': [item['image'] for item in batch],
            'tasks': [item['task'] for item in batch],
            'reference_plans': [item['reference_plan'] for item in batch],
            'image_paths': [item['image_path'] for item in batch]
        }

    def prepare_prompt(self, task: str) -> str:
        """Prepare prompt for generation"""
        messages = [
            {"role": "system", "content": self.prompt_template.system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"Task: {task}"}
            ]}
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    @torch.no_grad()
    def generate_candidates(
        self,
        image: Image.Image,
        task: str,
        image_path: str,
        num_candidates: int = 4
    ) -> Tuple[List[str], List[Dict]]:
        """
        Generate multiple candidate plans for a single input

        Returns:
            candidates: List of generated plan strings
            generation_info: List of dicts with log_probs etc.
        """
        prompt = self.prepare_prompt(task)

        # Process input
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        candidates = []
        generation_infos = []

        # Generate candidates using sampling
        num_model_candidates = num_candidates

        # Check if we can inject a negative sample
        if image_path in self.negative_samples and len(self.negative_samples[image_path]) > 0:
            num_model_candidates = num_candidates - 1  # Reserve one slot

        for _ in range(num_model_candidates):
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=self.config['training']['max_new_tokens'],
                temperature=self.config['training']['temperature'],
                do_sample=True,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True
            )

            # Decode generated text
            generated_ids = outputs.sequences[0]
            input_len = inputs['input_ids'].shape[1]
            generated_tokens = generated_ids[input_len:]

            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Parse the plan
            plan_str = generated_text.strip()
            if '\n' in plan_str:
                plan_str = plan_str.split('\n')[0]

            candidates.append(plan_str)
            generation_infos.append({
                'generated_ids': generated_tokens,
                'input_len': input_len
            })

        # Inject negative sample if available
        if image_path in self.negative_samples and len(self.negative_samples[image_path]) > 0:
            neg_plan = random.choice(self.negative_samples[image_path])
            neg_plan_str = ", ".join(neg_plan) if isinstance(neg_plan, list) else neg_plan
            candidates.append(neg_plan_str)
            generation_infos.append({
                'generated_ids': None,  # Negative sample, no generation info
                'input_len': None,
                'is_negative': True
            })

        return candidates, generation_infos

    def compute_log_probs(
        self,
        model,
        image: Image.Image,
        task: str,
        plan: str
    ) -> torch.Tensor:
        """
        Compute log probability of a plan given image and task

        Returns:
            log_prob: Scalar tensor
        """
        # Build full sequence
        messages = [
            {"role": "system", "content": self.prompt_template.system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"Task: {task}"}
            ]},
            {"role": "assistant", "content": plan}
        ]

        full_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Get prompt text for length calculation
        prompt_messages = [
            {"role": "system", "content": self.prompt_template.system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"Task: {task}"}
            ]}
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process full sequence (with image)
        inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Process prompt only (with image) to get actual prompt length including vision tokens
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=False
        )
        prompt_len = prompt_inputs['input_ids'].shape[1]

        # Forward pass
        with torch.set_grad_enabled(model.training):
            outputs = model(**inputs)
            logits = outputs.logits

        # Get sequence length
        seq_len = inputs['input_ids'].shape[1]

        # Handle edge case where prompt is longer than or equal to full sequence
        if prompt_len >= seq_len - 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Compute log probs for response tokens only
        # For next-token prediction: logits[t] predicts token[t+1]
        # We want log prob of tokens from prompt_len to end
        # So we use logits from prompt_len-1 to seq_len-2 to predict tokens from prompt_len to seq_len-1
        response_logits = logits[0, prompt_len-1:seq_len-1, :]  # [response_len, vocab_size]
        response_labels = inputs['input_ids'][0, prompt_len:seq_len]  # [response_len]

        # Handle edge case where response is empty
        if response_logits.shape[0] == 0 or response_labels.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Ensure dimensions match
        min_len = min(response_logits.shape[0], response_labels.shape[0])
        response_logits = response_logits[:min_len]
        response_labels = response_labels[:min_len]

        # Compute log probs
        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=response_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Sum log probs
        total_log_prob = token_log_probs.sum()

        return total_log_prob

    def grpo_step(self, batch: Dict) -> Dict[str, float]:
        """
        Perform one GRPO update step

        Args:
            batch: Dictionary with images, tasks, reference_plans, image_paths

        Returns:
            Dictionary with loss and metrics
        """
        images = batch['images']
        tasks = batch['tasks']
        image_paths = batch['image_paths']

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        total_reward = 0.0
        num_candidates = self.config['training']['num_candidates']
        num_valid_samples = 0  # 用于统计有效样本数

        batch_metrics = {
            'rewards': [],
            'advantages': [],
            'kl_divs': []
        }

        for image, task, image_path in zip(images, tasks, image_paths):
            # Generate candidates
            candidates, gen_infos = self.generate_candidates(
                image, task, image_path, num_candidates
            )

            if len(candidates) == 0:
                continue

            # Compute rewards for each candidate
            rewards = []
            for cand in candidates:
                reward_dict = self.reward_model.compute_reward(cand, task)
                rewards.append(reward_dict['r_total'])

            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            batch_metrics['rewards'].extend(rewards.tolist())

            # Compute advantages (group relative)
            if rewards.std() > 1e-8:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                advantages = rewards - rewards.mean()
            batch_metrics['advantages'].extend(advantages.tolist())

            # Compute policy gradients
            for i, (cand, gen_info, adv) in enumerate(zip(candidates, gen_infos, advantages)):
                # Skip negative samples for gradient computation
                if gen_info.get('is_negative', False):
                    continue

                num_valid_samples += 1

                # Compute log prob under policy
                self.policy_model.train()
                policy_log_prob = self.compute_log_probs(
                    self.policy_model, image, task, cand
                )

                # Compute log prob under reference (for KL)
                with torch.no_grad():
                    ref_log_prob = self.compute_log_probs(
                        self.ref_model, image, task, cand
                    )

                # KL divergence penalty
                kl_div = policy_log_prob - ref_log_prob
                batch_metrics['kl_divs'].append(kl_div.item())

                # GRPO loss: -advantage * log_prob + kl_coef * kl_div
                # Note: adv should be detached to avoid gradient flow through advantage computation
                kl_coef = self.config['training']['kl_coef']
                adv_detached = adv.detach()
                loss = -adv_detached * policy_log_prob + kl_coef * kl_div

                total_loss = total_loss + loss
                total_reward += rewards[i].item()

        # Average loss
        if num_valid_samples > 0:
            total_loss = total_loss / num_valid_samples
        else:
            # Return zero tensor with gradient when no valid samples
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return {
            'loss': total_loss,
            'mean_reward': np.mean(batch_metrics['rewards']) if batch_metrics['rewards'] else 0,
            'mean_advantage': np.mean(batch_metrics['advantages']) if batch_metrics['advantages'] else 0,
            'mean_kl': np.mean(batch_metrics['kl_divs']) if batch_metrics['kl_divs'] else 0
        }

    def train(self):
        """Run GRPO training loop"""
        self.load_data()

        # Setup scheduler
        num_training_steps = len(self.train_dataloader) * self.config['training']['num_epochs']
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )

        # Initialize wandb if configured
        if 'wandb' in self.config['training'].get('report_to', []):
            wandb.init(
                project=self.config['training'].get('wandb_project', 'harp-grpo'),
                name=self.config['training'].get('run_name', 'grpo_qwen2.5vl'),
                config=self.config
            )

        # Training loop
        global_step = 0
        best_reward = -float('inf')

        for epoch in range(self.config['training']['num_epochs']):
            logger.info(f"\n=== Epoch {epoch + 1}/{self.config['training']['num_epochs']} ===")

            epoch_loss = 0.0
            epoch_reward = 0.0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                # GRPO step
                metrics = self.grpo_step(batch)

                # Backward
                if isinstance(metrics['loss'], torch.Tensor) and metrics['loss'].requires_grad:
                    self.optimizer.zero_grad()
                    metrics['loss'].backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.config['training'].get('max_grad_norm', 1.0)
                    )

                    self.optimizer.step()
                    self.scheduler.step()

                # Logging
                loss_val = metrics['loss'].item() if isinstance(metrics['loss'], torch.Tensor) else metrics['loss']
                epoch_loss += loss_val
                epoch_reward += metrics['mean_reward']

                progress_bar.set_postfix({
                    'loss': f"{loss_val:.4f}",
                    'reward': f"{metrics['mean_reward']:.4f}",
                    'kl': f"{metrics['mean_kl']:.4f}"
                })

                # Detailed logging
                if global_step % self.config['training']['logging_steps'] == 0:
                    log_dict = {
                        'train/loss': loss_val,
                        'train/reward': metrics['mean_reward'],
                        'train/kl_div': metrics['mean_kl'],
                        'train/lr': self.scheduler.get_last_lr()[0]
                    }
                    if 'wandb' in self.config['training'].get('report_to', []):
                        wandb.log(log_dict, step=global_step)

                # Save checkpoint
                if global_step % self.config['training']['save_steps'] == 0 and global_step > 0:
                    self.save_checkpoint(global_step)

                global_step += 1

            # End of epoch
            avg_loss = epoch_loss / len(self.train_dataloader)
            avg_reward = epoch_reward / len(self.train_dataloader)
            logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save_checkpoint('best')

        # Save final model
        self.save_checkpoint('final')
        logger.info("\nGRPO training complete!")

    def save_checkpoint(self, step):
        """Save model checkpoint"""
        output_dir = Path(self.config['training']['output_dir']) / f"checkpoint-{step}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {output_dir}")

        # Save LoRA weights
        self.policy_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description="GRPO training for Qwen2.5-VL with H-PRM")
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer and train
    trainer = GRPOTrainerForVLM(config)
    trainer.train()


if __name__ == "__main__":
    main()
