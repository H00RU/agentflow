"""
Trainable Qwen Policy for GRPO (Group Relative Policy Optimization)
可训练的 Qwen 策略，用于 GRPO（组相对策略优化）

This module implements a pure GRPO policy:
1. Log probability computation for policy gradient
2. No value head (GRPO uses outcome-based advantages)
3. Gradient computation and backpropagation
4. Weight updates via optimizer

此模块实现纯 GRPO 策略：
1. 策略梯度的对数概率计算
2. 无价值头（GRPO 使用基于结果的优势）
3. 梯度计算和反向传播
4. 通过优化器更新权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np


class TrainableQwenPolicy(nn.Module):
    """
    Trainable wrapper for Qwen model for GRPO (no value head)
    GRPO 的可训练 Qwen 模型包装器（无价值头）

    Architecture:
    - Base: Qwen2.5-7B-Instruct (frozen or LoRA)
    - Policy: Generates actions and computes log probabilities
    - No value head (GRPO uses outcome-based advantages)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
        freeze_base: bool = False,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32
    ):
        """
        Initialize trainable Qwen policy for GRPO (no value head needed)

        Args:
            model_path: Path to Qwen model
            device: Device for training
            torch_dtype: Data type for model weights
            freeze_base: Whether to freeze base model (only train policy)
            use_lora: Use LoRA for efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
        """
        super().__init__()

        self.device = device
        self.model_path = model_path
        self.torch_dtype = torch_dtype

        print(f"[TrainableQwenPolicy] Loading Qwen model from {model_path}...")
        print(f"[TrainableQwenPolicy] Device: {device}, dtype: {torch_dtype}")
        print(f"[TrainableQwenPolicy] LoRA: {use_lora}, Freeze base: {freeze_base}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        # Don't use device_map="auto" for trainable models - it causes meta device issues
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        # Move to device after loading
        if device == "cuda":
            self.base_model = self.base_model.to(device)

        # Apply LoRA if requested
        if use_lora:
            print("[TrainableQwenPolicy] Applying LoRA...")
            self._apply_lora(lora_r, lora_alpha)

        # Freeze base model if requested
        if freeze_base:
            print("[TrainableQwenPolicy] Freezing base model...")
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Get hidden size from model config
        self.hidden_size = self.base_model.config.hidden_size

        # GRPO: No value head needed (outcome-based advantages only)

        print(f"✓ [TrainableQwenPolicy] Model loaded successfully")
        print(f"✓ [TrainableQwenPolicy] Hidden size: {self.hidden_size}")
        print(f"✓ [TrainableQwenPolicy] Using GRPO - no value head")

        # Training mode
        self.train()

    def _apply_lora(self, r: int, alpha: int):
        """
        Apply LoRA to model for efficient fine-tuning
        为模型应用 LoRA 以实现高效微调
        """
        try:
            from peft import get_peft_model, LoraConfig, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=r,
                lora_alpha=alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
                bias="none"
            )

            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()

        except ImportError:
            print("[TrainableQwenPolicy] Warning: peft not installed, skipping LoRA")
            print("[TrainableQwenPolicy] Install with: pip install peft")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute logits and log probs (GRPO - no values)
        前向传播计算 logits 和 log probs（GRPO - 无需 values）

        Args:
            input_ids: Input token IDs (bs, seq_len)
            attention_mask: Attention mask (bs, seq_len)
            response_mask: Mask for response tokens (bs, seq_len)

        Returns:
            Dict with:
                - logits: Token logits (bs, seq_len, vocab_size)
                - log_probs: Log probabilities of tokens (bs, seq_len)
                - hidden_states: Hidden states (bs, seq_len, hidden_size)
        """
        # Get model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        logits = outputs.logits  # (bs, seq_len, vocab_size)
        hidden_states = outputs.hidden_states[-1]  # Last layer (bs, seq_len, hidden_size)

        # Compute log probabilities for actual tokens
        # Shift logits and input_ids for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (bs, seq_len-1)

        # Pad to match sequence length
        token_log_probs = F.pad(token_log_probs, (0, 1), value=0.0)  # (bs, seq_len)

        # Apply response mask if provided
        if response_mask is not None:
            # Fix: The last position is padded and should be masked out
            # Clone to avoid modifying the original mask
            response_mask = response_mask.clone()
            response_mask[:, -1] = 0.0  # Mask out the padded position

            token_log_probs = token_log_probs * response_mask

        return {
            'logits': logits,
            'log_probs': token_log_probs,
            'hidden_states': hidden_states
        }

    def get_action_and_value(
        self,
        obs: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate action and compute log_prob for GRPO training (no values)
        生成动作并计算 log_prob 用于 GRPO 训练（无需 values）

        Args:
            obs: Observation text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple: (action_text, log_probs, response_mask)
        """
        # Tokenize observation
        inputs = self.tokenizer(
            obs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Generate action
        with torch.no_grad():
            generated_ids = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=False
            )

            # Decode generated text
            generated_text = self.tokenizer.decode(
                generated_ids.sequences[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

        # Compute log probs and values for the full sequence
        full_ids = generated_ids.sequences
        full_attention_mask = torch.ones_like(full_ids)

        # Create response mask (only for generated tokens)
        response_mask = torch.zeros_like(full_ids, dtype=torch.float32)
        response_mask[:, input_ids.shape[1]:] = 1.0

        # Fix: Exclude padding tokens from response mask
        if self.tokenizer.pad_token_id is not None:
            pad_mask = (full_ids != self.tokenizer.pad_token_id).float()
            response_mask = response_mask * pad_mask

        # Fix: Exclude tokens after EOS (keep first EOS, mask everything after)
        if self.tokenizer.eos_token_id is not None:
            # Find EOS positions: 1 where there's an EOS, 0 elsewhere
            eos_positions = (full_ids == self.tokenizer.eos_token_id).float()
            # Cumulative sum: positions after first EOS will have value > 1
            eos_cumsum = eos_positions.cumsum(dim=1)
            # Keep only positions where cumsum <= 1 (before or at first EOS)
            eos_mask = (eos_cumsum <= 1).float()
            response_mask = response_mask * eos_mask

        # Forward pass to get log_probs (GRPO: no values needed)
        outputs = self.forward(
            input_ids=full_ids,
            attention_mask=full_attention_mask,
            response_mask=response_mask
        )

        log_probs = outputs['log_probs']

        return generated_text, log_probs, response_mask

    def save_checkpoint(self, path: str):
        """Save model checkpoint (GRPO: no value head)"""
        checkpoint = {
            'base_model': self.base_model.state_dict(),
            'config': {
                'model_path': self.model_path,
                'hidden_size': self.hidden_size
            }
        }
        torch.save(checkpoint, path)
        print(f"[TrainableQwenPolicy] Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint (GRPO: no value head)"""
        checkpoint = torch.load(path, map_location=self.device)
        self.base_model.load_state_dict(checkpoint['base_model'])
        print(f"[TrainableQwenPolicy] Checkpoint loaded from {path}")


if __name__ == "__main__":
    # Test the trainable policy
    print("Testing TrainableQwenPolicy...")

    policy = TrainableQwenPolicy(
        model_path="/root/models/Qwen2.5-7B-Instruct",
        device="cuda",
        use_lora=True,
        freeze_base=False
    )

    # Test action generation
    obs = "Dataset: HumanEval\nCurrent Score: 0.65\nImprove the workflow."
    action, log_probs, mask = policy.get_action_and_value(obs)

    print(f"\nAction: {action}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Mask shape: {mask.shape}")

    print("\n✅ TrainableQwenPolicy test passed!")
