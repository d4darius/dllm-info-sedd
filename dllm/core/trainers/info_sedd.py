"""
Info-SEDD Mutual Information Trainer Support
Extends dLLM's MDLM trainer to incorporate specific masking variants
(joint and conditional) to measure mutual information correctly.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.core.trainers.mdlm import MDLMConfig, MDLMTrainer


@dataclass
class InfoSEDDConfig(MDLMConfig):
    is_parametric_marginal: bool = True
    variant: str = "j"  # "j" for joint, "c" for conditional
    var_indices: int | None = None  # fallback for prompt-response boundary


class InfoSEDDTrainer(MDLMTrainer):
    """
    Trainer for evaluating Mutual Information through conditionally
    masked target tokens matching the INFO-SEDD formulation.
    """

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        maskable_mask = labels != -100  # [b, l]

        # === 1. Sample diffusion timesteps ===
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)  # [b, l]

        # === 2. Apply stochastic masking & Info-SEDD Marginal Constraints ===
        masked_mask = (torch.rand((b, l), device=input_ids.device) < p_mask)

        # Identify X (prompt) and Y (summary/response) boundaries
        if "prompt_len" in inputs:
            prompt_lens = inputs["prompt_len"]
            if isinstance(prompt_lens, torch.Tensor):
                if prompt_lens.dim() == 0:
                    prompt_lens = prompt_lens.unsqueeze(0).expand(b)
            else:
                prompt_lens = torch.tensor([prompt_lens] * b, device=input_ids.device)
            prompt_lens = prompt_lens.unsqueeze(1)
        elif getattr(self.args, "var_indices", None) is not None:
            boundary = self.args.var_indices
            if isinstance(boundary, (list, tuple)):
                boundary = boundary[0]
            prompt_lens = torch.full((b, 1), boundary, device=input_ids.device)
        else:
            # Fallback: Infer prompt boundary dynamically from standard SFT label masking (-100)
            # Find the last -100 index in each sequence (which indicates the end of the prompt)
            # If a sequence has no -100, we fallback to sequence length
            prompt_lens = torch.zeros(b, dtype=torch.long, device=input_ids.device)
            for i in range(b):
                masked_indices = (labels[i] == -100).nonzero(as_tuple=True)[0]
                if len(masked_indices) > 0:
                    # The prompt ends exactly 1 token after the last -100 index
                    prompt_lens[i] = masked_indices[-1].item() + 1
                else:
                    prompt_lens[i] = l # Fallback to entire sequence if no prompt is masked
            prompt_lens = prompt_lens.unsqueeze(1)

        seq_idx = torch.arange(l, device=input_ids.device).unsqueeze(0)  # [1, l]
        x_mask = seq_idx < prompt_lens  # tokens in X [b, l]
        y_mask = seq_idx >= prompt_lens  # tokens in Y [b, l]

        force_mask_override = torch.zeros((b, l), dtype=torch.bool, device=input_ids.device)
        force_unmask_override = torch.zeros((b, l), dtype=torch.bool, device=input_ids.device)
        active_loss_mask = torch.ones((b, l), dtype=torch.bool, device=input_ids.device)

        if self.args.is_parametric_marginal:
            if self.args.variant == "j":
                marginal_flag = torch.randint(0, 3, (b, 1), device=input_ids.device)
                
                # flag == 0 -> mask X completely, evaluate strictly on Y
                force_mask_override = torch.where(marginal_flag == 0, x_mask, force_mask_override)
                active_loss_mask = torch.where(marginal_flag == 0, y_mask, active_loss_mask)

                # flag == 1 -> mask X and Y completely, evaluate strictly on X
                force_mask_override = torch.where(marginal_flag == 1, x_mask | y_mask, force_mask_override)
                active_loss_mask = torch.where(marginal_flag == 1, x_mask, active_loss_mask)
                
                # flag == 2 -> evaluate strictly on everything (default behaviour)
            
            elif self.args.variant == "c":
                marginal_flag = torch.randint(0, 2, (b, 1), device=input_ids.device)
                
                # flag == 0 -> unmask X completely (condition on it), evaluate strictly on Y
                force_unmask_override = torch.where(marginal_flag == 0, x_mask, force_unmask_override)
                
                # flag == 1 -> mask X completely, evaluate strictly on Y
                force_mask_override = torch.where(marginal_flag == 1, x_mask, force_mask_override)
                
                # Active loss strictly bounded to Y for all conditional cases
                active_loss_mask = y_mask

        # Apply the overrides to masked_mask
        masked_mask = masked_mask | force_mask_override
        masked_mask = masked_mask & (~force_unmask_override)
        masked_mask = masked_mask & maskable_mask

        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass through the model ===
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        # === 4. Compute per-token loss weights ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        # Info-SEDD applies active loss explicitly on certain variable ranges
        effective_loss_mask = masked_mask & active_loss_mask

        # === 5. Compute weighted cross-entropy ===
        # GPU Memory Optimization: F.cross_entropy over [b, l, V] is extremely memory intensive.
        # We only need the loss for tokens where effective_loss_mask is True.
        vocab_size = logits.size(-1)
        active_indices = effective_loss_mask.view(-1).nonzero(as_tuple=True)[0]
        
        token_nll_flat = torch.zeros((b * l,), dtype=logits.dtype, device=input_ids.device)
        
        if len(active_indices) > 0:
            active_logits = logits.view(-1, vocab_size)[active_indices]
            active_labels = input_ids.view(-1)[active_indices]
            
            # calculate loss ONLY on the needed tokens
            active_nll = F.cross_entropy(active_logits, active_labels, reduction="none")
            token_nll_flat[active_indices] = active_nll

        token_nll = token_nll_flat.view(b, l)
        
        # Free enormous unneeded VRAM tensors instantly
        del logits
        if not return_outputs:
            del outputs

        token_nll = token_nll * loss_weights * effective_loss_mask.to(token_nll.dtype)

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=effective_loss_mask.to(dtype=logits.dtype).detach(),
        )

        # === 6. Normalize loss ===
        if self.loss_norm_type == "token":
            token_nll /= effective_loss_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll /= effective_loss_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            token_nll /= b
        else:
            raise ValueError("Invalid loss_norm_type.")
        loss = token_nll.sum()

        # === 7. Return final loss (and optionally model outputs) ===
        return (loss, outputs) if return_outputs else loss
