# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            # reset input_ids, attention_mask, position_ids to ref model inputs if ref model input_ids is different from actor input_ids
            if "ref_input_ids" in micro_batch.keys():
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys() # handle when ref input_ids is different from actor input_ids
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    def _compute_entropy_aware_loss(
        self,
        old_log_prob,
        log_prob,
        ref_log_prob,
        response_mask,
        student_entropys,
        ref_entropys,
        loss_agg_mode,
        base_ref_log_prob=None,
        base_ref_entropys=None,
        opd_teacher=None,
    ):
        """FiRe-OPD: Entropy-aware distillation with trajectory filtering and token-level adaptive weighting.

        Supports single-teacher and multi-teacher distillation.

        Args:
            old_log_prob: (bsz, response_length) - student log probs (detached)
            log_prob: (bsz, response_length) - student log probs (with grad)
            ref_log_prob: (bsz, response_length) - teacher log probs (math teacher in multi-teacher mode)
            response_mask: (bsz, response_length) - mask for valid tokens
            student_entropys: (bsz, response_length) - student per-token entropy
            ref_entropys: (bsz, response_length) - teacher per-token entropy (math teacher)
            loss_agg_mode: str - loss aggregation mode
            base_ref_log_prob: (bsz, response_length) - code teacher log probs in multi-teacher mode (optional)
            base_ref_entropys: (bsz, response_length) - code teacher per-token entropy (optional)
            opd_teacher: list/tuple of str - per-sample teacher type ("math"/"code") for multi-teacher (optional)

        Returns:
            loss: scalar loss
            metrics: dict of metrics
        """
        plc = self.config.policy_loss
        bsz, seq_len = old_log_prob.shape
        multi_teacher = getattr(plc, 'multi_teacher_distill', False)

        # ============================================================
        # Step 1: Trajectory-level filtering
        # Skip bottom N% trajectories by normalized teacher log prob.
        # In multi-teacher mode, use the correct teacher's log prob per sample.
        # ============================================================

        seq_lengths = response_mask.sum(dim=-1).clamp(min=1)  # (bsz,)

        if multi_teacher and opd_teacher is not None and base_ref_log_prob is not None:
            normalized_teacher_logprob = torch.zeros(bsz, device=ref_log_prob.device)
            for i in range(bsz):
                teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                if teacher_type == "code":
                    normalized_teacher_logprob[i] = (base_ref_log_prob[i] * response_mask[i]).sum() / seq_lengths[i]
                else:
                    normalized_teacher_logprob[i] = (ref_log_prob[i] * response_mask[i]).sum() / seq_lengths[i]
        else:
            normalized_teacher_logprob = (ref_log_prob * response_mask).sum(dim=-1) / seq_lengths

        logprob_threshold = torch.quantile(
            normalized_teacher_logprob.float(), plc.traj_skip_percentile / 100.0
        )
        traj_keep_mask = normalized_teacher_logprob >= logprob_threshold  # (bsz,)

        # ============================================================
        # Step 2: Compute entropy-based continuous token weights
        # weight = (1 + α·teacher_confidence) × (1 + β·student_confusion)
        # In multi-teacher mode, use the correct teacher's entropy per sample.
        # ============================================================

        # For multi-teacher: combine math and code teacher entropy per sample
        if multi_teacher and opd_teacher is not None and base_ref_entropys is not None:
            combined_teacher_entropys = ref_entropys.clone()
            for i in range(bsz):
                teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                if teacher_type == "code":
                    combined_teacher_entropys[i] = base_ref_entropys[i]
            teacher_entropys = combined_teacher_entropys
        else:
            teacher_entropys = ref_entropys

        # Teacher confidence: normalize entropy to [0,1], invert
        valid_teacher_entropys = teacher_entropys[response_mask.bool()]
        if valid_teacher_entropys.numel() > 0:
            teacher_entropy_max = valid_teacher_entropys.max().clamp(min=1e-6)
        else:
            teacher_entropy_max = torch.tensor(1.0, device=teacher_entropys.device)
        teacher_confidence = (1.0 - teacher_entropys / teacher_entropy_max).clamp(min=0.0, max=1.0)

        # Student confusion: normalize entropy to [0,1]
        valid_student_entropys = student_entropys[response_mask.bool()]
        if valid_student_entropys.numel() > 0:
            student_entropy_max = valid_student_entropys.max().clamp(min=1e-6)
        else:
            student_entropy_max = torch.tensor(1.0, device=student_entropys.device)
        student_confusion = (student_entropys / student_entropy_max).clamp(min=0.0, max=1.0)

        # Continuous weight (detached — no gradient through weights)
        alpha = getattr(plc, 'entropy_alpha', 1.0)
        beta = getattr(plc, 'entropy_beta', 1.0)
        token_weight = ((1.0 + alpha * teacher_confidence) * (1.0 + beta * student_confusion)).detach()

        # Normalize token_weight to mean=1.0 over valid tokens
        valid_weight_sum = (token_weight * response_mask).sum()
        valid_token_count = response_mask.sum().clamp(min=1)
        valid_weight_mean = valid_weight_sum / valid_token_count
        token_weight = token_weight / valid_weight_mean.clamp(min=1e-6)

        # ============================================================
        # Step 3: Weighted advantages + standard PPO policy gradient
        # ============================================================

        # Compute reverse KL advantages: multi-teacher routes to correct teacher per sample
        if multi_teacher and base_ref_log_prob is not None and opd_teacher is not None:
            reverse_kl = torch.zeros_like(old_log_prob)
            for i in range(bsz):
                teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                if teacher_type == "code":
                    reverse_kl[i] = old_log_prob[i] - base_ref_log_prob[i]
                else:
                    reverse_kl[i] = old_log_prob[i] - ref_log_prob[i]
            advantages = -reverse_kl
        else:
            advantages = -(old_log_prob - ref_log_prob)

        # Apply token-level entropy weight to advantages
        weighted_advantages = (token_weight * advantages).detach()

        # Apply trajectory-level filtering
        traj_keep_expanded = traj_keep_mask.unsqueeze(1).expand_as(weighted_advantages).float()
        weighted_advantages = weighted_advantages * traj_keep_expanded

        # PPO ratio
        negative_approx_kl = log_prob - old_log_prob
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl)

        # Clipped PPO loss
        clip_ratio = getattr(self.config, 'clip_ratio', 0.2)
        pg_losses1 = -weighted_advantages * ratio
        pg_losses2 = -weighted_advantages * torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        pg_loss_mat = torch.maximum(pg_losses1, pg_losses2)

        # Aggregate with mask
        effective_mask = response_mask * traj_keep_expanded
        loss = agg_loss(loss_mat=pg_loss_mat, loss_mask=effective_mask, loss_agg_mode=loss_agg_mode)

        # ============================================================
        # Metrics
        # ============================================================
        metrics = {}
        with torch.no_grad():
            metrics["fire_opd/traj_keep_ratio"] = traj_keep_mask.float().mean().item()
            metrics["fire_opd/logprob_threshold"] = logprob_threshold.item()
            metrics["fire_opd/normalized_teacher_logprob_mean"] = normalized_teacher_logprob.mean().item()

            valid_mask = effective_mask.bool()
            valid_weights = token_weight[valid_mask]
            if valid_weights.numel() > 0:
                metrics["fire_opd/token_weight_mean"] = valid_weights.mean().item()
                metrics["fire_opd/token_weight_max"] = valid_weights.max().item()
                metrics["fire_opd/token_weight_min"] = valid_weights.min().item()

            valid_tc = teacher_confidence[valid_mask]
            valid_sc = student_confusion[valid_mask]
            if valid_tc.numel() > 0:
                metrics["fire_opd/teacher_confidence_mean"] = valid_tc.mean().item()
                metrics["fire_opd/student_confusion_mean"] = valid_sc.mean().item()

            if valid_teacher_entropys.numel() > 0:
                metrics["fire_opd/teacher_entropy_mean"] = valid_teacher_entropys.mean().item()
            if valid_student_entropys.numel() > 0:
                metrics["fire_opd/student_entropy_mean"] = valid_student_entropys.mean().item()

            ppo_kl = verl_F.masked_mean(-negative_approx_kl, effective_mask)
            pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), effective_mask)
            metrics["fire_opd/ppo_kl"] = ppo_kl.item()
            metrics["fire_opd/pg_clipfrac"] = pg_clipfrac.item()
            metrics["fire_opd/loss"] = loss.detach().item()

        return loss, metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
         # Include code teacher log probs for multi-teacher distillation
        if "base_ref_log_prob" in data.batch.keys():
            select_keys.append("base_ref_log_prob")
        # Include ref_log_prob for only_reverse_kl_advantages mode
        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in data.batch.keys():
            if "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")

        # Include entropy tensors for entropy-aware distillation
        entropy_aware = getattr(self.config.policy_loss, "entropy_aware_distill", False)
        if entropy_aware:
            if "student_entropys" in data.batch.keys():
                select_keys.append("student_entropys")
            if "ref_entropys" in data.batch.keys():
                select_keys.append("ref_entropys")
            if "base_ref_entropys" in data.batch.keys():
                select_keys.append("base_ref_entropys")
            if "ref_log_prob" in data.batch.keys() and "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # Include opd_teacher for multi-teacher distillation
        if "opd_teacher" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("opd_teacher")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    # For entropy-aware distillation, always compute current student entropy
                    calculate_entropy = entropy_coeff != 0 or entropy_aware
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # ============================================================
                    # Entropy-aware distillation path
                    # ============================================================
                    if entropy_aware and "ref_entropys" in model_inputs and "student_entropys" in model_inputs:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        ref_entropys = model_inputs["ref_entropys"]
                        # Use pre-computed student entropy from compute_log_prob (more accurate, same forward pass)
                        student_entropys = model_inputs["student_entropys"]

                        entropy_loss, entropy_metrics = self._compute_entropy_aware_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            ref_log_prob=ref_log_prob,
                            response_mask=response_mask,
                            student_entropys=student_entropys,
                            ref_entropys=ref_entropys,
                            loss_agg_mode=loss_agg_mode,
                            base_ref_log_prob=model_inputs.get("base_ref_log_prob", None),
                            base_ref_entropys=model_inputs.get("base_ref_entropys", None),
                            opd_teacher=model_inputs.get("opd_teacher", None),
                        )
                        pg_loss = entropy_loss
                        micro_batch_metrics.update(entropy_metrics)

                    else:
                        # Vanilla OPD fallback (no entropy-aware weighting)
                        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in model_inputs:
                            advantages = -(old_log_prob - model_inputs["ref_log_prob"])

                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
