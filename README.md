# FiRe-OPD: Filter, then Reweight for On-Policy Distillation

## Introduction

FiRe-OPD is an on-policy distillation framework that enhances knowledge transfer from large reasoning models (teachers) to smaller models (students). Unlike standard on-policy distillation (OPD) which treats all student-generated tokens equally, FiRe-OPD introduces two key mechanisms:

1. **Trajectory-level Filtering**: Discards low-quality rollouts based on normalized teacher log-probability, retaining only trajectories where the teacher assigns high confidence.
2. **Token-level Entropy-aware Reweighting**: Assigns higher weight to tokens where the teacher is confident and the student is confused, focusing learning on the most informative positions.

This approach significantly improves distillation efficiency and final performance, achieving strong results on both mathematical reasoning and code generation benchmarks.

### Multi-teacher Extension

FiRe-OPD supports routing to different teacher models per sample. Each training example carries an `opd_teacher` field (e.g., "math" or "code") that determines which teacher's log-probabilities are used for the KL computation. This allows simultaneous distillation from specialized teachers (e.g., a math reasoning model and a code generation model).

## Data

Training data (math and code) sourced from [G-OPD](https://github.com/RUCBM/G-OPD).

For multi-teacher training, each sample must include an `opd_teacher` field indicating which teacher generated the reference. The teacher models can be freely replaced.

## Environment Setup

The environment is consistent with [verl](https://github.com/volcengine/verl). Additionally, install:

```bash
conda create -n verl python==3.10
conda activate verl
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
```

## Training

### Single-Teacher FiRe-OPD (Math)

```bash
cd verl/examples/fire_opd
bash run_fire_opd_single_teacher.sh
```

Full parameters:

```bash
python3 -m verl.trainer.main_ppo \
    # === Data ===
    data.train_files=${TRAIN_DATA} \                          # training parquet path
    data.val_files="${VAL_DATA}" \                            # validation parquet paths (list)
    data.train_batch_size=1024 \                             # global batch size
    data.max_prompt_length=2048 \                            # max prompt token length
    data.max_response_length=16384 \                         # max response token length
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=42 \
    data.return_raw_chat=True \                              # required for OPD
    +data.apply_chat_template_kwargs.enable_thinking=False \ # set for thinking mode
    # === Models ===
    actor_rollout_ref.model.path=${STUDENT_MODEL} \          # student model (e.g., Qwen/Qwen3-4B)
    +actor_rollout_ref.ref.model.path=${TEACHER_MODEL} \     # teacher model path
    # === FiRe-OPD specific ===
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \  # enable OPD mode
    actor_rollout_ref.actor.policy_loss.entropy_aware_distill=True \        # enable entropy-aware weighting
    actor_rollout_ref.actor.policy_loss.traj_skip_percentile=20.0 \        # filter bottom 20% trajectories
    actor_rollout_ref.actor.policy_loss.entropy_alpha=1.0 \                # teacher confidence weight
    actor_rollout_ref.actor.policy_loss.entropy_beta=1.0 \                 # student confusion weight
    # === Training ===
    actor_rollout_ref.actor.optim.lr=1e-6 \                  # learning rate
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.use_remove_padding=True \
    # === Rollout (vLLM) ===
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \   # TP size, adjust per GPU count
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \                            # rollouts per prompt
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    # === Validation rollout ===
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \                 # samples per prompt for pass@k
    # === Ref model ===
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    # === Algorithm ===
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=5.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive \
    # === Trainer ===
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='fire-opd' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=8 \                              # GPUs per node
    trainer.nnodes=1 \                                       # number of nodes
    trainer.save_freq=50 \                                   # checkpoint save frequency (steps)
    trainer.default_local_dir=${CHECKPOINT_DIR} \             # checkpoint output directory
    trainer.test_freq=10 \                                   # validation frequency (steps)
    trainer.total_epochs=3 \
    trainer.resume_mode=auto
```

**Key parameters to modify:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `actor_rollout_ref.model.path` | Student model path (HuggingFace or local) | `Qwen/Qwen3-4B` |
| `+actor_rollout_ref.ref.model.path` | Teacher model path | — |
| `entropy_aware_distill` | Enable FiRe-OPD entropy weighting | `True` |
| `traj_skip_percentile` | Bottom % of trajectories to discard | `20.0` |
| `entropy_alpha` | Teacher confidence weight (higher = more weight on confident tokens) | `1.0` |
| `entropy_beta` | Student confusion weight (higher = more weight on confused tokens) | `1.0` |
| `data.train_batch_size` | Global batch size | `1024` |
| `actor_rollout_ref.actor.optim.lr` | Learning rate | `1e-6` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | TP size for vLLM | `4` |
| `trainer.n_gpus_per_node` | GPUs per node | `8` |
| `trainer.nnodes` | Number of nodes | `1` |

---

### Multi-Teacher FiRe-OPD (Math + Code)

```bash
cd verl/examples/fire_opd
bash run_fire_opd_multi_teacher.sh
```

Differences from single-teacher:

```bash
    # === Models (two teachers) ===
    actor_rollout_ref.model.path=${STUDENT_MODEL} \              # student model
    +actor_rollout_ref.ref.model.path=${MATH_TEACHER} \          # math teacher (handles "math" samples)
    +actor_rollout_ref.ref.model.base_model_path=${CODE_TEACHER} \ # code teacher (handles "code" samples)
    # === Multi-teacher flag ===
    actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true \  # enable multi-teacher routing
```

**Additional parameters to modify:**
| Parameter | Description |
|-----------|-------------|
| `+actor_rollout_ref.ref.model.path` | Math teacher model path |
| `+actor_rollout_ref.ref.model.base_model_path` | Code teacher model path |
| `multi_teacher_distill` | Must be `true` for multi-teacher |

**Data requirement:** Each training sample must have `extra_info.opd_teacher` set to `"math"` or `"code"` to route to the corresponding teacher:
```python
data = {
    ...,
    "extra_info": {
        "opd_teacher": "math",  # or "code"
        ...
    }
}
```

---

### Pure OPD Baseline (no filtering or reweighting)

```bash
cd verl/examples/fire_opd
bash run_opd_baseline.sh
```

Differences from FiRe-OPD: the following parameters are **removed** (not set):
- `entropy_aware_distill` (defaults to `False`)
- `traj_skip_percentile`
- `entropy_alpha` / `entropy_beta`
- `multi_teacher_distill`

Only `only_reverse_kl_advantages=True` is kept for standard reverse-KL distillation.

---

## Evaluation

### Math Reasoning Evaluation

Math evaluation data is in the `data/` folder (AIME24, AIME25, HMMT25-Feb, HMMT25-Nov, MATH500, MinervaMath, OlympiadBench, AMC2023). Evaluation code and script are in the `math_eval/` folder.

```bash
cd math_eval/
bash run_eval_math.sh
```

Modify `MODEL_PATH` and `MODEL_NAME` at the top of the script to point to your checkpoint. The script evaluates all 8 benchmarks in parallel across GPUs (2 GPUs per benchmark, 4 benchmarks per batch).

Parameters in `run_eval_math.sh`:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `MODEL_PATH` | Path to the model to evaluate | `Qwen/Qwen3-4B` |
| `MODEL_NAME` | Name for output files | `Qwen3-4B` |
| `--n` | Number of samples per problem (for pass@k / maj@k) | `32` |
| `--temperature` | Sampling temperature | `1.0` |
| `--top_p` | Top-p sampling | `1.0` |
| `--max_tokens` | Max generation tokens | `16384` |

### Code Generation Evaluation

Code evaluation scripts are in the `code_eval/scripts/` folder. Our evaluation is mainly based on [Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner).

#### EvalPlus (HumanEval+ / MBPP+)

```bash
bash code_eval/scripts/run_evalplus.sh <DATASET> <MODEL_PATH> [GREEDY] [TEMP] [TOP_P] [N_SAMPLES]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `DATASET` | `humaneval` or `mbpp` | `humaneval` |
| `MODEL_PATH` | Path to the model | `Qwen/Qwen3-4B` |
| `GREEDY` | 1 = greedy decoding, 0 = sampling | `1` |
| `TEMP` | Sampling temperature | `0.8` |
| `TOP_P` | Top-p | `0.9` |
| `N_SAMPLES` | Number of samples | `1` |

#### LiveCodeBench

Download data first:
```bash
git clone https://hf-mirror.com/datasets/livecodebench/code_generation_lite code_eval/coding/LiveCodeBench/code_generation_lite
```

```bash
bash code_eval/scripts/run_lcb_gen.sh --local_model_path <MODEL_PATH> [OPTIONS]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `-l, --local_model_path` | Path to the local model | — |
| `-m, --model` | Model name (for output naming) | `Qwen/Qwen3-4` |
| `-g, --gpu` | CUDA GPU IDs | `7` |
| `-n, --n` | Number of samples | `4` |
| `-t, --temperature` | Sampling temperature | `1.0` |
| `-p, --top_p` | Top-p sampling | `1.0` |
| `-k, --max_tokens` | Max generation tokens | `16384` |
| `-b, --batch_size` | Batch size | `128` |

Example:
```bash
CUDA_VISIBLE_DEVICES=0 bash code_eval/scripts/run_lcb_gen.sh --model Qwen3-4B --local_model_path /path/to/checkpoint
```

## Acknowledgments

Our models, data, and pure OPD implementation are based on [G-OPD](https://github.com/RUCBM/G-OPD). Our training code is mainly based on [verl](https://github.com/volcengine/verl). Our evaluation code is mainly based on [Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner), which is built upon [EvalPlus](https://github.com/evalplus/evalplus) and [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).
