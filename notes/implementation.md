## Implementing Tensor-Based (CoE) Reward, Hidden-State Feature Extraction, and Metadata Saving in EasyR1

This document is a self-contained implementation guide to port the tensor-based reward (using hidden states), hidden-state feature extraction, and per-step metadata saving from this repo into the upstream EasyR1 repo. Follow it step-by-step to reproduce the features exactly.

Reference upstream repository: `https://github.com/hiyouga/EasyR1`

### What you get
- Hidden-state extraction from the actor during PPO log-prob recomputation
- A CoE-style reward manager that passes per-sequence pooled hidden states to a custom reward function
- Rollout-group reward normalization (per-prompt) before PPO ingestion
- Optional, detailed per-step metadata saving (prompts, responses, rewards, advantages, and hidden-state `.pt` tensors)

---

## 1) Architecture Overview

- During training, when the trainer recomputes `old_log_probs`, the actor performs a forward pass with `output_hidden_states=True`. The code pools hidden states across response tokens for every transformer layer, yielding a tensor of shape `[batch_size, num_layers, hidden_dim]`. This pooled tensor is cached.
- The trainer then RPC-fetches the pooled hidden states from the worker and attaches them to the current batch.
- The reward manager (`coe_batch`) calls a sequential reward function per sequence with signature `(response_text, ground_truth, hidden_states_for_sequence)` and builds token-level rewards. It also normalizes rewards per prompt across its rollout group before returning to PPO.
- Optionally, the trainer saves per-step metadata (JSON) and hidden-state tensors (.pt) to disk for analysis.

High-level flow:
1. Generate sequences → Recompute `old_log_probs` (actor forward with hidden-states)
2. Fetch pooled hidden states → Attach to batch
3. Compute rewards via CoE reward manager (per-sequence calls) → Normalize across rollout group
4. Compute advantages/returns → PPO updates
5. Optionally save per-step metadata

---

## 2) File-by-File Implementation Guide

Below are minimal, surgical edits to apply to the upstream codebase. If you do not find the exact insertion points, use the function/class names and comments as anchors and adapt to variations.

### 2.1 `verl/trainer/config.py` — Trainer metadata configuration
Add metadata fields to `TrainerConfig` (under existing fields). These control the metadata saving behavior.

```python
@dataclass
class TrainerConfig:
    ...
    # Metadata configuration
    save_metadata: bool = False                 # Enable/disable metadata saving
    metadata_save_freq: int = 5                 # Save metadata for first N steps (<= N)
    metadata_max_sequences: int = -1            # Max sequences to save per prompt (-1 = save all)
    metadata_save_hidden_states: bool = False   # Save hidden-state tensors to .pt files
    metadata_save_full_texts: bool = True       # Save full prompt/response text files
    metadata_truncate_texts: bool = True        # Truncate texts inside metadata.json
    metadata_max_words: int = 100               # Truncation limit for words
    metadata_save_path: str = "generated_metadata"  # Base output directory
```

### 2.2 `verl/trainer/main.py` — Reward manager selection
Allow selecting the new CoE reward manager when `reward_type: coe_batch`.

```python
from ..workers.reward.function import (
    SequentialFunctionRewardManager,
    BatchFunctionRewardManager,
    CoEBatchFunctionRewardManager,   # add import
)

# Training reward
if config.worker.reward.reward_type == "sequential":
    TrainRewardManager = SequentialFunctionRewardManager
elif config.worker.reward.reward_type == "batch":
    TrainRewardManager = BatchFunctionRewardManager
elif config.worker.reward.reward_type == "coe_batch":
    TrainRewardManager = CoEBatchFunctionRewardManager
else:
    raise NotImplementedError(...)

# Validation reward (same branching for config.worker.val_reward.reward_type)
```

### 2.3 `verl/workers/actor/dp_actor.py` — Hidden-state extraction and caching
Add:
- A helper `extract_hidden_states()` to mean-pool response-token hidden states per layer
- Storage for `last_hidden_states` in `DataParallelPPOActor`
- In `compute_log_prob()`, call the model with `output_hidden_states=True`, pool the hidden states, and cache them
- A getter `get_hidden_states()` that returns the cached tensor

Key snippets:

```python
def extract_hidden_states(
    hidden_states: Tuple[torch.Tensor, ...],
    response_mask: torch.Tensor,
    response_length: int,
    responses: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    # For each layer, select response-token slice, mask exact valid tokens, mean-pool
    # Returns: {"pooled": [batch_size, num_layers, hidden_dim]}

class DataParallelPPOActor(BasePPOActor):
    def __init__(...):
        ...
        self.last_hidden_states: Optional[torch.Tensor] = None

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        ...
        output = self.actor_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **multi_modal_inputs,
            use_cache=False,
            output_hidden_states=True,   # crucial for CoE
        )
        logits = output.logits
        ...
        # Pool layer-wise hidden states across response tokens
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            response_mask = attention_mask[:, -response_length:]
            hs_dict = extract_hidden_states(output.hidden_states, response_mask, response_length, responses)
            if "pooled" in hs_dict:
                pooled = hs_dict["pooled"]  # [B, L, D]
                self.last_hidden_states = pooled if self.last_hidden_states is None else torch.cat([
                    self.last_hidden_states, pooled
                ], dim=0)

    def get_hidden_states(self) -> Optional[torch.Tensor]:
        return getattr(self, "last_hidden_states", None)
```

Notes:
- The pooling uses an improved mask computed from both the response mask and actual non-pad token count to produce stable means per layer.

### 2.4 `verl/workers/fsdp_workers.py` — RPC to retrieve hidden states
Add a worker method that retrieves the pooled hidden states from the actor and packs them into a `DataProto`. Also clear the actor cache afterward to reduce memory pressure.

```python
from ..protocol import DataProto

@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def get_hidden_states(self):
    assert self._has_actor
    try:
        hidden_states = self.actor.get_hidden_states()  # [B, L, D] or None
        tensors = {"hidden_states": hidden_states} if hidden_states is not None else {}
        output = DataProto.from_dict(tensors=tensors) if tensors else DataProto()
        if hasattr(self.actor, 'last_hidden_states'):
            del self.actor.last_hidden_states
            self.actor.last_hidden_states = None
        return output
    except Exception:
        torch.cuda.empty_cache()
        return DataProto()
```

### 2.5 `verl/workers/reward/function.py` — CoE reward manager (tensor-based)
Extend the reward framework with a new manager that passes hidden states to a per-sequence reward function and normalizes rollout-group rewards.

```python
from typing import Callable, Optional

class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]

# New: signature for CoE reward functions
CoEBatchRewardFunction = Callable[[str, str, Optional[torch.Tensor]], RewardScore]

class CoEBatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: CoEBatchRewardFunction

    def _extract_sequence_hidden_states(self, data: DataProto, i: int) -> Optional[torch.Tensor]:
        # Expects data.batch["hidden_states"] with shape [B, L, D]
        hs = data.batch.get("hidden_states")
        if hs is None or len(hs.shape) != 3 or i >= hs.shape[0]:
            return None
        return hs[i]  # [L, D]

    def _normalize_rewards_across_rollouts(self, rewards: torch.Tensor, rollout_n: int) -> torch.Tensor:
        # Per-prompt min-max group normalization to [0, 1]
        if len(rewards) == 0 or len(rewards) % rollout_n != 0:
            return rewards
        num_prompts = len(rewards) // rollout_n
        rr = rewards.view(num_prompts, rollout_n)
        mn, mx = rr.min(dim=1, keepdim=True)[0], rr.max(dim=1, keepdim=True)[0]
        rng = torch.where(mx - mn == 0, torch.ones_like(mx), mx - mn)
        return ((rr - mn) / rng).view(-1)

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        # Build per-sequence inputs and call sequential function, passing hidden states
        response_ids = data.batch["responses"]
        response_len = data.batch["response_mask"].sum(dim=-1)
        responses, gts, scores = [], [], []
        for i in range(len(data)):
            valid_ids = response_ids[i][: response_len[i]]
            responses.append(self.tokenizer.decode(valid_ids, skip_special_tokens=self.config.skip_special_tokens))
            gts.append(data.non_tensor_batch["ground_truth"][i])
        for i in range(len(data)):
            seq_hidden = self._extract_sequence_hidden_states(data, i)  # [L, D] or None
            scores.append(self.reward_fn(responses[i], gts[i], seq_hidden))

        raw = torch.tensor([s["overall"] for s in scores], dtype=torch.float32)
        rollout_n = 5  # match your config.worker.rollout.n
        norm = self._normalize_rewards_across_rollouts(raw, rollout_n)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, (sc, nr) in enumerate(zip(scores, norm)):
            rl = int(response_len[i].item())
            reward_tensor[i, rl - 1] = nr.item()
            reward_metrics["old_overall"].append(sc["overall"])  # raw
            for k, v in sc.items():
                if k != "overall":
                    reward_metrics[k].append(v)
            reward_metrics["overall"].append(nr.item())  # normalized
        return reward_tensor, reward_metrics
```

Tip: Export `CoEBatchFunctionRewardManager` in `verl/workers/reward/__init__.py` if the upstream uses package imports.

### 2.6 `verl/trainer/ray_trainer.py` — Fetch hidden states and save metadata
Edits in three places:

1) In `__init__`, set up a run-scoped directory for metadata if enabled:

```python
if getattr(config.trainer, 'save_metadata', False):
    self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    self.model_name = self._get_model_name()
    self.batch_save_dir = f"{self.config.trainer.metadata_save_path}/{self.model_name}/run_{self.run_timestamp}"
    os.makedirs(self.batch_save_dir, exist_ok=True)
```

Add helper methods if missing:

```python
def _get_model_name(self) -> str:
    ...  # Extract last component from worker.actor.model.model_path

def _truncate_text(self, text: str, max_words: int = 100) -> str:
    ...

def _save_step_metadata(self, batch: DataProto, reward_tensor: torch.Tensor, step: int, reward_metrics: Optional[Dict[str, List[float]]] = None) -> None:
    # Writes a nested folder per step/prompt containing metadata.json (+ optional full texts and .pt hidden-states)
    # Includes per-response reward, advantage, and CoE sub-metrics
```

2) In training loop (and validation), right after `compute_log_probs` on actor, fetch and attach hidden states if available:

```python
with timer("old", timing_raw):
    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
    batch = batch.union(old_log_probs)
    try:
        hs_data = self.actor_rollout_ref_wg.get_hidden_states()
        if getattr(hs_data, 'batch', None) and "hidden_states" in hs_data.batch:
            batch.batch["hidden_states"] = hs_data.batch["hidden_states"]
    except Exception:
        pass
```

3) Optionally save metadata for the first N steps (controlled by `metadata_save_freq`):

```python
if (
    getattr(self.config.trainer, 'save_metadata', False)
    and getattr(self.config.trainer, 'metadata_save_freq', 0) > 0
    and self.global_step <= self.config.trainer.metadata_save_freq
    and "token_level_scores" in batch.batch
):
    self._save_step_metadata(batch, batch.batch["token_level_scores"], self.global_step, None)
```

Also remember to `del batch.batch["hidden_states"]` after use to save memory.

### 2.7 `examples/reward_function/training_coe.py` — Example CoE reward function
Provide an example reward function that consumes a single sequence’s hidden states `[num_layers, hidden_dim]` and returns metrics. The overall score can be a combination of first- and second-order changes across layers.

```python
import torch
from typing import Dict

def coe_reward(hidden: torch.Tensor) -> Dict[str, float]:
    if hidden is None or not isinstance(hidden, torch.Tensor) or len(hidden.shape) != 2:
        return {"d1_mean": 0.0, "d2_mean": 0.0}
    if hidden.shape[0] < 3:
        return {"d1_mean": 0.0, "d2_mean": 0.0}

    d1 = hidden[1:] - hidden[:-1]
    d2 = hidden[2:] - 2 * hidden[1:-1] + hidden[:-2]
    rep_start, rep_end = hidden[0], hidden[-1]
    d1 = d1 / (torch.norm(rep_end - rep_start, p=2) + 1e-8)
    d2 = d2 / (torch.norm(d1[-1] - d1[0], p=2) + 1e-8)
    return {
        "d1_mean": torch.norm(d1, dim=1).mean().item(),
        "d2_mean": torch.norm(d2, dim=1).mean().item(),
    }

def compute_score(predict: str, ground_truth: str, hidden: torch.Tensor = None, reward_strategy: str = "d1_mean_plus_d2_mean") -> Dict[str, float]:
    s = coe_reward(hidden)
    overall = (s["d1_mean"] + s["d2_mean"]) / 2
    return {"overall": overall, "d1_mean": s["d1_mean"], "d2_mean": s["d2_mean"]}
```

You can blend format/accuracy components if desired; the manager forwards any extra keys as metrics.

---

## 3) Configuration

Reward configuration (training) — use your actual path:

```yaml
worker:
  reward:
    reward_type: "coe_batch"
    reward_function: "./examples/reward_function/training_coe.py:compute_score"
    reward_function_kwargs:
      reward_strategy: "d1_mean_plus_d2_mean"

  val_reward:
    reward_type: "batch"  # e.g., simple math reward for validation
    reward_function: "./examples/reward_function/validation_math.py:compute_score"
```

Trainer metadata configuration:

```yaml
trainer:
  save_metadata: true
  metadata_save_freq: 1
  metadata_max_sequences: -1
  metadata_save_hidden_states: true
  metadata_save_full_texts: true
  metadata_truncate_texts: true
  metadata_max_words: 100
  metadata_save_path: "generated_metadata"
```

Note: The CoE reward manager uses a hardcoded `rollout_n = 5` for normalization in the example. Align this with `config.worker.rollout.n` or refactor to read from config.

---

## 4) Using the Features

1. Implement the file edits above in upstream EasyR1.
2. Add your CoE reward under `examples/reward_function/training_coe.py` (or any path), ensuring `compute_score(response_text, ground_truth, hidden)` signature.
3. Update your `examples/config.yaml` (or equivalent) to enable `reward_type: coe_batch` and metadata saving.
4. Launch training normally. Hidden states will be extracted during `old_log_probs` recompute and passed to the reward function.
5. If metadata is enabled, per-step files will be saved under `{metadata_save_path}/{model_name}/run_{timestamp}/` with:
   - `step_XXXX/prompt_YYY/metadata.json`
   - optional `prompt_full.txt`, `response_k_full.txt`
   - optional `hidden_states_k.pt` containing `[num_layers, hidden_dim]`

---

## 5) Validation Checklist

- Logs show hidden-state retrieval after `compute_log_probs`
- Reward metrics include both `overall` (normalized) and `old_overall` (raw)
- Metadata directory created and populated with JSON/pt files when enabled
- PPO training proceeds normally (no changes to the PPO core loop beyond reward inputs)

---

## 6) Performance and Pitfalls

- Memory: hidden states can be large. We cache only pooled `[B, L, D]` and clear it after retrieval. Ensure `.pt` saving is disabled if disk or I/O is constrained.
- Normalization: per-prompt min-max across rollout candidates stabilizes PPO scaling. Keep `rollout_n` consistent with your rollout configuration.
- Missing hidden states: the reward manager tolerates absence (returns zero metrics), so training won’t crash if extraction fails.
- KL interactions: no special changes; PPO KL logic is unchanged.

---

## 7) Minimal Diff Summary

- `verl/trainer/config.py`: add metadata fields to `TrainerConfig`
- `verl/trainer/main.py`: add `coe_batch` reward manager selection
- `verl/workers/actor/dp_actor.py`: add pooling helper; request `output_hidden_states=True`; cache `last_hidden_states`; add `get_hidden_states()`
- `verl/workers/fsdp_workers.py`: add registered `get_hidden_states()` returning `DataProto`
- `verl/workers/reward/function.py`: add `CoEBatchFunctionRewardManager`, types, and rollout normalization
- `verl/trainer/ray_trainer.py`: fetch hidden states after `compute_log_probs`; optional metadata saving helpers and calls
- `examples/reward_function/training_coe.py`: add CoE reward example
- Update config to set `reward_type: coe_batch` and metadata keys

---

## 8) References

- Upstream project: `https://github.com/hiyouga/EasyR1`


---

## 9) Prerequisites and Assumptions

- HF Transformers model supports `output_hidden_states=True` and returns `ModelOutput.hidden_states`.
- Actor uses HF models via VERL’s FSDP stack (as in upstream EasyR1) so the same forward signatures apply.
- Your PPO setup relies on recomputing `old_log_probs` on the actor (default in EasyR1/veRL HybridEngine flow).
- Python 3.9+, `torch`, `transformers>=4.51.0`, and other EasyR1 prerequisites (see upstream README).

Tip: For large models, ensure enough GPU memory or enable offloading in your config. Hidden-state pooling is efficient but still requires an extra forward with `output_hidden_states=True`.

---

## 10) Imports Checklist (Do not miss)

Make sure the following imports exist in the specific files:

- `verl/workers/actor/dp_actor.py`
  - `from typing import Any, Dict, List, Optional, Tuple`
  - `import torch`
  - `from ...protocol import DataProto, batch_collate`
  - `from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs`
  - `from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input` (fallback provided in file)

- `verl/workers/fsdp_workers.py`
  - `from ..protocol import DataProto`
  - `from ..single_controller.base.decorator import Dispatch, register`
  - `import torch`

- `verl/workers/reward/function.py`
  - `from typing import Callable, Dict, List, Optional, Tuple, TypedDict`
  - `import torch`
  - `from ...protocol import DataProto`

- `verl/trainer/main.py`
  - `from ..workers.reward.function import SequentialFunctionRewardManager, BatchFunctionRewardManager, CoEBatchFunctionRewardManager`

- `verl/trainer/ray_trainer.py`
  - Already imports `time`, `json`, `torch` etc. Ensure those are present.

- `verl/workers/reward/__init__.py` (if present)
  - Export the new manager: `from .function import CoEBatchFunctionRewardManager` and include in `__all__`.

---

## 11) Exact Insertion Anchors and Full Snippets

Use these anchors to locate insertion points quickly in the upstream code.

### 11.1 Actor: pooling and caching (dp_actor.py)

Anchor: class `DataParallelPPOActor(BasePPOActor)` and method `compute_log_prob`.

Add helper near the top (after imports in this file):

```python
def extract_hidden_states(
    hidden_states: Tuple[torch.Tensor, ...],
    response_mask: torch.Tensor,
    response_length: int,
    responses: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if hidden_states is None or len(hidden_states) == 0:
        return {}

    layer_hidden_states: List[torch.Tensor] = []
    batch_size = hidden_states[0].shape[0]
    num_layers = len(hidden_states)

    # Determine exact valid response lengths per sample
    actual_lengths: List[int] = []
    for batch_idx in range(batch_size):
        mask_for_batch = response_mask[batch_idx]
        valid_positions = torch.nonzero(mask_for_batch, as_tuple=False).flatten()
        mask_actual_len = int(valid_positions[-1].item() + 1) if len(valid_positions) > 0 else 1
        if responses is not None:
            response_for_batch = responses[batch_idx]
            pad_positions = torch.where(response_for_batch == 0)[0]
            response_actual_len = int(pad_positions[0].item()) if len(pad_positions) > 0 else int(response_for_batch.shape[0])
            actual_len = min(mask_actual_len, response_actual_len)
        else:
            actual_len = mask_actual_len
        actual_lengths.append(max(1, min(actual_len, response_length)))

    # Mean-pool response token hidden states for each layer
    for layer_idx in range(num_layers):
        layer_output = hidden_states[layer_idx]              # [B, Seq, D]
        response_hidden = layer_output[:, -response_length:, :]  # [B, R, D]
        improved_mask = torch.zeros_like(response_mask, dtype=torch.float32)
        for b in range(batch_size):
            improved_mask[b, : actual_lengths[b]] = 1.0
        masked_hidden = response_hidden * improved_mask.unsqueeze(-1)
        sum_hidden = masked_hidden.sum(dim=1)
        valid_tokens = improved_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        layer_mean = sum_hidden / valid_tokens
        layer_hidden_states.append(layer_mean)

    stacked_mean = torch.stack(layer_hidden_states, dim=1)  # [B, L, D]
    return {"pooled": stacked_mean}
```

Inside `DataParallelPPOActor.__init__`, add:

```python
self.last_hidden_states: Optional[torch.Tensor] = None
```

Inside `compute_log_prob` (in the micro-batch loop), make sure the forward is:

```python
output = self.actor_module(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    **multi_modal_inputs,
    use_cache=False,
    output_hidden_states=True,
)
```

Right after computing `log_probs`, pool and cache hidden states:

```python
if hasattr(output, "hidden_states") and output.hidden_states is not None:
    response_mask = attention_mask[:, -response_length:]
    hs_dict = extract_hidden_states(output.hidden_states, response_mask, response_length, responses)
    if "pooled" in hs_dict:
        hidden_states_pooled_lst.append(hs_dict["pooled"])  # collect per micro-batch
...
# After the loop
if hidden_states_pooled_lst:
    self.last_hidden_states = torch.concat(hidden_states_pooled_lst, dim=0)
else:
    self.last_hidden_states = None
```

Optionally, expose a getter method:

```python
def get_hidden_states(self) -> Optional[torch.Tensor]:
    return getattr(self, "last_hidden_states", None)
```

### 11.2 Worker: hidden-state RPC (fsdp_workers.py)

Anchor: `class FSDPWorker(Worker)`; add this registered method alongside other DP_COMPUTE_PROTO methods.

Also ensure:

```python
from ..single_controller.base.decorator import Dispatch, register
from ..protocol import DataProto
```

Method:

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def get_hidden_states(self):
    """Get pooled hidden states from actor for reward computation."""
    assert self._has_actor
    try:
        hidden_states = self.actor.get_hidden_states()
        tensors = {"hidden_states": hidden_states} if hidden_states is not None else {}
        output = DataProto.from_dict(tensors=tensors) if tensors else DataProto()
        if hasattr(self.actor, 'last_hidden_states'):
            del self.actor.last_hidden_states
            self.actor.last_hidden_states = None
        return output
    except Exception as e:
        print(f"Error in get_hidden_states: {str(e)}")
        torch.cuda.empty_cache()
        return DataProto()
```

### 11.3 Trainer: attach hidden states + save metadata (ray_trainer.py)

Anchors:
- In `__init__`, after computing `self.training_steps`
- In the training loop after `compute_log_probs`
- In validation path likewise
- Around the rewards/advantages section to call `_save_step_metadata`

Add in `__init__`:

```python
if getattr(config.trainer, 'save_metadata', False):
    self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    self.model_name = self._get_model_name()
    self.batch_save_dir = f"{self.config.trainer.metadata_save_path}/{self.model_name}/run_{self.run_timestamp}"
    os.makedirs(self.batch_save_dir, exist_ok=True)
```

Fetching hidden states (train and val), immediately after computing actor `old_log_probs`:

```python
hidden_states_data = self.actor_rollout_ref_wg.get_hidden_states()
if (
    hidden_states_data is not None
    and hasattr(hidden_states_data, "batch")
    and hidden_states_data.batch is not None
    and "hidden_states" in hidden_states_data.batch
):
    batch.batch["hidden_states"] = hidden_states_data.batch["hidden_states"]
```

Call metadata saver (first N steps):

```python
if (
    getattr(self.config.trainer, 'save_metadata', False)
    and getattr(self.config.trainer, 'metadata_save_freq', 0) > 0
    and self.global_step <= self.config.trainer.metadata_save_freq
    and "token_level_scores" in batch.batch
):
    self._save_step_metadata(batch, batch.batch["token_level_scores"], self.global_step, None)
```

Memory hygiene after use:

```python
if "hidden_states" in batch.batch:
    try:
        del batch.batch["hidden_states"]
    except Exception:
        pass
```

Ensure helper methods exist: `_get_model_name`, `_truncate_text`, and a robust `_save_step_metadata` that writes:

- `step_XXXX/prompt_YYY/metadata.json` with prompt text, ground truth, per-response text, token lengths, rewards (normalized/unnormalized), advantages, CoE sub-metrics, hidden-states shape and file name
- Optional `prompt_full.txt` and `response_k_full.txt`
- Optional `hidden_states_k.pt` containing `[num_layers, hidden_dim]`

### 11.4 Reward manager: CoE with normalization (function.py)

Anchors: existing `FunctionRewardManager`, `SequentialFunctionRewardManager`, `BatchFunctionRewardManager` classes.

Add at top-level types:

```python
class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]

CoEBatchRewardFunction = Callable[[str, str, Optional[torch.Tensor]], RewardScore]
```

Add the new manager:

```python
class CoEBatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: CoEBatchRewardFunction

    def _extract_sequence_hidden_states(self, data: DataProto, sequence_idx: int) -> Optional[torch.Tensor]:
        if "hidden_states" not in data.batch or data.batch["hidden_states"] is None:
            return None
        hidden_states = data.batch["hidden_states"]
        if len(hidden_states.shape) != 3 or sequence_idx >= hidden_states.shape[0]:
            return None
        return hidden_states[sequence_idx]  # [L, D]

    def _normalize_rewards_across_rollouts(self, rewards: torch.Tensor, rollout_n: int = 5) -> torch.Tensor:
        if len(rewards) == 0 or len(rewards) % rollout_n != 0:
            return rewards
        num_prompts = len(rewards) // rollout_n
        rr = rewards.view(num_prompts, rollout_n)
        mn, mx = rr.min(dim=1, keepdim=True)[0], rr.max(dim=1, keepdim=True)[0]
        rng = torch.where(mx - mn == 0, torch.ones_like(mx), mx - mn)
        return ((rr - mn) / rng).view(-1)

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        responses, labels, scores = [], [], []
        for i in range(len(data)):
            valid_ids = response_ids[i][: response_length[i]]
            responses.append(self.tokenizer.decode(valid_ids, skip_special_tokens=self.config.skip_special_tokens))
            labels.append(data.non_tensor_batch["ground_truth"][i])
        for i in range(len(data)):
            hs = self._extract_sequence_hidden_states(data, i)
            scores.append(self.reward_fn(responses[i], labels[i], hs))
        raw = torch.tensor([s["overall"] for s in scores], dtype=torch.float32)
        rollout_n = 5  # align with config.worker.rollout.n
        norm = self._normalize_rewards_across_rollouts(raw, rollout_n)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, (sc, nr) in enumerate(zip(scores, norm)):
            rl = int(response_length[i].item())
            reward_tensor[i, rl - 1] = nr.item()
            reward_metrics["old_overall"].append(sc["overall"])  # raw value
            for k, v in sc.items():
                if k != "overall":
                    reward_metrics[k].append(v)
            reward_metrics["overall"].append(nr.item())  # normalized
        return reward_tensor, reward_metrics
```

### 11.5 Runner: reward selection (trainer/main.py)

Anchor: inside the `Runner.run` method where reward managers are chosen.

```python
from ..workers.reward.function import CoEBatchFunctionRewardManager

if config.worker.reward.reward_type == "sequential":
    TrainRewardManager = SequentialFunctionRewardManager
elif config.worker.reward.reward_type == "batch":
    TrainRewardManager = BatchFunctionRewardManager
elif config.worker.reward.reward_type == "coe_batch":
    TrainRewardManager = CoEBatchFunctionRewardManager
else:
    raise NotImplementedError(...)

# Mirror the same logic for config.worker.val_reward
```

---

## 12) Reward Function Example: Full Copy-Paste (training_coe.py)

```python
import torch
from typing import Dict

def coe_reward(hidden: torch.Tensor) -> Dict[str, float]:
    if hidden is None or not isinstance(hidden, torch.Tensor) or len(hidden.shape) != 2:
        return {"d1_mean": 0.0, "d2_mean": 0.0}
    if hidden.shape[0] < 3:
        return {"d1_mean": 0.0, "d2_mean": 0.0}
    d1 = hidden[1:] - hidden[:-1]
    d2 = hidden[2:] - 2 * hidden[1:-1] + hidden[:-2]
    rep_start, rep_end = hidden[0], hidden[-1]
    d1 = d1 / (torch.norm(rep_end - rep_start, p=2) + 1e-8)
    d2 = d2 / (torch.norm(d1[-1] - d1[0], p=2) + 1e-8)
    return {
        "d1_mean": torch.norm(d1, dim=1).mean().item(),
        "d2_mean": torch.norm(d2, dim=1).mean().item(),
    }

def compute_score(predict: str, ground_truth: str, hidden: torch.Tensor = None, reward_strategy: str = "d1_mean_plus_d2_mean") -> Dict[str, float]:
    s = coe_reward(hidden)
    overall = (s["d1_mean"] + s["d2_mean"]) / 2
    return {"overall": overall, "d1_mean": s["d1_mean"], "d2_mean": s["d2_mean"]}
```

---

## 13) Optional Quality-of-Life Improvements

- Replace hardcoded `rollout_n = 5` with a config value:
  - Option A: add `normalization_rollout_n: int` to `RewardConfig` and read it in the manager
  - Option B: pass `rollout_n` via `reward_function_kwargs` and store on `self.config`
  - Option C: infer from batch size and `config.worker.rollout.n` (requires controller access)

- Store and display more CoE metrics (e.g., separate min/max over layers) in `reward_metrics` and metadata JSON.

---

## 14) Troubleshooting

- Hidden states are None in reward:
  - Ensure the actor forward uses `output_hidden_states=True` in `compute_log_prob`
  - Ensure trainer actually calls `get_hidden_states()` right after recomputing `old_log_probs`
  - Confirm you did not delete `batch.batch["hidden_states"]` before reward computation

- OOM during hidden-state extraction:
  - Reduce batch sizes; enable offload; avoid saving `.pt` hidden states by disabling `metadata_save_hidden_states`
  - Verify that we pool hidden states to `[B, L, D]` only (not all-token tensors)

- All normalized rewards are zeros or identical:
  - Check that per-prompt grouping is correct (batch-size divisible by `rollout_n`)
  - If all raw rewards are equal, min-max normalization becomes zeros; consider adding epsilon or using z-score

- Metadata not written:
  - Ensure `trainer.save_metadata: true` and `trainer.metadata_save_freq > 0`
  - The saver runs only for steps `<= metadata_save_freq`
  - Confirm `token_level_scores` exists in batch (it is set before KL penalty application)

- Reward function path errors:
  - Use `path/to/file.py:function_name` format and verify it exists; upstream `RewardConfig.post_init` splits on `:`

---

## 15) End-to-End Verification

1. Enable `coe_batch` reward for training and a simple `batch` reward for validation.
2. Run a short training job (1–2 steps) with `save_metadata: true`, `metadata_save_freq: 1`.
3. Inspect logs for messages about retrieving hidden states and saving step metadata.
4. Inspect the metadata directory for JSON and optional `.pt` files. Verify JSON fields for rewards (normalized and raw), advantages, and hidden-states shape.
5. Confirm PPO proceeds without errors and that validation runs normally.
