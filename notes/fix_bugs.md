### Fix plan for CoE implementation integration

This document tracks issues discovered in the current CoE implementation and prescribes precise fixes. Each item includes context, root cause, changes required, and a validation plan.

### 1) Rollout-group normalization breaks after batch reordering

- Problem: The trainer reorders batches to balance token lengths, so groups of size `rollout_n` are not contiguous. The current normalization reshapes flat rewards via `.view(num_prompts, rollout_n)`, yielding incorrect min-max groups.
- Affected files:
  - `verl/trainer/ray_trainer.py`: `_balance_batch` changes order
  - `verl/workers/reward/function.py`: `CoEBatchFunctionRewardManager._normalize_rewards_across_rollouts`
- Fix:
  - Group rewards by `uid` (present in `batch.non_tensor_batch['uid']`) instead of relying on contiguity.
  - Build an index map from `uid -> list[indices]`, compute min/max per group, and scatter normalized values back.
- Steps:
  1. In `compute_reward`, pull `uids = data.non_tensor_batch['uid']`.
  2. Build `uid_to_indices`: dict[str, list[int]].
  3. For each group, compute min/max from `raw[group_indices]`; normalize within group.
  4. Remove the `.view(num_prompts, rollout_n)` logic.
- Tests:
  - Unit test with shuffled batches and repeated `uid`s; verify identical normalized values across the same `uid` group.

### 2) Hidden-state order misaligned when dynamic batching is enabled

- Problem: When `dynamic_batching=True`, `log_probs` are restored to original order via `restore_dynamic_batch`, but pooled hidden states are not, causing misalignment.
- Affected files: `verl/workers/actor/dp_actor.py` (in `compute_log_prob`)
- Fix:
  - Collect pooled hidden states per micro-batch similarly to logprobs.
  - If dynamic batching was used, call `restore_dynamic_batch(hidden_states, batch_idx_list)` on the stacked [B, L, D] tensor.
  - Then set `self.last_hidden_states` to the restored tensor.
- Steps:
  1. Track `hidden_states_pooled_lst` per micro-batch (already present).
  2. After concatenation, if `self.config.dynamic_batching` is True, apply `restore_dynamic_batch` to the pooled tensor.
- Tests:
  - Construct a synthetic micro-batch sharding with different reorderings; assert hidden states align with `input_ids` indices after restore.

### 3) Auxiliary forward for hidden states misses 3D position_ids handling (mrope)

- Problem: The second forward used to get hidden states does not transpose 3D `position_ids` (Qwen2-VL), unlike the main path.
- Affected files: `verl/workers/actor/dp_actor.py` (auxiliary forward inside `compute_log_prob`)
- Fix:
  - Before the auxiliary forward, if `position_ids.dim() == 3`, apply `position_ids = position_ids.transpose(0, 1)`.
- Steps:
  1. Insert the transpose logic mirroring `_forward_micro_batch`.
- Tests:
  - Run a minimal Qwen2-VL config; ensure shapes match and no runtime error.

### 4) get_hidden_states RPC should return CPU tensors

- Problem: `get_hidden_states` returns GPU tensors; other RPCs `.to('cpu')` before serialization.
- Affected files: `verl/workers/fsdp_workers.py`
- Fix:
  - Add `output = output.to('cpu')` before return.
- Tests:
  - Smoke test under Ray to ensure driver receives CPU tensors; no CUDA device mismatch.

### 5) Padding-free path yields no hidden states

- Problem: In padding-free mode, hidden-state extraction is skipped, so CoE rewards fall back to zeros.
- Options:
  - A) Implement pooling for the padding-free (varlen) path: request `output_hidden_states=True` in that forward, then map the varlen outputs to response-token segments before pooling.
  - B) Document and guard: when `coe_batch` is active, force `padding_free=False` and warn.
- Recommendation: Start with B (safer) and optionally implement A later.
- Steps (B):
  1. In trainer startup, if `reward_type=='coe_batch'` and `worker.actor.padding_free=True`, print a warning and set it to False.
- Tests:
  - Verify hidden states are present when CoE reward is enabled.

### 6) Token-id heuristic for response actual length assumes pad_token_id==0

- Problem: The pooling helper tries to infer length by checking for token value `0`. This can be wrong if pad id != 0.
- Affected files: `verl/workers/actor/dp_actor.py` (extract_hidden_states)
- Fix:
  - Rely solely on `response_mask` (already available) to determine valid positions within the response window; remove the token-id heuristic.
- Steps:
  1. Delete the branch computing `resp_actual_len` from token ids.
  2. Use `response_mask` to compute `actual_lengths` robustly.
- Tests:
  - Run with models where pad token id is not 0; confirm stable pooling.

### 7) Performance: avoid a second forward by capturing hidden states in the existing forward

- Problem: We currently do a second full forward to obtain `hidden_states`, doubling compute and memory.
- Fix:
  - Non-padding-free path: set `output_hidden_states=True` in the main forward, reuse logits to compute log_probs, and pool hidden states from the same output object.
  - Padding-free path: defer per item 5 (B) or implement varlen pooling if needed.
- Steps:
  1. In `_forward_micro_batch`, non-padding-free branch: request `output_hidden_states=True` and pool immediately.
  2. Plumb pooled tensor back to `compute_log_prob` via a return side-channel or instance buffer.
- Tests:
  - Compare log_probs numerically vs baseline; ensure identical results and lower step time.

### 8) Metadata grouping uses index math; should group by uid

- Problem: Metadata saver uses `i // rollout_n` to determine prompt folder, but batch is globally reordered; responses from different prompts can mix.
- Affected files: `verl/trainer/ray_trainer.py` (`_save_step_metadata`)
- Fix:
  - Group files under `prompt_{uid}` or `uid/{k}` rather than index-based folders.
- Steps:
  1. Read `uids = batch.non_tensor_batch['uid']`.
  2. Use `prompt_dir = os.path.join(base_dir, f"prompt_{uids[i]}")` and ensure it is a safe path component (sanitize if needed).
- Tests:
  - Check output folder structure under shuffled order; all candidates of the same prompt appear under the same `uid` folder.

### 9) Respect trainer.metadata_max_sequences

- Problem: The field exists but is not enforced when saving metadata.
- Fix:
  - Track how many responses per-uid have been saved and stop once it reaches `metadata_max_sequences` (unless -1).
- Steps:
  1. Within `_save_step_metadata`, maintain a map `uid_counts` during the loop; skip extras per uid.
- Tests:
  - Run with `metadata_max_sequences=1` and verify only one response per prompt is saved.

### 10) Normalization rollout_n source of truth

- Problem: CoE normalization uses a default `5` via `getattr(self.config, 'normalization_rollout_n', 5)`.
- Fix:
  - Read `rollout_n` directly from `data.non_tensor_batch['uid']` grouping (no magic number required), or use `self.config.rollout.n` at manager init if stable.
- Steps:
  1. Remove the default; compute group sizes from `uid_to_indices` map.
- Tests:
  - Vary `rollout.n` in config and ensure normalized rewards are correct.

### 11) Optional: separate validation reward manager

- Observation: Validation uses the same reward manager type as training. If we need different logic, extend config to include `val_reward: RewardConfig`.
- Fix (optional):
  - Add `val_reward: RewardConfig` to `WorkerConfig` and wire selection accordingly.

---

### Implementation order (recommended)

1. (1) Group-by-uid normalization in CoE manager
2. (2) Restore hidden-state order when dynamic batching
3. (3)(4) Mrope handling and CPU return in `get_hidden_states`
4. (8)(9) Metadata grouping by uid and max-sequences enforcement
5. (6) Remove token-id length heuristic
6. (7) Single-forward hidden-state capture (perf)
7. (5) Padding-free policy (warn/guard), or implement varlen pooling later
8. (10)(11) Cleanup and config polish

### Validation checklist

- Unit tests for CoE manager normalization with shuffled inputs and varying `rollout.n`
- E2E smoke run (1–2 steps) with `save_metadata=true`; verify:
  - Hidden states fetched and attached
  - Reward metrics include `overall` and a sensible distribution
  - Metadata directories grouped by uid; respect `metadata_max_sequences`
- Compare step time before/after eliminating the second forward

