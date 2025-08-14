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


def compute_score(
    predict: str,
    ground_truth: str,
    hidden: torch.Tensor = None,
    reward_strategy: str = "d1_mean_plus_d2_mean",
) -> Dict[str, float]:
    s = coe_reward(hidden)
    print(f"[CoE Reward] d1_mean={s['d1_mean']}, d2_mean={s['d2_mean']}")
    overall = (s["d1_mean"] + s["d2_mean"]) / 2
    return {"overall": overall, "d1_mean": s["d1_mean"], "d2_mean": s["d2_mean"]}

