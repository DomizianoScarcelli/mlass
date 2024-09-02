from typing import Tuple

import torch
import os
import json
from pathlib import Path


def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> Tuple[torch.LongTensor, ...]:
    """Convert flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).

    """
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim
    return tuple(coord[::-1])


def get_topk(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, Tuple[torch.LongTensor, ...]]:
    """Get top k elements in tensor.

    Args:
        x: A tensor of values of any shape

    Returns:
        A tuple containing:
        - Tensor containing the k greatest elements in ascending order. Shape: (k,)
        - Tuple of indices for the top k elements. The number of elements in the
         tuple is the same as the number of dimensions in token_idx

    """
    # log_post_sum, log_post_index = torch.topk(log_posterior.flatten(), n_samples)
    x_sorted, log_post_index = torch.sort(x.flatten(), dim=-1)
    x_sorted, log_post_index = x_sorted[-k:], log_post_index[-k:]
    idx = unravel_indices(log_post_index, x.shape)
    return x_sorted, idx


def normalize_logits(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Normalize Logits between (-inf, 0.0] with additional temperature.

    Args:
        x: Tensor containing the logits to normalize. This must have values
         between (-inf, inf) and shape (*, N)
        temperature: temperature to use in normalization

    Returns:
        The normalized logits between (-inf, 0.0], with the same shape as token_idx

    """
    return torch.distributions.Categorical(logits=x / temperature).logits


def compute_sdr(s_ref: torch.Tensor, s_est: torch.Tensor) -> float:
    # SDR for (s_ref, s_est)
    power_ref = torch.sum(s_ref ** 2)
    power_error = torch.sum((s_ref - s_est) ** 2)
    sdr_1 = 10 * torch.log10(power_ref / power_error)
    
    print(f"SDR_1: {sdr_1}")
    # SDR for (s_est, s_ref)
    power_est = torch.sum(s_est ** 2)
    power_error_reverse = torch.sum((s_est - s_ref) ** 2)
    sdr_2 = 10 * torch.log10(power_est / power_error_reverse)
    
    print(f"SDR_2: {sdr_2}")
    # Mean of both SDRs
    sdr_mean = 0.5 * (sdr_1 + sdr_2)
    
    return sdr_1.item(), sdr_2.item(), sdr_mean.item()

def save_sdr(sdr:float, path: Path):
    if not os.path.exists(path):
        content = {"sdr_1":[], "sdr_2":[], "sdr_mean":[]}
    else:
        with open(path, "r") as f:
            content = json.load(f)
    content["sdr_1"].append(sdr[0])
    content["sdr_2"].append(sdr[1])
    content["sdr_mean"].append(sdr[2])
    with open(path, "w") as f:
        json.dump(content, f)
