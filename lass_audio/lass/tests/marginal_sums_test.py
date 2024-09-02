"""
This file tests the correctness of the
`lass_audio/lass/train_sums_marginal.py` sums computation.
"""

from tqdm import tqdm
from lass_audio.jukebox.hparams import setup_hparams
from lass_audio.lass.train_sums_graphical_model import split_datapoints_by_step, get_mixtures
from typing import Union, List, Tuple
import torch
import sparse
from diba.tests.utils import test
import numpy as np
import hashlib

NUM_SOURCES = 3

def mock_compute_latent(x: torch.Tensor, k: int, flatten: bool = True):
    # Ensure x is 2D (batch_size, ...), where each row is a different sample
    batch_size = x.shape[0]
    # Initialize an empty tensor to store the latents for each sample in the batch
    latents = torch.empty((batch_size, k), dtype=torch.int64)
    for i in range(batch_size):
        # Hash the individual sample from the batch to create a unique seed
        x_hash = hashlib.sha256(x[i].numpy().tobytes()).hexdigest()
        seed = int(x_hash, 16) % (2**32)  # Convert hash to a seed value
        # Set the seed for deterministic behavior
        torch.manual_seed(seed)
        # Generate a tensor of k integers from 0 to k-1 for this sample
        latents[i] = torch.randint(0, k, (k,))
    torch.manual_seed(42)
    if flatten:
        latents = latents.flatten().tolist()
    return latents

def compute_marginal_sums(x: torch.Tensor, k: int, num_sources: int, iterations: int):
    """
    Compute the dense sums of using num_sources arrays of length k.
    """
    buffer_adds = []
    buffer_sums = []
    batch_size = 8 
    device = torch.device("cpu")
    xs = split_datapoints_by_step(x, batch_size, step=NUM_SOURCES, device=device)
    assert xs.shape == (num_sources, batch_size // num_sources, 524288, 1), f"xs wrong shape: {xs.shape}"
    mixtures = get_mixtures(datapoints=xs, device=device)
    assert mixtures.shape == (NUM_SOURCES-1, 2, 524288, 1), f"mixtures wrong shape: {mixtures.shape}"
    for i, x_i in enumerate(xs):
        latent = mock_compute_latent(x_i, k, flatten=False)
        # assert len(latent) == ((batch_size // NUM_SOURCES), k), f"x_i latent wrong shape: {torch.tensor(latent).shape}"
        buffer_adds.append(latent)
    for i, m_i in enumerate(mixtures):
        latent = mock_compute_latent(m_i, k, flatten=False)
        # assert len(latent) == ((batch_size // NUM_SOURCES), k), f"m_i latent wrong shape: {torch.tensor(latent).shape}"
        buffer_sums.append(latent)

    codes = torch.stack(buffer_adds, dim=0)
    codes_mixture = torch.stack(buffer_sums, dim=0)
    sums = torch.zeros((NUM_SOURCES, k, k)).to(device)
    for _ in range(iterations):
        for i in range(NUM_SOURCES):
            for j, code in enumerate(codes[i]):
                sums[i, code, codes_mixture[j]] += 1
    return sums

def get_coords(x: torch.Tensor):
    non_zero_indices = torch.nonzero(x, as_tuple=True)
    coords_dict = {tuple(id.item() for id in idx): x[idx].item() for idx in zip(*non_zero_indices)}
    return coords_dict

@test
def test_sums():
    """
    Actually performs the test.
    """
    K = 12
    NUM_SOURCES = 3
    ITERATIONS = 1
    batch_size = 8
    x = torch.randn((batch_size, 524288, 1))
    sums = compute_marginal_sums(x=x, k=K, num_sources=NUM_SOURCES, iterations=ITERATIONS)
    # print(sums)
    print(get_coords(sums))
    



if __name__ == "__main__":
    torch.manual_seed(42)
    test_sums()
