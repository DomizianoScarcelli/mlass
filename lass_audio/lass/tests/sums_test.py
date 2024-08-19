"""
This file tests the correctness of the
`lass_audio/lass/train_sums_graphical_model.py` sums computation. Since sparse
tensors operations are not trivial and can lead to errors, the idea is to
compute a smaller sums using a smaller dataset and a smaller K and compare it
to its dense tensor counterpart. If they are equal, then the tensor in the
sparse domain is computer correctly. This cannot be done for the real larger K,
since the memory requirement is too large.

TODO: there is a problem, since the VQVAE is trained with K=2048, it cannot be
used with a different K, hence I should train another VQVAE just for this test,
which I don't  think I will do. 

TODO @1002: the solution is rather simple, just fake the data with a smaller
K, create the mixture and see if the sparse matrix computed in the train_sums
is equal to the dense matrix.
"""

from tqdm import tqdm
from lass_audio.jukebox.hparams import setup_hparams
from lass_audio.lass.train_sums_graphical_model import split_datapoints_by_step, get_mixtures
from typing import Union, List
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

def compute_sparse_sums(k: int, num_sources: int, iterations: int):
    """
    Compute the sparsesums of using num_sources arrays of length k.
    """
    prefix_s, prefix_i, prefix_j, prefix_k, prefix_data = [], [], [], [], []
    buffer_adds: List[List[int]] = [[] for _ in range(NUM_SOURCES)]
    buffer_sums: List[List[int]] = [[] for _ in range(NUM_SOURCES-1)]
    batch_size = 8 
    device = torch.device("cpu")
    x = torch.randn((batch_size, 524288, 1))
    xs = split_datapoints_by_step(x, batch_size, step=NUM_SOURCES, device=device)
    assert xs.shape == (num_sources, batch_size // num_sources, 524288, 1), f"xs wrong shape: {xs.shape}"
    mixtures = get_mixtures(datapoints=xs, device=device)
    assert mixtures.shape == (NUM_SOURCES-1, 2, 524288, 1), f"mixtures wrong shape: {mixtures.shape}"
    for i, x_i in enumerate(xs):
        latent = mock_compute_latent(x_i, k)
        assert len(latent) == ((batch_size // NUM_SOURCES) * k), f"x_i latent wrong shape: {torch.tensor(latent).shape}"
        buffer_adds[i].extend(latent)
    for i, m_i in enumerate(mixtures):
        latent = mock_compute_latent(m_i, k)
        assert len(latent) == ((batch_size // NUM_SOURCES) * k), f"m_i latent wrong shape: {torch.tensor(latent).shape}"
        buffer_sums[i].extend(latent)
    coords = []
    data = []
    for _ in tqdm(range(iterations), desc="Computing sparse sums"):
        for i in range(NUM_SOURCES-1):
            if i == 0:
                coords = [
                        prefix_s + [i] * len(buffer_adds[0]) * 2,
                        prefix_i + buffer_adds[i] + buffer_adds[i+1],
                        prefix_j + buffer_adds[i+1] + buffer_adds[i],
                        prefix_k + buffer_sums[i] + buffer_sums[i]
                ]
                data = prefix_data + [1] * (len(buffer_adds[1]) * 2)
            else:
                coords=[
                    coords[0] + [i] * (len(buffer_adds[0]) + len(buffer_sums[0])),
                    coords[1] + buffer_sums[i-1] + buffer_adds[i+1],
                    coords[2] + buffer_adds[i+1] + buffer_sums[i-1],
                    coords[3] + buffer_sums[i] + buffer_sums[i],
                ]
                data += [1] * (len(buffer_adds[1]) * 2)

        sum_dist = sparse.COO(
            coords=coords,
            data=data,
            shape=[NUM_SOURCES-1, k, k, k],
        )
        prefix_s, prefix_i, prefix_j, prefix_k = sum_dist.coords.tolist()
        prefix_data = sum_dist.data.tolist()
        buffer_adds: List[List[int]] = [[] for _ in range(NUM_SOURCES)]
        buffer_sums: List[List[int]] = [[] for _ in range(NUM_SOURCES-1)]
    
    return sum_dist

def compute_dense_sums(k: int, num_sources: int, iterations: int):
    """
    Compute the dense sums of using num_sources arrays of length k.
    """
    prefix_s, prefix_i, prefix_j, prefix_k, prefix_data = [], [], [], [], []
    buffer_adds: List[List[int]] = [[] for _ in range(NUM_SOURCES)]
    buffer_sums: List[List[int]] = [[] for _ in range(NUM_SOURCES-1)]
    batch_size = 8 
    device = torch.device("cpu")
    x = torch.randn((batch_size, 524288, 1))
    xs = split_datapoints_by_step(x, batch_size, step=NUM_SOURCES, device=device)
    assert xs.shape == (num_sources, batch_size // num_sources, 524288, 1), f"xs wrong shape: {xs.shape}"
    mixtures = get_mixtures(datapoints=xs, device=device)
    assert mixtures.shape == (NUM_SOURCES-1, 2, 524288, 1), f"mixtures wrong shape: {mixtures.shape}"
    for i, x_i in enumerate(xs):
        latent = mock_compute_latent(x_i, k, flatten=True)
        assert len(latent) == ((batch_size // NUM_SOURCES) * k), f"x_i latent wrong shape: {torch.tensor(latent).shape}"
        buffer_adds[i].append(latent)
    for i, m_i in enumerate(mixtures):
        latent = mock_compute_latent(m_i, k, flatten=True)
        assert len(latent) == ((batch_size // NUM_SOURCES) *k), f"m_i latent wrong shape: {torch.tensor(latent).shape}"
        buffer_sums[i].append(latent)
    sums_dist = torch.full((NUM_SOURCES-1, k, k, k), fill_value=0)
    for _ in tqdm(range(iterations), desc="Computing dense sums"):
        for i in range(NUM_SOURCES-1):
            if i == 0:
                sums_dist[i, buffer_sums[i], buffer_adds[i], buffer_adds[i+1]] += 1
            else:
                sums_dist[i, buffer_sums[i], buffer_sums[i-1], buffer_adds[i+1]] += 1
    return sums_dist

def test_sums():
    """
    Actually performs the test.
    """
    K = 8
    NUM_SOURCES = 3
    ITERATIONS = 10_000
    sparse_sums = compute_sparse_sums(k=K, num_sources=NUM_SOURCES, iterations=ITERATIONS)
    dense_sums = compute_dense_sums(k=K, num_sources=NUM_SOURCES, iterations=ITERATIONS)
    
    assert dense_sums == sparse_sums.todense(), f"""
    Sparse sums: {sparse_sums.todense()}
    Dense sums: {dense_sums}
    """



if __name__ == "__main__":
    torch.manual_seed(42)
    test_sums()
