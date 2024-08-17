"""
This file tests the correctness of the
`lass_audio/lass/train_sums_graphical_model.py` sums computation. Since sparse
tensors operations are not trivial and can lead to errors, the idea is to
compute a smaller sums using a smaller dataset and a smaller K and compare it
to its dense tensor counterpart. If they are equal, then the tensor in the
sparse domain is computer correctly. This cannot be done for the real larger K,
since the memory requirement is too large.
"""

from pathlib import Path
from lass_audio.lass.train_sums_graphical_model import estimate_distribution
from lass_audio.lass.train_sums_graphical_model_dense import estimate_distribution as estimate_distribution_dense
import torch
import sparse
from diba.diba.sparse_utils import convert_sparse_coo_to_torch_coo, sparse_normalize
from diba.diba.utils import normalize_logits
from diba.tests.utils import test

audio_root = Path(__file__).parent.parent.parent
device = torch.device("cpu")

def load_sums(sums_path: Path):
    """
    Loads and normalizes the sparse tensor given at the `sums_path` location.
    """
    sums_coo: sparse.COO = sparse.load_npz(sums_path)
    sums = convert_sparse_coo_to_torch_coo(sums_coo, device)
    sums = sparse_normalize(sums, dim=-1)
    return sums


def compute_sparse_sums(k: int = 128):
    """
    Compute the sparse sums using a given k and an audio directory.
    """
    ROOT_DIRECTORY = Path(__file__).parent.parent.parent
    vqvae_path = str(ROOT_DIRECTORY / "checkpoints/vqvae.pth.tar")
    audio_files_dir = str(ROOT_DIRECTORY / "data/train")
    sr = 44100
    sl = 11.88862

    output_dir = str(ROOT_DIRECTORY / "logs/vqvae_sum_distribution_gm")
    alpha = (0.5, 0.5)
    batch_size = 8
    save_iters = 500
    checkpoint_path=None
    vqvae_level=2


    hps = setup_hparams(
            "vqvae",
            dict(
                l_bins=k,
                sr=sr,
                restore_vqvae=vqvae_path,
                sample_length=int(sl * sr),
                audio_files_dir=audio_files_dir,
                min_duration=sl + 0.01,  # add a small epsilon
                labels=False,
                aug_shift=True,
                aug_blend=True,
                ),
            )
    sums = estimate_distribution(hps=hps,
                                 output_dir=output_dir,
                                 epochs=1,
                                 batch_size=batch_size,
                                 alpha=alpha,
                                 save_iters=save_iters,
                                 vqvae_level=vqvae_level,
                                 checkpoint_path=checkpoint_path)
    return sums

def compute_dense_sums(k: int, audio_dir: Path | str):
    """
    Compute the dens sums using a given k and an audio directory.
    """
    pass



@test
def test_sums():
    """
    Actually performs the test.
    """
    K = 256
    small_audio_dir = Path("")
    pass



if __name__ == "__main__":
    test_sums()
