from pathlib import Path
import torch
import sparse
from diba.diba.sparse_utils import convert_sparse_coo_to_torch_coo, sparse_normalize
from diba.diba.utils import normalize_logits
from diba.tests.utils import test

audio_root = Path(__file__).parent.parent.parent
custom_sums_path: Path = audio_root / "checkpoints/sum_dist_8800.npz"
default_sums_path: Path = audio_root / "checkpoints/sum_frequencies.npz"

device = torch.device("cpu")

def load_sums(sums_path: Path):
    # with open(sums_frequn, "rb") as f:
    sums_coo: sparse.COO = sparse.load_npz(sums_path)
    sums = convert_sparse_coo_to_torch_coo(sums_coo, device)
    sums = sparse_normalize(sums, dim=-1)
    return sums


# @test
def test_sums():
    custom_sums = load_sums(custom_sums_path)[0].coalesce()
    default_sums = load_sums(default_sums_path).coalesce()

    print("Custom sums ", custom_sums)
    print("Default sums", default_sums)
    custom_sum_sum = torch.sparse.sum(custom_sums).item()
    default_sum_sum = torch.sparse.sum(default_sums).item()
    print(f"delta", default_sum_sum - custom_sum_sum)

    custom_max_value = torch.max(custom_sums.values())
    default_max_value = torch.max(default_sums.values())

    print(f"Max value", custom_max_value, default_max_value)




if __name__ == "__main__":
    test_sums()
