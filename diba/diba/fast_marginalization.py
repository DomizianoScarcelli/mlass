import torch


def fast_marginalization(likelihood: torch.Tensor, source_idx: int):
    # p(m | z^3) = #sum_z1 sumz2 p(m, z1, z2, z3)
    pass
