import functools
from pathlib import Path
from typing import Any, Tuple, Union

import torch
from hydra.core.config_store import ConfigStore

from diba.diba import Likelihood, SeparationPrior
from transformers import GPT2LMHeadModel

from diba.diba.sparse_utils import slice_along_index


class UnconditionedTransformerPrior(SeparationPrior):
    def __init__(self, transformer: GPT2LMHeadModel, sos: int):
        self.transformer = transformer
        self.transformer.eval()
        self.label = sos

    def __repr__(self):
        return f"UnconditionedTransformerPrior, label={self.label}"

    def __str__(self):
        return f"UnconditionedTransformerPrior, label={self.label}"

    def get_sos(self):
        return torch.tensor(self.label, dtype=torch.long).view(1).to(self.get_device())

    def get_device(self) -> torch.device:
        return self.transformer.device

    def get_tokens_count(self) -> int:
        return self.transformer.lm_head.out_features

    def get_logits(self, token_ids: torch.LongTensor, past_key_values: Any = None) -> Tuple[torch.Tensor, Any]:
        token_ids = token_ids[:, -
                              1:] if past_key_values is not None else token_ids
        output = self.transformer(
            input_ids=token_ids, past_key_values=past_key_values)
        logits, past = output.logits, output.past_key_values
        return logits[:, -1, :], past

    def reorder_cache(self, past_key_values: Any, beam_idx) -> Any:
        return self.transformer._reorder_cache(past_key_values, beam_idx)


class DenseLikelihood(Likelihood):
    def __init__(self, sums: torch.Tensor):
        # normalize the sums over the conditionals in order to get the likelihood function at each z_m
        normalization = torch.sum(sums, dim=-1)
        mask = normalization != 0.0
        sums[mask] = sums[mask] / normalization[mask].unsqueeze(-1)
        self.sum = sums

    def get_device(self) -> torch.device:
        return self.sum.device

    def get_tokens_count(self) -> int:
        return self.sum.shape[0]

    @functools.lru_cache(512)
    def get_log_likelihood(self, token_idx: int) -> Tuple[torch.LongTensor, torch.Tensor]:
        # Shape of mixture_slice is (num_tokens, num_tokens) (In the case of MNIST this is 256x256)
        mixture_slice = self.sum[:, :, token_idx]
        # The coords shape is (2, non_zero) (In the case of MNIST this is 2xnon_zero)
        coords = torch.nonzero(mixture_slice).transpose(0, 1)
        return coords, torch.log(mixture_slice[coords[0], coords[1]])

    def get_dense_log_likelihood(self, token_idx: Union[int, None] = None) -> torch.Tensor:
        if token_idx is None:
            return self.sum
        mixture_slice = self.sum[:, :, token_idx]
        return torch.log(mixture_slice)


class SparseLikelihood(Likelihood):
    def __init__(self, sums: torch.Tensor):
        # TODO: the normalization now is done on dense tensors, but should be done on sparse tensors directly

        # Dense sums normalized
        normalization = torch.sparse.sum(sums, dim=-1)
        dense_normalization = normalization.to_dense()
        dense_mask = dense_normalization != 0.0
        dense_sums = sums.to_dense()
        dense_sums[dense_mask] = (dense_sums[dense_mask] /
                                  dense_normalization[dense_mask].unsqueeze(-1))
        self.sum = dense_sums.to_sparse()

        # # TODO: Sparse sums normalized
        # # Create a mask for non-zero sums to avoid division by zero
        # normalization = normalization.unsqueeze(-1)
        # non_zero_indices = normalization._indices()
        # non_zero_values = normalization._values()

        # # Norm values are tensor([[    0.57,     0.39,     0.32,  ...,     0.02,     0.03,     0.03],
        # # [    0.57,     0.00,     0.01,  ...,     0.00,     0.00,     0.03]])

        # print(f"Sums indices have shape {sums._indices().shape}")
        # print(f"non_zero_indices have shaep {non_zero_indices.shape}")

        # indices_to_normalize = sums._indices(
        # )[non_zero_indices]

        # print(f"Indices to normalize are {indices_to_normalize}")

        # norm_values = sums._values().t()[indices_to_normalize] / \
        #     non_zero_values.unsqueeze(-1)

        # print(
        #     f"Norm values are {norm_values}, the indices have shape {non_zero_indices.shape} and the values have shape {non_zero_values.shape}")
        # print(f"Dense sums non zero are {dense_sums[dense_sums != 0.0]}")

        # new_values = sums._values().detach().clone()
        # new_values[non_zero_indices] = norm_values

        # print(f"New values shape is {new_values.shape}")
        # print(f"New values are {new_values}")
        # sums = torch.sparse_coo_tensor(
        #     sums._indices(), new_values, sums.shape)

        # assert torch.allclose(torch.nonzero(
        #     dense_sums), torch.nonzero(sums.to_dense()))

        # assert torch.allclose(sums.to_dense(), dense_sums), f"""
        #     Dense sums are not equal to sparse sums
        #     Dense sums: {dense_sums} with shape {dense_sums.shape}
        #     Sparse sums: {sums.to_dense()} with shape {sums.to_dense().shape}
        #     There are a total of different {torch.sum(dense_sums != sums.to_dense())} elements out of non zero elements {torch.sum(dense_sums != 0.0)}
        #     The non zero elements for the dense sums are at indices {torch.nonzero(dense_sums)}
        #     The non zero elements for the sparse sums are at indices {torch.nonzero(sums.to_dense())}
        # """

        self.num_sources = len(self.sum.shape) - 1

    def get_tokens_count(self) -> int:
        return self.sum.shape[0]

    @functools.lru_cache(512)
    def get_log_likelihood(self, token_idx: int) -> torch.Tensor:
        return slice_along_index(self.sum, token_idx)

    def get_m_marginal_likelihood(self):
        return torch.log(torch.sparse.sum(
            self.sum, dim=tuple(i for i in range(self.num_sources))).to_dense() + 1e-12)

    def get_marginal_likelihood(self, source_idx: int) -> torch.Tensor:
        sum_dims = tuple(i for i in range(self.num_sources) if i != source_idx)
        dense_marginal = torch.sparse.sum(
            self.sum, dim=sum_dims).to_dense()

        return torch.log(dense_marginal + 1e-12)
