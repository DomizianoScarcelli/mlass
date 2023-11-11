import functools
from pathlib import Path
from typing import Any, Tuple, Union

import time
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
        # TODO: the normalization now is done on dense tensors, but should be done on sparse tensors directly to avoid memory issues
        # This can also be done directly when computing the likelihood tensor

        # Dense sums normalized
        normalization = torch.sparse.sum(sums, dim=-1)
        dense_normalization = normalization.to_dense()
        dense_mask = dense_normalization != 0.0
        dense_sums = sums.to_dense()
        dense_sums[dense_mask] = (dense_sums[dense_mask] /
                                  dense_normalization[dense_mask].unsqueeze(-1))
        self.sum = dense_sums.to_sparse()

        self.num_sources = len(self.sum.shape) - 1

    def get_tokens_count(self) -> int:
        return self.sum.shape[0]

    @functools.lru_cache(512)
    def get_log_likelihood(self, token_idx: int) -> torch.Tensor:
        return slice_along_index(self.sum, token_idx)

    def get_m_marginal_likelihood(self):
        return torch.log(torch.sparse.sum(
            self.sum, dim=tuple(i for i in range(self.num_sources))).to_dense() + 1e-12)

    # TODO: this has to be optimized with belief propagation,
    # sinnce it takes 0.01 sec for 2 sources, but 3 sec for 3 sources
    def get_marginal_likelihood(self, source_idx: int) -> torch.Tensor:
        start = time.time()

        sum_dims = tuple(i for i in range(self.num_sources) if i != source_idx)
        sparse_marginal = torch.sparse.sum(
            self.sum, dim=sum_dims)

        end = time.time()
        print(f"Time to compute marginal likelihood: {end-start}")

        dense_marginal = sparse_marginal.to_dense()
        return torch.log(dense_marginal + 1e-12)


class DenseMarginalLikelihood():
    def __init__(self, sums: torch.Tensor):
        # normalize the sums over the conditionals in order to get the likelihood function at each z_m
        n_sources = sums.shape[0]
        print(f"Computing marginal probability")
        for i in range(n_sources):
            p_zi = torch.sum(sums[i], dim=0)
            sums[i] /= (p_zi + 1e-12)
        print(f"Marginal probability correctly computed")
        self.sums = torch.log(sums + 1e-12)
