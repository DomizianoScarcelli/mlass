import torch
from tqdm import tqdm
import math
from diba.diba import SeparationPrior
import numpy as np
from typing import List, Optional, Union

from diba.diba.sparse_utils import convert_torch_coo_to_sparse_coo, sparse_elementwise_div, sparse_expand, sparse_expand_as, sparse_normalize, sparse_permute, convert_sparse_coo_to_torch_coo
from .utils import normalize_logits
from functools import wraps
import sparse
import time

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

class SparseDirectedGraphicalModel:
    """
    Represents the Bayesian Network with the latent codes z_1,...,z_n and the
    mixtures m_1,...,m_n at all the stages i.
    """

    def __init__(self, 
                 priors: List[SeparationPrior],
                 sums: torch.Tensor,
                 num_tokens: int,
                 num_sources: int,
                 topk: Optional[int] = None):
        self.K = num_tokens #possible discrete values in the latent codes
        self.num_sources = num_sources
        self.priors = priors
        self.past_key = [None for _ in range(num_sources)]
        self.num_beams = 50
        self.device = sums.device
        self.topk = topk

        print(f"Using device: {self.device}")
        
        # Take the log
        if len(sums.shape) == 3:
            sums = sums.unsqueeze(0)
        sums = sparse_permute(sums, (0,3,1,2))
        sums = sparse_normalize(sums, dim=1).to(self.device)
        #Indices of n_sums and sums are equal :)
        self.p_mmzs = sums
        self.pm = 0 

           
    @timeit 
    def compute_priors(self, past) -> List[SeparationPrior]:
        self.p_zs = torch.empty((self.num_sources, self.num_beams, self.K)).to(self.device)
        for i in range(self.num_sources):
            log_prior, past_key = self.priors[i]._get_logits(
                    past[i],
                    self.past_key[i])
            log_prior = (normalize_logits(log_prior) - self.pm).to(self.device)
            self.past_key[i] = past_key.to(self.device) if past_key is not None else past_key
            self.p_zs[i] = log_prior
        return self.priors
    
    @timeit
    def forward_pass(self, i: int, token_idx: torch.Tensor):
        """
        It computes the message μα in the graphical model forward pass.
        """
        # shape is [256, 256, 256] = [K, K, K]
        # shape is [2, 256] = [num_sources, K]
        prior = torch.logsumexp(self.p_zs[i], dim=[0])
        sums = torch.log(self.p_mmzs[i, token_idx].unsqueeze(0).to_dense() + 1e-12)
        if i == 0:
            return torch.logsumexp(prior + sums, dim=1)
        old_message = torch.logsumexp(prior + self.forward_results[i-1], dim=0)
        final_message = torch.logsumexp(old_message + sums, dim=1)
        return final_message

    @timeit
    def backward_pass(self, i: int, token_idx: torch.Tensor):
        """
        It computes the message μβ in the graphical model backward pass.
        """
        prior = torch.logsumexp(self.p_zs[i+1], dim=[0])
        sums = torch.log(self.p_mmzs[i, token_idx].unsqueeze(0).to_dense() + 1e-12)
        if i == self.num_sources-2:
            message =  torch.logsumexp(prior + sums, dim=0)
            return torch.logsumexp(prior + message, dim=-1)
        
        old_message = torch.logsumexp(sums + self.backward_results[i+1], dim=0)
        final_message = torch.logsumexp(prior + old_message, dim=0) 
        return final_message

    @timeit 
    def compute_marginals(self, i: int) -> torch.Tensor:
        # prior = torch.logsumexp(self.p_zs[i], dim=-1).unsqueeze(-1)
        prior = self.p_zs[i]
        if i == 0:
            return prior + self.backward_results[i] 
        elif i == self.num_sources-1:
            return prior + self.forward_results[i-1] 
        else:
            return prior + torch.logsumexp(self.forward_results[i-1] + self.backward_results[i], dim=0)
    
    @timeit
    def single_sample(self, marginals: torch.Tensor, topk: Optional[int]=None) -> torch.Tensor:
        if topk:
            _, topk_indices = torch.topk(marginals, k=topk, dim=-1)
            m = torch.full_like(marginals, fill_value=math.log(1e-12)).to(self.device)
            m[:, :,topk_indices] = marginals[:, :,topk_indices]
            marginals = m
        sample = torch.distributions.Categorical(logits=marginals).sample().to(self.device)
        return sample

    def single_separate(self, mixture: torch.Tensor, i: int) -> torch.Tensor:
        self.forward_results: List[torch.Tensor] = []
        self.backward_results: List[Optional[torch.Tensor]] = [None for _ in range(self.num_sources-1)]
        self.marginal_results = []

        for j in range(self.num_sources-1):
            forward = self.forward_pass(j, mixture[i]).squeeze()
            # print("Forward: ", forward.shape)
            self.forward_results.append(forward)
        
        backward_range = list(reversed([k for k in range(self.num_sources-1)]))
        # print(f"Initializing backward pass on the sequence: {backward_range}")
        for j in backward_range: 
            backward = self.backward_pass(j, mixture[i])
            # print("Backward: ", backward.shape)
            self.backward_results[j] = backward
            # print([f"Full: {elem.shape}" if elem is not None else "Empty" for elem in self.backward_results])
        
        for j in range(self.num_sources):
            marginal = self.compute_marginals(j)
            self.marginal_results.append(marginal)

        marginals = torch.stack(self.marginal_results)
        # marginals = self.one_shot(mixture[i])
        
        result = self.single_sample(marginals, topk=self.topk)
        return result

    # def one_shot(self, token: torch.Tensor) -> torch.Tensor:
    #     # z0 = torch.logsumexp(self.p_zs[0], dim=-1)
    #     # z1 = torch.logsumexp(self.p_zs[1], dim=-1)
    #     z0 = self.p_zs[0]
    #     z1 = self.p_zs[1]
    #     marginal_0 = z0 + torch.logsumexp(z1 + self.p_mmzs[0, token], dim=-1)
    #     marginal_1 = z1 + torch.logsumexp(z0 + self.p_mmzs[0, token], dim=0)
    #     return torch.stack([marginal_0, marginal_1])

    def separate(self, mixture: torch.Tensor) -> torch.Tensor:
        #NOTE: I'm pretty sure this is correct
        self.prior_past = torch.full((self.num_sources, self.K, len(mixture)+1), fill_value=-1, dtype=torch.long).to(self.device)
        self.prior_past[:,:, 0] = 0
        
        for i in tqdm(range(len(mixture)), desc="Separating mixture...", leave=False):
            self.compute_priors(past=self.prior_past[:, :self.num_beams, :i+1])
            sample = self.single_separate(mixture, i).to(self.device)
            self.prior_past[:, :self.num_beams, i+1] = sample
        print(self.prior_past)
        return self.prior_past[:,:self.num_beams,1:][:,-1]




