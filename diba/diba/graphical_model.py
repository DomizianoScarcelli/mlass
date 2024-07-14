import torch
from tqdm import tqdm
import math
from diba.diba import SeparationPrior
from typing import List, Optional, Union
from .utils import normalize_logits
import torch.nn.functional as F

class DirectedGraphicalModel:
    """
    Represents the Bayesian Network with the latent codes z_1,...,z_n and the
    mixtures m_1,...,m_n at all the stages i.
    """

    def __init__(self, 
                 priors: List[SeparationPrior],
                 sums: torch.Tensor,
                 num_sources:int):
        self.K = 256 #possible discrete values in the latent codes
        self.num_sources = num_sources
        self.priors = priors
        self.past_key = [None for _ in range(num_sources)]

        self.p_mmzs = torch.log(1e-12 + sums)
        print(self.p_mmzs.shape)
        # self.p_mmzs -= torch.logsumexp(self.p_mmzs, dim=[2,3]).unsqueeze(2).unsqueeze(3)
        self.p_mmzs -= torch.logsumexp(self.p_mmzs, dim=1).unsqueeze(1)
        # self.pm = torch.logsumexp(self.p_mmzs, dim=[2,3])[-1].squeeze()
        self.pm = 0
    
    def compute_priors(self, past) -> List[SeparationPrior]:
        # p_zs[i] = p(z_i)
        self.p_zs = torch.empty((self.num_sources, self.K, self.K))
        for i in range(self.num_sources):
            log_prior, past_key = self.priors[i]._get_logits(
                    past[i],
                    self.past_key[i])
            log_prior = normalize_logits(log_prior) - self.pm

            # NOTE: this is pretty much useless for the UnconditionedTransformerPrior since the past key is always none
            self.past_key[i] = past_key

            self.p_zs[i] = log_prior
                
        return self.priors

    def forward_pass(self, i: int, token_idx: torch.Tensor):
        """
        It computes the message μα in the graphical model forward pass.
        """
        # shape is [256, 256, 256] = [K, K, K]
        # shape is [2, 256] = [num_sources, K]
        prior = torch.logsumexp(self.p_zs[i], dim=-1)
        if i == 0:
            return torch.logsumexp(self.p_mmzs[i, token_idx].unsqueeze(0) + prior, dim=1)
        old_message = torch.logsumexp(prior + self.forward_results[i-1], dim=-1) # this was -1, but with 0 the performances are better
        final_message = torch.logsumexp(self.p_mmzs[i, token_idx].unsqueeze(0) + old_message , dim=1)
        return final_message

    def backward_pass(self, i: int, token_idx: torch.Tensor):
        """
        It computes the message μβ in the graphical model backward pass.
        """
        prior = torch.logsumexp(self.p_zs[i+1], dim=-1)
        if i == self.num_sources-2:
            return torch.logsumexp(prior + self.p_mmzs[i, token_idx].unsqueeze(0), dim=-1)
        
        old_message = torch.logsumexp(self.p_mmzs[i, token_idx].unsqueeze(0) + self.backward_results[i+1], dim=0)
        final_message = torch.logsumexp(prior + old_message, dim=-1) # this was -1, but with 0 the performances are better
        return final_message

 
    def compute_marginals(self, i: int) -> torch.Tensor:
        prior = self.p_zs[i]
        if i == 0:
            return prior + self.backward_results[i] 
        elif i == self.num_sources-1:
            return prior + self.forward_results[i-1] 
        else:
            return prior + torch.logsumexp(self.forward_results[i-1] + self.backward_results[i], dim=0)

    def single_sample(self, marginals: torch.Tensor, topk: Union[bool, int]= True) -> torch.Tensor:
        if topk:
            topk_values, topk_indices = torch.topk(marginals, k=topk, dim=-1)
            m = torch.full_like(marginals, fill_value=math.log(1e-12))
            m[:, :,topk_indices] = marginals[:, :,topk_indices]
            marginals = m
        sample = torch.distributions.Categorical(logits=marginals).sample()
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

        result = self.single_sample(marginals, topk=64)
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
        self.prior_past = torch.full((self.num_sources, self.K, len(mixture)+1), fill_value=-1, dtype=torch.long)
        self.prior_past[:,:, 0] = 0

        for i in tqdm(range(len(mixture)), desc="Separating mixture...", leave=False):
            self.compute_priors(past=self.prior_past[:, :, :i+1])
            sample = self.single_separate(mixture, i)
            self.prior_past[:, :, i+1] = sample
            # print(self.prior_past)
        return self.prior_past[:,:,1:]




