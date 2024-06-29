import torch
from tqdm import tqdm
import math
from lass_mnist.lass.diba_interaces import UnconditionedTransformerPrior
from typing import List, Optional
from .utils import normalize_logits
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

class DirectedGraphicalModel:
    """
    Represents the Bayesian Network with the latent codes z_1,...,z_n and the
    mixtures m_1,...,m_n at all the stages i.
    """

    def __init__(self, 
                 transformer: GPT2LMHeadModel,
                 num_sources:int):
        self.K = 256 #possible discrete values in the latent codes
        self.num_sources = num_sources
        self.transformer = transformer
        self.past_key = [None for _ in range(num_sources)]

        # p_mmzs[i] = p(m_i | m{i-1}, z_i)
        p_mmzs_path = "./lass_mnist/models/sums-MNIST-gm/best.pt"
        with open(p_mmzs_path, "rb") as f:
            #TODO: the permute and unsqueeze is used just because I'm using the LASS pre-computed likelihood
            sums = torch.load(f)
            # normalization = torch.sum(sums, dim=-1)
            # mask = normalization != 0.0
            # sums[mask] = sums[mask] / normalization[mask].unsqueeze(-1)
            self.p_mmzs = torch.log(sums + 1e-12).permute(2,0,1).unsqueeze(0)
            # Normalization
            self.p_mmzs -= torch.logsumexp(self.p_mmzs, dim=1)
            self.pm = torch.logsumexp(self.p_mmzs, dim=[2,3]).squeeze()
            # self.p_mmzs = normalize_logits(self.p_mmzs)

            # print(f"p_mmzs: {self.p_mmzs}")
    
    def compute_priors(self, past) -> List[torch.Tensor]:
        priors = []
        for _ in range(self.num_sources):
            priors.append(UnconditionedTransformerPrior(transformer=self.transformer, sos=0))

        # p_zs[i] = p(z_i)
        self.p_zs = []
        for i in range(self.num_sources):
            log_prior, past_key = priors[i].get_logits(
                token_ids=past[i],
                past_key_values=self.past_key[i],
            )
            log_prior = normalize_logits(log_prior).squeeze() - self.pm

            # NOTE: this is pretty useful for the UnconditionedTransformerPrior since the past key is always none
            self.past_key[i] = past_key

            self.p_zs.append(log_prior)
                

        return priors

    def forward_pass(self, i: int, token_idx: torch.Tensor):
        """
        It computes the message μα in the graphical model forward pass.
        """
        # shape is [256, 256, 256] = [K, K, K]
        # shape is [2, 256] = [num_sources, K]
        if i == 0:
            return torch.logsumexp(self.p_zs[i] + self.p_mmzs[i, token_idx], dim=0)
        raise NotImplementedError()
        new_message = torch.logsumexp(curr_p_mmz + curr_p_z, dim=1)
        old_message = torch.logsumexp(self.p_zs[i] + self.forward_results[i-1], dim=1)
        final_message = new_message + old_message
        return final_message

    def backward_pass(self, i: int, token_idx: torch.Tensor):
        """
        It computes the message μβ in the graphical model backward pass.
        """
        if i == self.num_sources-1:
            #TODO: see better the axis where to perform logsumexp
            raise NotImplementedError()
            message = torch.logsumexp(self.p_zs[i+1], dim=0) + torch.logsumexp(self.p_mmzs[i], dim=0)
            old_message = torch.logsumexp(self.p_mmzs[i+1] + self.backward_results[i+2], dim=0)
            # print(f"backward old_message shape: {old_message.shape}")
            final_message = message + old_message
            # print(f"backward final_message shape: {final_message.shape}")
            return final_message

        return torch.logsumexp(self.p_zs[i+1] + self.p_mmzs[i, token_idx], dim=-1)
 

        # other_message = torch.logsumexp(self.p_mmzs[i], dim=-1) + torch.logsumexp(self.p_zs[i], dim=0)
        # other_message = torch.logsumexp(self.p_mmzs[i] + self.p_zs[i], dim=-1)

        # assert torch.allclose(message, other_message), f""" 
        # messge: {message}
        # other_message: {other_message}
        # """
        return message
    
    def compute_marginals(self, i: int) -> torch.Tensor:
        if i == 0:
            return self.p_zs[i] + self.backward_results[i]
        elif i == self.num_sources-1:
            return self.p_zs[i] + self.forward_results[i-1]
        else:
            raise NotImplementedError()
            return self.p_zs[i] + torch.logsumexp(self.forward_results[i-1] + self.backward_results[i], dim=0)

    def single_sample(self, marginals: torch.Tensor, mixture: torch.Tensor, i: int) -> torch.Tensor:
        # topk_indices = torch.topk(marginals, k=32, dim=-1).indices
        # mask = torch.zeros_like(marginals, dtype=torch.bool)
        # mask[:, topk_indices] = True
        # marginals[~mask] = math.log(1e-12)
        return torch.distributions.Categorical(logits=marginals).sample()
    
    def single_separate(self, mixture: torch.Tensor, i: int) -> torch.Tensor:
        self.forward_results: List[Optional[torch.Tensor]] = []
        self.backward_results: List[Optional[torch.Tensor]] = [None for _ in range(self.num_sources-1)]
        self.marginal_results = []
        #TODO: if I do not comment until one shot marginals, the result is different. Super strange stuff

        # for i in range(self.num_sources-1):
        #     forward = self.forward_pass(i, mixture[i])
        #     self.forward_results.append(forward)
        
        # backward_range = list(reversed([i for i in range(self.num_sources-1)]))
        # print(f"Initializing backward pass on the sequence: {backward_range}")
        # for i in backward_range: 
        #     backward = self.backward_pass(i, mixture[i])
        #     self.backward_results[i] = backward
            # print([f"Full: {elem.shape}" if elem is not None else "Empty" for elem in self.backward_results])
        
        # for i in range(self.num_sources):
        #     marginal = self.compute_marginals(i)
        #     self.marginal_results.append(marginal)

        # marginals = torch.stack(self.marginal_results)
        marginals = self.one_shot(mixture[i])

        # assert torch.allclose(marginals, o_marginals), f"""
        # marginals: {marginals}
        # o_marginals: {o_marginals}
        # mse: {F.mse_loss(marginals, o_marginals)}
        # """
        result = self.single_sample(marginals, mixture, i)
        return result

    def one_shot(self, token: torch.Tensor) -> torch.Tensor:
        # z0 = torch.logsumexp(self.p_zs[0], dim=-1)
        # z1 = torch.logsumexp(self.p_zs[1], dim=-1)
        z0 = self.p_zs[0]
        z1 = self.p_zs[1]
        marginal_0 = z0 + torch.logsumexp(z1 + self.p_mmzs[0, token], dim=-1)
        marginal_1 = z1 + torch.logsumexp(z0 + self.p_mmzs[0, token], dim=0)
        return torch.stack([marginal_0, marginal_1])

    def separate(self, mixture: torch.Tensor) -> torch.Tensor:
        self.prior_past = torch.full((2, self.K, len(mixture)+1), fill_value=-1, dtype=torch.long)
        self.prior_past[:,:, 0] = 0

        for i in tqdm(range(len(mixture)), desc="Separating mixture...", leave=False):
            self.compute_priors(past=self.prior_past[:, :, :i+1])
            sample = self.single_separate(mixture, i)
            self.prior_past[:, :, i+1] = sample
            # print(self.prior_past)
        return self.prior_past[:,:,1:]




