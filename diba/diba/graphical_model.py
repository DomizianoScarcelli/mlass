
import torch
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
        # Defining the strucure of the model
        self.K = 256 #possible discrete values in the latent codes
        past = torch.zeros((num_sources, self.K, 1)).long()
        self.num_sources = num_sources

        #autoregressive priors
        priors = []
        for _ in range(num_sources):
            priors.append(UnconditionedTransformerPrior(
                transformer=transformer, sos=0))

        # p_zs[i] = p(z_i)
        self.p_zs = []
        for i in range(num_sources):
            log_prior, _ = priors[i].get_logits(
                token_ids=past[i],
                past_key_values=None,
            )
            log_prior = normalize_logits(log_prior).squeeze()
            print(f"log_prior {i} has shape {log_prior.shape}")
            self.p_zs.append(log_prior)
        
        # p_mmzs[i] = p(m_i | m{i-1}, z_i)
        p_mmzs_path = "./lass_mnist/models/sums-MNIST-gm/best.pt"
        with open(p_mmzs_path, "rb") as f:
            self.p_mmzs = torch.log(torch.load(f) + 1e-10)
            print(self.p_mmzs)
            print(f"p_mmzs has shape: {self.p_mmzs.shape}")

    def forward_pass(self, i: int, mixture: torch.Tensor):
        """
        It computes the message μα in the graphical model forward pass.
        """
        #TODO: apart from updating the shapes, you should also re-consider the
        # whole thing, since the forward pass is only defined at i > 0, while
        # the backward pass is defined at all indices with i != num_sources.

        # shape is [256, 256, 256] = [K, K, K]
        curr_p_mmz = self.p_mmzs[i]
        # shape is [2, 256] = [num_sources, K]
        curr_p_z = self.p_zs[i]
        # print(f"curr_p_mmz shape is: {curr_p_mmz.shape}")
        # print(f"curr_p_z shape is: {curr_p_z.shape}")
        if i == 0:
            # shape before sum is [mixture_length, K, num_sources]
            # shape after sum is [mixture_length, num_sources]
            # message = torch.sum(curr_p_mmz @ curr_p_z.T, dim=1)
            message = torch.logsumexp(curr_p_mmz + curr_p_z, dim=1)
            print(f"message shape is: {message.shape}")
            return message
        #shape [49,2]
        new_message = torch.logsumexp(curr_p_mmz + curr_p_z, dim=1)
        # shape: [49, 1]
        old_message = torch.logsumexp(self.p_zs[i-1] + self.forward_results[i-1], dim=0)
        # print(f"old message shape is: {old_message.shape}")
        # shape: [49,2]
        final_message = new_message + old_message
        # print(f"final message shape is {final_message.shape}")
        return final_message

    def backward_pass(self, i: int, mixture: torch.Tensor):
        """
        It computes the message μβ in the graphical model backward pass.
        """
        print(f"backward: currently on i: {i}")
        if i == -1:
            message = torch.logsumexp(self.p_zs[i+1], dim=0) #shape should be K
            old_message = torch.logsumexp(self.p_mmzs[i+1] + self.backward_results[i+2], dim=0)
            print(f"backward old_message shape: {old_message.shape}")
            final_message = message + old_message
            print(f"backward final_message shape: {final_message.shape}")
            return final_message

        #shape: [2]
        message = torch.logsumexp(self.p_zs[i+1], dim=0) 
        print(f"backward message shape: {message.shape}")
        return message
    
    def compute_marginals(self, i: int) -> torch.Tensor:
        print(f"{i} p_zi shape: {self.p_zs[i].shape}")
        print(f"{i} forward shape: {self.forward_results[i].shape}")
        print(f"{i} backward shape: {self.backward_results[i].shape}")
        if i == 0:
            return self.p_zs[i]*self.backward_results[i]
        else:
            past_message = torch.logsumexp(self.forward_results[i] + self.backward_results[i], dim=0)
            message = self.p_zs[i] + past_message
            return message

    def sample(self, mixture) -> torch.Tensor:
        # sample i times
        results = torch.full((2, self.K, len(mixture)),
                            fill_value=-1, dtype=torch.long)
        for i in range(len(mixture)):
            # print(f"marginal_results shape: {self.marginal_results[0][i].shape}")
            # print(f"marginal_results shape: {self.marginal_results[0][:,i].shape}")
            # print(f"first sample shape is: {torch.distributions.Categorical(logits=self.marginal_results[0]).sample().shape}")
            # print(f"second sample shape is: {torch.distributions.Categorical(logits=self.marginal_results[0]).sample().shape}")
            s0 = torch.distributions.Categorical(logits=self.marginal_results[0][:, i]).sample()
            s1 = torch.distributions.Categorical(logits=self.marginal_results[1][:, i]).sample()
            print(s0,s1)
            results[0,:,i] = s0
            results[1,:,i] = s1
        results = results.long()
        print(f"results: {results}")
        return results
        

        
    
    def separate(self, mixture: torch.Tensor):
        self.forward_results = []
        self.backward_results: List[Optional[torch.Tensor]] = [None for _ in range(self.num_sources)]
        self.marginal_results = []
        for i in range(self.num_sources):
            forward = self.forward_pass(i, mixture)
            self.forward_results.append(forward)
        
        backward_range = [i for i in range(self.num_sources-2, -2, -1)]
        print(f"Initializing backward pass on the sequence: {backward_range}")
        for i in backward_range: 
            backward = self.backward_pass(i, mixture)
            self.backward_results[i+1] = backward
            print([f"Full: {elem.shape}" if elem is not None else "Empty" for elem in self.backward_results])
        
        for i in range(self.num_sources):
            marginal = self.compute_marginals(i)
            self.marginal_results.append(marginal)

        # marginals = torch.stack(self.marginal_results)[:, :, mixture].long()
        marginals = torch.stack(self.marginal_results)
        # This should have shape 2,256,256
        print(f"marginal shape: {marginals.shape}")
        # This should have shape 2,256,49
        result = self.sample(mixture)
        return result




