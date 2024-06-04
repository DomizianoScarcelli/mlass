
import torch
from lass_mnist.lass.diba_interaces import UnconditionedTransformerPrior
from typing import List
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
        K = 256 #possible discrete values in the latent codes
        past = torch.zeros((num_sources, K, 1)).long()
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
                token_ids=past,
                past_key_values=None,
            )
            log_prior = normalize_logits(log_prior).squeeze()
            print(f"log_prior {i} has shape {log_prior.shape}")
            self.p_zs.append(F.softmax(log_prior))
        
        # p_mmzs[i] = p(m_i | m{i-1}, z_i)
        p_mmzs_path = "./lass_mnist/models/sums-MNIST-gm/best.pt"
        with open(p_mmzs_path, "rb") as f:
            self.p_mmzs = torch.load(f)
            print(f"p_mmzs has shape: {self.p_mmzs.shape}")



    def forward_pass(self, i: int, mixture: torch.Tensor):
        """
        It computes the message μα in the graphical model forward pass.
        """
        # shape is [49, 256, 256] = [mixture_length, K, K]
        curr_p_mmz = self.p_mmzs[i][mixture]
        # shape is [2, 256] = [num_sources, K]
        curr_p_z = self.p_zs[i]
        # print(f"curr_p_mzz shape is: {curr_p_mmz.shape}")
        # print(f"curr_p_z shape is: {curr_p_z.shape}")
        if i == 0:
            # shape before sum is [mixture_length, K, num_sources]
            # shape after sum is [mixture_length, num_sources]
            message = torch.sum(curr_p_mmz @ curr_p_z.T, dim=1)
            print(f"message shape is: {message.shape}")
            return message
        new_message = torch.sum(curr_p_mmz @ curr_p_z.T, dim=1)
        old_message = torch.sum(self.p_zs[i-1].T @ self.forward_pass(i-1, mixture).T, dim=0)
        print(f"old message shape is: {old_message.shape}")
        final_message = new_message * old_message.unsqueeze(1)
        print(f"final message shape is {final_message.shape}")
        return final_message

    def backward_pass(self, i: int, mixture: torch.Tensor):
        """
        It computes the message μβ in the graphical model backward pass.
        """
        raise NotImplementedError()
    
    def separate(self, mixture: torch.Tensor):
        forward_results = []
        backward_results = []
        for i in range(self.num_sources):
            forward_results.append(self.forward_pass(i, mixture))
        for i in range(self.num_sources, 0, -1):
            backward_results.append(self.backward_pass(i, mixture))

        #TODO: compute marginals
        raise NotImplementedError()



