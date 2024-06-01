import torch
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from ...lass_mnist.lass.diba_interaces import UnconditionedTransformerPrior
from typing import List
from .utils import normalize_logits
import torch.nn.functional as F

class DirectedGraphicalModel:
    """
    Represents the Bayesian Network with the latent codes z_1,...,z_n and the
    mixtures m_1,...,m_n at all the stages i.
    """

    def __init__(self, zs: torch.Tensor, ms: torch.Tensor, priors: List[UnconditionedTransformerPrior], num_sources:int):
        # Defining the strucure of the model
        K = 256 #possible discrete values in the latent codes
        edges = [ ('z0', 'm1'), ('z1', 'm1')]
        for i in range(1, num_sources+1):
            edges.append((f"m{i}", f"m{i+1}"))
            edges.append((f"z{i+1}", f"m{i+1}"))
        self.model = BayesianNetwork(edges)
        past = torch.LongTensor(torch.zeros((num_sources, K, 1)))

        cpds_z: List[TabularCPD] = []
        cpds_m: List[TabularCPD] = []
        for i in range(num_sources+1):
            log_prior, _ = priors[i].get_logits(
                token_ids=past,
                past_key_values=None,
            )
            log_prior = normalize_logits(log_prior)

            cpd = TabularCPD(variable=f"z{i}", variable_card=K, values=F.softmax(log_prior)) #TODO: maybe numerical instability?
            cpds_z.append(cpd)

        for i in range(1, num_sources+1):
            #TODO: discover how to know the values
            if i == 1:
                cpd = TabularCPD(variable=f"m{i}", variable_card=K, values=None)
            pass






    def forward_pass(self, m_i: torch.Tensor, z_i: torch.Tensor):
        """
        It computes the message μα in the graphical model forward pass.
        """
        raise NotImplementedError()

    def backward_pass(self, m_i):
        """
        It computes the message μβ in the graphical model backward pass.
        """
        raise NotImplementedError()
