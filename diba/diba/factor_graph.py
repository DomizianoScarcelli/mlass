from pathlib import Path
from typing import List, Union
import torch

from lass_mnist.lass.diba_interaces import UnconditionedTransformerPrior
from transformers import GPT2LMHeadModel, GPT2Config


class Variable:
    def __init__(self, classIdx: Union[int, None], idx: int, mixture: bool = False):
        self.classIdx = classIdx
        self.idx = idx
        self.mixture = mixture
        self.neigh_factors: List[Factor] = []


class Factor:
    def __init__(self, type: str, idx: int, connected_vars: List[Variable], value: torch.Tensor):
        self.type = type
        self.idx = idx
        self.connected_vars = connected_vars
        self.value = value


class Message:
    def __init__(self, factor: Factor, variable: Variable, value: torch.Tensor):
        self.factor = factor
        self.variable = variable
        self.value = value


class FactorGraph:
    def __init__(self, num_classes: int, mixture: torch.Tensor) -> None:
        """
        Initialize the factor graph
        @params num_classes: number of sources to separate 
        @params mixture: the latent code of the mixture to separate
        """
        self.num_classes = num_classes  # 'k' in the equations
        self.mixture = mixture  # 'm' in the equations
        self.mixture_length = len(mixture)  # 'n' in the equations
        self.num_latent_codes = 256  # 256 in the case of MNIST

        # initialize the transformer
        transformer_config = GPT2Config(
            vocab_size=self.num_latent_codes,
            n_positions=self.mixture_length,
            n_embd=128,
            n_layer=3,
            n_head=2,
            use_cache=False,
            bos_token_id=0,
            eos_token_id=511,)

        self.transformer = GPT2LMHeadModel(
            config=transformer_config)

        # list of the autoregressive priors
        self.priors = [UnconditionedTransformerPrior(
            transformer=self.transformer, sos=0) for _ in range(self.num_classes)]

        ##########################
        # Initialize the factors #
        ##########################

        # list of dictionaries, each dictionary is a factor
        self.factors: List[Factor] = []

        # add the marginal distribution of the mixture
        # p(m_1), p(m_2), ..., p(m_n)
        for i in range(self.mixture_length):
            factor = Factor("mixture_marginal",
                            idx=i,
                            connected_vars=[
                                Variable(classIdx=None, idx=i, mixture=True),
                            ],
                            value=torch.zeros((self.num_latent_codes, )))
            self.factors.append(factor)

        # add the marginal distribution of the sources
        # p(z^j_1), p(z^j_2), ..., p(z^j_n)
        for j in range(self.num_classes):
            for i in range(self.mixture_length):
                factor = Factor(type="source_marginal",
                                idx=i,
                                connected_vars=[
                                    Variable(classIdx=j, idx=i),
                                ],
                                value=torch.zeros((self.num_latent_codes, )))
                self.factors.append(factor)
        # add the autoregressive priors
        # p(z^1_i | z^1_{i-1}), p(z^2_i | z^2_{i-1}), ..., p(z^k_i | z^k_{i-1})
        for j in range(self.num_classes):
            for i in range(self.mixture_length):
                # the first prior is null
                if i > 0:
                    factor = Factor(type="prior",
                                    idx=i,
                                    connected_vars=[Variable(classIdx=j, idx=i),
                                                    Variable(classIdx=j, idx=i-1)],
                                    value=torch.zeros((self.num_latent_codes, )))
                self.factors.append(factor)

        # add the posterior factors
        # p(z^j_i | m_i)
        for j in range(self.num_classes):
            for i in range(self.mixture_length):
                factor = Factor(type="posterior",
                                idx=i,
                                connected_vars=[
                                    Variable(classIdx=j, idx=i),
                                    Variable(classIdx=None, idx=i, mixture=True)],
                                value=torch.zeros((self.num_latent_codes, )))
                self.factors.append(factor)

        # add the likelihood factors, which don't depend on the indices
        # p(m| z^1, z^2, ..., z^k)
        likelihood_factor = Factor(type="likelihood",
                                   idx=0,
                                   connected_vars=[
                                       Variable(classIdx=None,
                                                idx=None, mixture=True),
                                       *[Variable(classIdx=j, idx=None) for j in range(self.num_classes)]],
                                   value=torch.zeros(tuple(self.num_latent_codes for _ in range(self.num_classes))))
        self.factors.append(likelihood_factor)

        ###########################
        # Initialize the messages #
        ###########################

        # factor->variables messages list
        self.msg_fv: List[Message] = []
        # variables->factor messages list
        self.msg_vf: List[Message] = []

        for factor in self.factors:
            for var in factor.connected_vars:
                message = Message(
                    factor=factor, variable=var, value=factor.value)
                self.msg_fv.append(message)
                self.msg_vf.append(message)
                var.neigh_factors.append(factor)

    def belief_propagation(self):
        """
        Run the belief propagation algorithm to compute the marginals
        """
        pass


if __name__ == "__main__":
    # TODO: experimenting
    factor_graph = FactorGraph(
        num_classes=2, mixture=torch.randn((1, 256)))
    print(factor_graph.priors)

    logits_0, past = factor_graph.priors[0].get_logits(
        token_ids=torch.ones((256, 1), dtype=torch.long), past_key_values=None)

    print(factor_graph.msg_fv)
