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


class Factor:
    def __init__(self, factor_type: str, idx: int, connected_vars: List[Variable], value: torch.Tensor):
        self.factor_type = factor_type
        self.idx = idx
        self.connected_vars = connected_vars
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

        # list of dictionaries, each dictionary is a factor
        self.factors: List = []

        ##########################
        # Initialize the factors #
        ##########################

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
                factor = Factor("source_marginal",
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
                    factor = Factor("prior",
                                    idx=i,
                                    connected_vars=[Variable(classIdx=j, idx=i),
                                                    Variable(classIdx=j, idx=i-1)],
                                    value=torch.zeros((self.num_latent_codes, )))
                self.factors.append(factor)

        # add the posterior factors
        # p(z^j_i | m_i)
        for j in range(self.num_classes):
            for i in range(self.mixture_length):
                factor = Factor("posterior",
                                idx=i,
                                connected_vars=[
                                    Variable(classIdx=j, idx=i),
                                    Variable(classIdx=None, idx=i, mixture=True)],
                                value=torch.zeros((self.num_latent_codes, )))
                self.factors.append(factor)

        # add the likelihood factors, which don't depend on the indices
        # p(m| z^1, z^2, ..., z^k)
        likelihood_factor = Factor("likelihood",
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

        self.msg_fv = {}  # factor->variables messages (dictionary)
        self.msg_vf = {}  # variables->factor messages (dictionary)
        # neighboring factors of variables (list of list)
        self.ne_var = [[] for i in range(self.mixture_length)]

        for index, factor in enumerate(self.factors):
            factor_type = factor['type']
            variable = factor['var']

            # marginal factors (unary)
            if factor_type == 'source_marginal' or factor_type == "mixture_marginal":
                # I just connect the factor to the variable
                factor_to = (index, variable)
                self.msg_fv[factor_to] = torch.zeros(
                    (1, self.num_latent_codes))
            # prior factors (binary)
            elif factor_type == "prior":
                # i first connect the variable to the factor
                # (z^j_i, f_i)
                from_variable = (variable, index)
                # then i connect the factor to the other variable in the factor
                # (f_i, z^j_{i-1}, )
                to_factor = (index, variable)
                pass
            # likelihood factor (n-ary)
            elif factor_type == "likelihood":
                pass

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
