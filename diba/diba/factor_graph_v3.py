from __future__ import annotations
from enum import Enum
import time
from typing import Dict, List, Set, Tuple, NamedTuple, Union
import torch
from tqdm import tqdm
from diba.diba.utils import normalize_logits

from lass_mnist.lass.diba_interaces import DenseLikelihood, UnconditionedTransformerPrior
from transformers import GPT2LMHeadModel, GPT2Config
import logging

"""
For each timestep_t the joint posterior distribution is computed from the likelihood and the prior. The prior is computed using the autoregressive
priors conditioned by the current sampled tokens. The likelihood is sliced conditioned to the current observed mixture token.

The factor graph is constructed where the factors are the joint posterior distribution, and belief propagation is used in order
to find the marginal posterior distribution for each source.

Finally a sample for each source is drawn from the marginal posterior distribution.

This goes on until the end of the mixture is reached (timestep_t is equal to the length of the mixture).
"""

log = logging.getLogger("logger")
DEBUG = True

if DEBUG:
    logging.basicConfig(filename='factor_graph.log', level=logging.DEBUG,
                        format='%(name)s - %(levelname)s - %(message)s',
                        filemode='w')


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        log.debug(f"{func.__name__} took {end_time - start_time} seconds.")
        return result
    return wrapper


def _check_if_nan_or_inf(tensor: torch.Tensor) -> bool:
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


#########################################
# Labelled Tensor
#########################################

class LabeledTensor(NamedTuple):
    # Source: https://jessicastringham.net/2019/01/09/sum-product-message-passing/
    tensor: torch.Tensor
    axes_labels: Tuple[str, ...]


def name_to_axis_mapping(labeled_tensor: LabeledTensor):
    return {
        name: axis
        for axis, name in enumerate(labeled_tensor.axes_labels)
    }


def other_axes_from_labeled_axes(labeled_tensor: LabeledTensor, axis_label: str):
    # returns the indexes of the axes that are not axis label
    return tuple(
        axis
        for axis, name in enumerate(labeled_tensor.axes_labels)
        if name != axis_label
    )


def is_conditional_prob(labeled_tensor: LabeledTensor, var_name: str):
    '''
    labeled_array (LabeledArray)
    variable (str): name of variable, i.e. 'a' in p(a|b)
    '''
    summation = torch.sum(
        labeled_tensor.tensor,
        dim=name_to_axis_mapping(labeled_tensor)[var_name]
    )
    return torch.all(torch.isclose(torch.sum(
        labeled_tensor.tensor,
        dim=name_to_axis_mapping(labeled_tensor)[var_name]
    ), torch.tensor(1.0))).item()


def is_joint_prob(labeled_tensor: LabeledTensor):
    return torch.all(torch.isclose(torch.sum(labeled_tensor.tensor), torch.tensor(1.0))).item()

#########################################
# Nodes (Variables and PosteriorFactors)
#########################################


class Variable:
    _variables = {}

    def __new__(cls, class_idx: Union[int, None], idx: int):
        key = (class_idx, idx, mixture)
        if key in Variable._variables:
            return Variable._variables[key]
        else:
            instance = super(Variable, cls).__new__(cls)
            instance.class_idx = class_idx
            instance.neighbors: List[PosteriorFactor] = []
            instance.idx = idx
            Variable._variables[key] = instance
            return instance

    def __init__(self, class_idx: Union[int, None], idx: int):
        # Initialize only if it's a new instance
        self.neighbors: List[PosteriorFactor] = []
        self.class_idx = class_idx
        self.idx = idx

        # TODO: do not hardcode this
        self.marginal = torch.full((256,), (1 / 256))

    def __hash__(self):
        return hash((self.class_idx, self.idx))

    def __repr__(self):
        if self.class_idx is None:
            return f"m_{self.idx}"
        return f"z^{self.class_idx}_{self.idx}"

    def add_neighbor(self, neighbor: PosteriorFactor):
        assert self.is_valid_neighbor(neighbor)
        self.neighbors.append(neighbor)

    def is_valid_neighbor(self, neighbor: PosteriorFactor) -> bool:
        return isinstance(neighbor, PosteriorFactor)


class PosteriorFactor():
    def __init__(self, idx: int):
        self.idx = idx
        self.neighbors: List[Variable] = []
        self.value: torch.Tensor = None

    def add_neighbor(self, neighbor: Variable):
        assert self.is_valid_neighbor(neighbor)
        self.neighbors.append(neighbor)

    def is_valid_neighbor(self, variable: Variable):
        # PosteriorFactors can only neighbor Variables
        return isinstance(variable, Variable)

    def __str__(self):
        return f"PosteriorFactor, neighbors: {self.neighbors}"

    def __repr__(self):
        return f"PosteriorFactor, neighbors: {self.neighbors}"


#########################################
# PosteriorFactor Graph
#########################################

class PosteriorFactorGraph:
    def __init__(self, num_sources: int, mixture_t: int, likelihood: DenseLikelihood, transformer: GPT2LMHeadModel, past_z: Tuple[List[Variable]]):
        '''
        num_sources (int): number of sources
        mixture (torch.Tensor): shape (mixture_length)
        likelihood (DenseLikelihood): likelihood function
        '''
        self.num_sources = num_sources
        self.mixture_t = mixture_t
        self.likelihood = likelihood
        self.log_posterior: torch.Tensor = None
        self.past_z = past_z
        self.num_latent_variables = likelihood.get_tokens_count()
        self._factors: Dict[str, PosteriorFactor] = {}
        self._variables: Dict[str, Variable] = {}
        self.transformer = transformer
        self.marginal_posteriors: torch.Tensor = None
        UNCOND_BOS = 0
        self.priors = [UnconditionedTransformerPrior(
            transformer=self.transformer, sos=UNCOND_BOS) for _ in range(self.num_sources)]
        self._create_graph()
        self.msg_vf: Dict[Variable, Dict[PosteriorFactor, int]] = {}
        self.msg_fv: Dict[PosteriorFactor, Dict[Variable, int]] = {}
        self._init_messages()

    def _create_graph(self) -> Tuple[List[PosteriorFactor], List[Variable]]:
        '''
        Returns:
            factors (List[PosteriorFactor]): list of factors
            variables (List[Variable]): list of variables
        '''
        # Create variables
        for j in range(self.num_sources):
            for i in tqdm(range(self.num_latent_variables), desc="Creating source variables"):
                variable = Variable(class_idx=j, idx=i)
                self._variables[variable] = variable

        for i in tqdm(range(self.num_latent_variables), desc="Creating mixture variables"):
            variable = Variable(class_idx=None, idx=i)
            self._variables[variable] = variable

        # Compute the prior and the posterior
        self._compute_posterior()

        # Create factors
        for i in tqdm(range(self.num_latent_variables), desc="Creating factors"):
            # Create posterior factors #p(z^1_i, z^2_i | z^1_{s<i}, z^2_{s<i}, m_i)
            posterior = PosteriorFactor(idx=i)
            posterior.value = self.log_posterior[:, i]

            assert posterior.value.shape == torch.Size(
                [self.num_latent_variables])

            variables = []
            # for var in self.past_z[j]:
            # variables.append(var)
            for j in range(self.num_sources):
                variables.append(Variable(class_idx=j, idx=i))

            variable: Variable
            for variable in variables:
                posterior.add_neighbor(variable)
                variable.add_neighbor(posterior)

            self._factors[posterior] = posterior

    def _compute_posterior(self):
        dense_log_likelihood = self.likelihood.get_dense_log_likelihood().permute(2, 0, 1)
        sliced_likelihood = dense_log_likelihood[self.mixture_t]
        log_priors = torch.empty((self.num_sources, self.num_latent_variables))
        for j in range(self.num_sources):
            prior = self.priors[j]
            past = torch.tensor(
                [var.idx for var in self.past_z[j]]).view(1, -1).to(torch.long)

            log.debug(f"Past shape is {past.shape}")
            log_prior, _ = prior.get_logits(
                token_ids=past,
                past_key_values=None,
            )

            log_prior = normalize_logits(log_prior).squeeze(0)

            log_priors[j] = log_prior

        log_posterior = sliced_likelihood + log_prior[0] + log_prior[1]
        self.log_posterior = log_posterior

    def _init_messages(self):
        factor: PosteriorFactor
        for factor in self._factors:
            self.msg_fv[factor] = {}
            for variable in factor.neighbors:
                self.msg_fv[factor][variable] = 0
        variable: Variable
        for variable in self._variables:
            self.msg_vf[variable] = {}
            for factor in variable.neighbors:
                self.msg_vf[variable][factor] = 0

    def _get_other_indexes(self, variable: Variable, factor: PosteriorFactor) -> Tuple[int]:
        return tuple(var.idx for var in factor.neighbors if var != variable)

    def belief_propagation(self):
        # Update factor-to-variable messages
        for factor, msg_to_v in self.msg_fv.items():
            incoming_messages = []
            for variable in factor.neighbors:
                incoming_messages.extend(
                    [self.msg_vf[var][factor] for var in msg_to_v if var != variable])
                print(
                    f"Incomign message with factor {factor} is {incoming_messages}")
                exp_sum_incoming_messages = torch.exp(torch.sum(
                    torch.tensor(incoming_messages), dim=0))

                updated_message = factor.value * \
                    exp_sum_incoming_messages

                msg_to_v[variable] = updated_message

                print(f"Updated message is {updated_message}")
                raise NotImplementedError

        pass


#########################################
# Main code
#########################################

if __name__ == "__main__":
    #############
    # Main code #
    #############
    mixture = torch.tensor([210, 135, 8, 202, 135, 8, 56, 39, 63, 168, 149, 119, 70, 56, 137, 149, 93, 217, 217, 217, 8, 210, 66,
                            254, 26, 9, 168, 135, 210, 29, 26, 88, 222, 75, 210, 56, 56, 88, 4, 34, 80, 8, 56, 137, 75, 7, 41, 8, 56])
    # MIXTURE_LENGTH = 49

    SUMS_CHECKPOINT_PATH = "lass_mnist/checkpoints/sum/256-sigmoid-big.pt"
    with open(SUMS_CHECKPOINT_PATH, 'rb') as f:
        sums = torch.load(f, map_location=torch.device('cpu'))

    transformer_config = GPT2Config(
        vocab_size=256,
        n_positions=len(mixture),
        n_embd=128,
        n_layer=3,
        n_head=2,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=511,)

    transformer = GPT2LMHeadModel(
        config=transformer_config)

    AUTOREGRESSIVE_CHECKPOINT_PATH = "lass_mnist/checkpoints/unconditioned/256-sigmoid-big.pt"
    with open(AUTOREGRESSIVE_CHECKPOINT_PATH, 'rb') as f:
        transformer.load_state_dict(
            torch.load(f, map_location=torch.device('cpu')))

    transformer.eval()

    likelihood = DenseLikelihood(sums=sums)

    mixture_length = len(mixture)

    past = ([Variable(class_idx=0, idx=0)],
            [Variable(class_idx=1, idx=0)])

    for t in range(mixture_length):
        factor_graph = PosteriorFactorGraph(
            num_sources=2, mixture_t=mixture[t], likelihood=likelihood, transformer=transformer, past_z=past)

        factor_graph.belief_propagation()

        print(
            f"Number of factor-to-variable messages: {len(factor_graph.msg_fv)}")
        print(
            f"Number of variable-to-factor messages: {len(factor_graph.msg_vf)}")

        raise Exception("STOP")

    # sources = factor_graph.separate()

    # log.debug(f"The sampled sources are: {sources}")
