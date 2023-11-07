from enum import Enum
import time
from typing import Dict, List, NamedTuple, Set, Tuple, Union
import numpy as np
import torch
from tqdm import tqdm

from lass_mnist.lass.diba_interaces import DenseLikelihood, SparseLikelihood, UnconditionedTransformerPrior
from transformers import GPT2LMHeadModel, GPT2Config

import logging


"""
### Sum-product algorithm
**Inputs:**
* `num_vars`, `num_states`, `factors`, `msg_fv`, `msg_vf`, `ne_var`

**Outputs:**
* `marginals`: `num_vars` x `num_states` array of estimated max-marginals
* `est`: array comprising the estimated state of each variable

Note: $\chi_f$ are the variables that are involved for the factor $f$

**Algorithm Pseudocode:**
* For `N` iterations do:
 * Update all unary factor-to-variable messages: $\lambda_{f\rightarrow x}(x) = f(x)$
 * Update all pairwise factor-to-variable messages: $$\lambda_{f \rightarrow x}(x)=\log \left(\sum_{\mathcal{X}_f \backslash x} f\left(\mathcal{X}_f\right) \exp \left[\sum_{y \in\{n e(f) \backslash x\}} \lambda_{y \rightarrow f}(y)\right]\right)$$
 * Update all variable-to-factor messages: $\lambda_{x\rightarrow f}(x) = \sum_{g\in\{ ne(x)\setminus f\}}\lambda_{g\rightarrow x}(x)$
            
* Calculate Marginals: $\gamma_x(x) = \sum_{g\in\{ ne(x)\}}\lambda_{g\rightarrow x}(x)$
"""

log = logging.getLogger("logger")
DEBUG = True
torch.set_printoptions(precision=2, sci_mode=False)

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


def is_conditional_prob(tensor: torch.Tensor, dim: int):
    return torch.all(torch.isclose(torch.sum(tensor, dim=dim), torch.tensor(1.0))).item()


def is_joint_prob(tensor: torch.Tensor):
    return torch.all(torch.isclose(torch.sum(tensor), torch.tensor(1.0))).item()

#########################################
# Factors (and Variables)
#########################################


class FactorType(Enum):
    PRIOR = "prior"
    POSTERIOR = "posterior"
    LIKELIHOOD = "likelihood"
    MARGINAL_POSTERIOR = "marginal_posterior"
    MARGINAL = "marginal"


class NewFactor:
    _factors = {}

    def __new__(cls, type: str, value: torch.Tensor):
        key = (type, value)
        if key in NewFactor._factors:
            return NewFactor._factors[key]
        else:
            instance = super(NewFactor, cls).__new__(cls)
            instance.type = type
            instance.value = value
            NewFactor._factors[key] = instance
            return instance

    def __init__(self, type: str, value: torch.Tensor):
        # Initialize only if it's a new instance
        self.type = type
        self.value = value

    def __hash__(self):
        return hash((self.type))

    def __repr__(self):
        return f"NewFactor(type={self.type}, value_shape={self.value.shape})"

#########################################
# Factor Graph
#########################################


class FactorGraph:
    @timeit
    def __init__(self, num_sources: int, mixture_t: torch.Tensor, likelihood: DenseLikelihood, transformer: GPT2LMHeadModel, past_z: torch.Tensor) -> None:
        """
        Initialize the factor graph
        @params num_classes: number of sources to separate 
        @params mixture: the latent code of the mixture to separate
        """
        self.num_sources = num_sources  # 'k' in the equations
        self.mixture_t = mixture_t  # 'm' in the equations

        # always None in the case of uncoditional priors
        self.past_z = past_z
        self.likelihood = likelihood
        self.num_latent_codes = self.likelihood.get_tokens_count()  # 256 in the case of MNIST
        # This should have shape (num_sources + 1, num_latent_codes)
        self.marginal_posteriors = torch.empty(
            (self.num_sources, self.num_latent_codes))
        self.msg_fv: Dict[NewFactor, torch.Tensor] = {}
        self.msg_vf: Dict[NewFactor, torch.Tensor] = {}

        self.marginal_posterior = torch.empty(
            size=(self.num_sources, self.num_latent_codes))

        self.transformer = transformer

        # list of the autoregressive priors
        UNCOND_BOS = 0
        self.priors = [UnconditionedTransformerPrior(
            transformer=self.transformer, sos=UNCOND_BOS) for _ in range(self.num_sources)]

        ##########################
        # Initialize the factors #
        ##########################

        self.factors: List[NewFactor] = []

        posterior_value = self._compute_posterior()
        self.factors.append(NewFactor(type=FactorType.POSTERIOR,
                                      value=posterior_value))

    def __repr__(self):
        return f"""FactorGraph(
            num_sources={self.num_sources},
            num_latent_codes={self.num_latent_codes},
            factors={self.factors}"""

    def _compute_posterior(self):
        # dense_log_likelihood = self.likelihood.get_dense_log_likelihood().permute(2, 0, 1)
        # sliced_likelihood = dense_log_likelihood[self.mixture_t]
        sparse_likelihood = self.likelihood.get_log_likelihood(self.mixture_t)
        log_priors = torch.empty(
            (self.num_sources, self.num_latent_codes))

        for j in range(self.num_sources):
            prior = self.priors[j]
            past = self.past_z[j].view(1, -1).to(torch.long)

            log.debug(f"Past shape is {past.shape}")
            log_prior, _ = prior.get_logits(
                token_ids=past,
                past_key_values=None,
            )

            log_prior = log_prior.squeeze(0)

            # log_prior = normalize_logits(log_prior).squeeze(0)

            log_priors[j] = log_prior

        sparse_log_priors = log_priors.to_sparse()

        sparse_log_posterior = sparse_likelihood + \
            sparse_log_priors[0] + sparse_log_priors[1] + sparse_log_priors[2]

        return sparse_log_posterior

    def _initialize_messages(self):
        for factor in self.factors:
            # since messages from extremal node factors are initialized to factor and
            # messages from other factors are initialized to 1
            # this means that:
            #   the shape of mgs_vf is equal to the shape of the variable
            #   the shape of msg_fv is equal to the shape of the factor

            log.debug(f"Factor value is {factor.value}")

            fv_shape = (self.num_sources, *factor.value.shape)

            vf_shape = (self.num_sources, self.num_latent_codes)

            # shape of the variable state space for each variable
            self.msg_fv[factor] = torch.zeros(size=fv_shape)
            self.msg_vf[factor] = torch.zeros(
                size=vf_shape)

    def _get_other_indexes(self, index: int) -> Tuple[int]:
        return tuple(i for i in range(self.num_sources) if i != index)

    def _exclude_row(self, tensor: torch.Tensor, row: int) -> torch.Tensor:
        return torch.cat((tensor[:row], tensor[row+1:]), dim=0)

    def _get_marginal_posterior(self):
        for i in range(self.num_sources):
            incoming_messages = torch.cat(
                [self.msg_fv[fac] for fac in self.factors], dim=0)

            log.debug(f"Incoming message shape is {incoming_messages.shape}")
            sum_incoming_message = torch.sum(
                incoming_messages, dim=0)

            self.marginal_posterior[i] = sum_incoming_message

    @timeit
    def belief_propagation(self, iterations: int = 30):
        """
        Run the sum-product algorithm for the given number of iterations in order to compute the marginal distributions.
        """
        # The priors are updated only at the start of the belief propagation
        self._initialize_messages()

        # for it in tqdm(range(iterations), desc="Belief Propagation"):
        for it in range(iterations):
            # Update all factor-to-variable messages
            for factor, message in self.msg_fv.items():
                for i in range(self.num_sources):
                    log.debug(
                        f"The shape of the factor is {factor.value.shape}")

                    incoming_messages = self.msg_vf[factor]

                    updated_message = torch.sum(factor.value * torch.exp(
                        torch.sum(self._exclude_row(
                            incoming_messages, i), dim=0)
                    ), dim=self._get_other_indexes(i))

                    updated_message = torch.log(
                        updated_message)

                    self.msg_fv[factor][i] = updated_message

                    assert self.msg_fv[factor].shape == message.shape, f"""
                    Assert error for factor {factor} during factor-to-variable message update
                    The shapes between the original message and the updated message are not compatible
                    Original message shape: {message.shape}
                    Updated message shape: {self.msg_fv[factor].shape}
                    """

            # Update all variable-to-factor messages
            # NOTE: I just have to take the sum of the incoming messages, except for the one that is coming from the factor itself
            for factor, message in self.msg_vf.items():
                for i in range(self.num_sources):
                    incoming_messages = []
                    for fact, message in self.msg_fv.items():
                        if fact != factor:
                            incoming_messages.append(message)

                    if incoming_messages == []:
                        # If there are no incoming messages, then the message is initialized to the factor value
                        self.msg_vf[factor][i] = torch.zeros(
                            size=(256,))
                        continue

        # Compute the marginals for all the variables
        # self._update_marginals()
        self._get_marginal_posterior()

        return

    def sample(self):
        """
        Naive marginalizaiton via summing
        """
        joint_posterior_factor = self.factors[0].value.squeeze(0)
        log.debug(
            f"Joint posterior factor value shape is {joint_posterior_factor.shape}")
        for j in range(self.num_sources):
            joint_posterior_sum = torch.softmax(torch.sum(torch.softmax(
                joint_posterior_factor, dim=j), dim=self._get_other_indexes(j)), dim=0)

            log.debug(f"Joint posterior sum is {joint_posterior_sum}")

            log.debug(
                f"Joint posterior sum on index {self._get_other_indexes(j)} shape is {joint_posterior_sum.shape}")

            self.marginal_posteriors[j] = joint_posterior_sum

            log.debug(f"Marginal posterior are: {self.marginal_posterior[j]}")

        log.debug(
            f"Marginal posterior shape is {self.marginal_posterior.shape}")
        samples = torch.multinomial(
            self.marginal_posteriors, num_samples=1, replacement=True)
        return samples


def separate(mixture: torch.Tensor, likelihood: DenseLikelihood, transformer: GPT2LMHeadModel, sources: int):
    past = torch.tensor([[0], [0], [0]])
    mixture_length = len(mixture)
    all_samples = past.detach().clone()
    for t in range(mixture_length - 1):
        factor_graph = FactorGraph(
            num_sources=sources, mixture_t=mixture[t], likelihood=likelihood, transformer=transformer, past_z=past)
        samples = factor_graph.sample()
        log.debug(
            f"Samples shape is {samples.shape}, while all_samples shape is {all_samples.shape}")
        all_samples = torch.cat((all_samples, samples), dim=1)
        log.debug(f"Past at time step {t} is {past}")
        past = all_samples
    return all_samples


if __name__ == "__main__":
    #############
    # Main code #
    #############
    mixture = torch.tensor([210, 135, 8, 202, 135, 8, 56, 39, 63, 168, 149, 119, 70, 56, 137, 149, 93, 217, 217, 217, 8, 210, 66,
                            254, 26, 9, 168, 135, 210, 29, 26, 88, 222, 75, 210, 56, 56, 88, 4, 34, 80, 8, 56, 137, 75, 7, 41, 8, 56])
    # MIXTURE_LENGTH = 49

    SUMS_CHECKPOINT_PATH = "lass_mnist/checkpoints/sum/sums_epoch_368.pt"
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

    likelihood = SparseLikelihood(sums=sums)

    all_samples = separate(mixture=mixture, likelihood=likelihood,
                           transformer=transformer, sources=3)

    log.debug(f"All samples are {all_samples} with shape {all_samples.shape}")
