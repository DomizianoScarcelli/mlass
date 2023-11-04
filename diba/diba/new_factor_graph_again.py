from enum import Enum
import time
from typing import Dict, List, NamedTuple, Set, Tuple, Union
import torch
from tqdm import tqdm
from diba.diba.utils import normalize_logits

from lass_mnist.lass.diba_interaces import DenseLikelihood, UnconditionedTransformerPrior
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
    def __init__(self, num_sources: int, mixture: torch.Tensor, likelihood: DenseLikelihood, transformer: GPT2LMHeadModel) -> None:
        """
        Initialize the factor graph
        @params num_classes: number of sources to separate 
        @params mixture: the latent code of the mixture to separate
        """
        self.num_sources = num_sources  # 'k' in the equations
        self.mixture = mixture  # 'm' in the equations
        self.mixture_length = len(mixture)  # 'n' in the equations

        # always None in the case of uncoditional priors
        self.pasts = [torch.tensor([0]).view(-1, 1)
                      for _ in range(self.num_sources)]
        self.likelihood = likelihood
        self.num_latent_codes = self.likelihood.get_tokens_count()  # 256 in the case of MNIST
        # This should have shape (num_sources + 1, num_latent_codes)
        self.marginals: torch.Tensor = torch.empty(size=(self.num_sources + 1,
                                                         self.num_latent_codes))
        self.msg_fv: Dict[NewFactor, torch.Tensor] = {}
        self.msg_vf: Dict[NewFactor, torch.Tensor] = {}

        log.debug(f"Length of the mixture is: {self.mixture_length}")

        self.transformer = transformer

        # list of the autoregressive priors
        UNCOND_BOS = 0
        self.priors = [UnconditionedTransformerPrior(
            transformer=self.transformer, sos=UNCOND_BOS) for _ in range(self.num_sources)]

        ##########################
        # Initialize the factors #
        ##########################

        self.factors: List[NewFactor] = []

        prior_value = self._init_prior_values()
        assert is_conditional_prob(prior_value, dim=1)
        assert not is_joint_prob(prior_value)

        self.factors.append(NewFactor(type=FactorType.PRIOR,
                                      value=prior_value))

        # Likelihood factors P(m| z^1, z^2, ..., z^k): tensor[m, z^1, z^2, ..., z^k]
        likelihood_value = torch.softmax(
            self.likelihood.get_dense_log_likelihood().permute(2, 0, 1), dim=0)

        assert is_conditional_prob(likelihood_value, dim=0)
        assert not is_joint_prob(likelihood_value)

        self.factors.append(NewFactor(type=FactorType.LIKELIHOOD,
                                      value=likelihood_value))

    def __repr__(self):
        return f"""FactorGraph(
            num_sources={self.num_sources},
            mixture={self.mixture},
            mixture_length={self.mixture_length}, 
            num_latent_codes={self.num_latent_codes},
            factors={self.factors}"""

    def _init_prior_values(self):
        prior_factor = torch.empty((self.num_sources,
                                    self.num_latent_codes,
                                    self.num_latent_codes))
        pbar = tqdm(total=self.num_sources *
                    self.num_latent_codes, desc="Initializing prior values")
        for class_idx in range(self.num_sources):
            for mixture_i in range(self.num_latent_codes):
                class_prior: UnconditionedTransformerPrior = self.priors[class_idx]
                current_source = self.pasts[class_idx]
                current_source = current_source.to(torch.long)

                latest_past = mixture_i

                log.debug(f"Latest past is {latest_past}")

                log.debug(
                    f"During autoregressive prior update, the current source shape is {current_source.shape}")

                # Log priors shape should be torch.Size([1, 256])
                # Current source shape sould be torch.Size([1, n]) where n is the number of tokens samples so far
                log_priors, _ = class_prior._get_logits(
                    current_source,
                    past_key_value=None
                )

                log_priors = normalize_logits(log_priors, 1.0).squeeze(0)

                probs = torch.softmax(log_priors, dim=0)

                assert is_conditional_prob(probs, dim=0)

                log.debug(
                    f"Prior slice shape is {prior_factor[class_idx, :, latest_past].shape}, while probs shape is {probs.shape}")

                prior_factor[class_idx, :, latest_past] = probs

                pbar.update(1)

                log.debug(
                    f"Prior factor value for current source: {current_source}, classIdx: {class_idx} and latest_past: {latest_past} is {prior_factor[class_idx, :, latest_past]} and log_priors is {log_priors}")
        return prior_factor

    def _update_autoregressive_prior(self):
        torch.set_printoptions(precision=2, sci_mode=False)
        """
        Updates the value for the factor that refers to the autoregressive priors 
        p(z^j_i | z^j_{i-1}) for all the j and i.
        """
        prior_factor = [
            factor for factor in self.factors if factor.type == FactorType.PRIOR][0]

        # Prior gives me the probability distribution of the next code, for the codes that have z^j_{i-1} = self.pasts[j][-1]
        for class_idx in range(self.num_sources):
            class_prior: UnconditionedTransformerPrior = self.priors[class_idx]
            current_source = self.pasts[class_idx]
            current_source = current_source.to(torch.long)

            latest_past = current_source[:, -1].view(-1).item()

            log.debug(f"Latest past is {latest_past}")

            log.debug(
                f"During autoregressive prior update, the current source shape is {current_source.shape}")

            # Log priors shape should be torch.Size([1, 256])
            # Current source shape sould be torch.Size([1, n]) where n is the number of tokens samples so far
            log_priors, _ = class_prior._get_logits(
                current_source,
                past_key_value=None
            )

            log_priors = normalize_logits(log_priors, 1.0).squeeze(0)

            probs = torch.softmax(log_priors, dim=0)

            assert is_conditional_prob(probs, dim=0)

            log.debug(
                f"Prior slice shape is {prior_factor.value[class_idx, :, latest_past].shape}, while probs shape is {probs.shape}")

            prior_factor.value[class_idx, :, latest_past] = probs

            log.debug(
                f"Prior factor value for current source: {current_source}, classIdx: {class_idx} and latest_past: {latest_past} is {prior_factor.value[class_idx, :, latest_past]} and log_priors is {log_priors}")

    def _get_marginal_posterior(self):
        """
        Updates the value for the factor that refers to the marginal posterior
        """
        # log P(z^i | m) = log P(m | z^i) + log P(z^i) - log P(m)
        # log P(m | z^i) is the likelihood factor given the z^i, so has shape (256,256)
        # log P(z^i) is the marginal factor given the z^i, so has shape (256)
        # log P(m) is the marginal factor for the mixture, so has shape (256)
        incoming_messages = []
        for fact, message in self.msg_fv.items():
            if message.size(0) == self.num_sources + 1:
                incoming_messages.append(message[1:, :])
            else:
                incoming_messages.append(message)

        log.debug(
            f"There are {len(incoming_messages)} incoming messages, with shapes {set([message.shape for message in incoming_messages])}")

        stacked_incoming_messages = torch.stack(
            incoming_messages, dim=0)

        log.debug(
            f"Shape of stacked incoming message is {stacked_incoming_messages.shape}")

        log.debug(f"Stacked incoming message is {incoming_messages}")

        sum_stacked_incoming_messages = torch.sum(
            stacked_incoming_messages, dim=0)

        sum_stacked_incoming_messages = sum_stacked_incoming_messages / \
            torch.sum(sum_stacked_incoming_messages)

        log.debug(
            f"Shape of marginal posterior is : {sum_stacked_incoming_messages.shape}")

        log.debug(
            f"Final marginal posterior is: {sum_stacked_incoming_messages}")

        return sum_stacked_incoming_messages

    def _initialize_messages(self):
        for factor in self.factors:
            fv_shape = (self.num_sources + 1, self.num_latent_codes, self.num_latent_codes) if factor.type != FactorType.PRIOR else (
                self.num_sources, self.num_latent_codes, self.num_latent_codes)

            vf_shape = (self.num_sources + 1, *factor.value.shape) if factor.type != FactorType.PRIOR else (
                self.num_sources, *factor.value.shape)

            # shape of the variable state space for each variable
            self.msg_fv[factor] = torch.zeros(size=fv_shape)
            self.msg_vf[factor] = torch.zeros(
                size=vf_shape)

    @timeit
    def belief_propagation(self, iterations: int = 30):
        """
        Run the sum-product algorithm for the given number of iterations in order to compute the marginal distributions.
        """
        # The priors are updated only at the start of the belief propagation
        self._initialize_messages()
        self._update_autoregressive_prior()

        # for it in tqdm(range(iterations), desc="Belief Propagation"):
        for it in range(iterations):
            # Update all factor-to-variable messages
            for factor, message in self.msg_fv.items():
                if factor.type == FactorType.PRIOR:
                    for i in range(self.num_sources):
                        zi_factor_value = factor.value[i, :, :].squeeze(0)

                        log.debug(
                            f"zi factor value is {zi_factor_value}")

                        updated_message = zi_factor_value * torch.exp(
                            torch.sum(self.msg_vf[factor]
                                      [:, i, :, :], dim=0)
                        )

                        log.debug(
                            f"Prior updated message at first op is {updated_message}")

                        assert updated_message.shape == (zi_factor_value.shape), f"""
                        Assert error for factor {factor} during factor-to-variable message update
                        The updated message shape is not correct
                        Updated message shape: {updated_message.shape}
                        Factor value shape: {zi_factor_value.shape}
                        """

                        updated_message = torch.sum(
                            updated_message, dim=tuple(j for j in range(self.num_sources) if i != j))

                        log.debug(
                            f"Prior updated message before normalization is {updated_message}")

                        # TODO: don't know if I need this
                        updated_message = updated_message / \
                            torch.sum(updated_message)

                        log.debug(
                            f"Prior updated message before log is {updated_message}")
                        assert updated_message.shape == (self.num_latent_codes,), f"""
                        Assert error for factor {factor} during factor-to-variable message update
                        The updated message shape after the summation is not correct
                        Updated message shape: {updated_message.shape}, this should be {(self.num_latent_codes,)}
                        """

                        updated_message = torch.log(
                            updated_message)

                        log.debug(
                            f"Prior updated message is {updated_message}")

                        assert not _check_if_nan_or_inf(updated_message), f"""
                        Assert error for factor {factor} during factor-to-variable message update
                        The updated message contains NaN or Inf values
                        Updated message before log operation: {updated_message}
                        Updated message: {updated_message}
                        """

                        self.msg_fv[factor][i] = updated_message

                    assert self.msg_fv[factor].shape == message.shape, f"""
                    Assert error for factor {factor} during factor-to-variable message update
                    The shapes between the original message and the updated message are not compatible
                    Original message shape: {message.shape}
                    Updated message shape: {self.msg_fv[factor].shape}
                    """

                else:
                    for i in range(self.num_sources + 1):
                        updated_message = factor.value * torch.exp(
                            torch.sum(self.msg_vf[factor], dim=0)
                        )

                        assert updated_message.shape == (factor.value.shape), f"""
                        Assert error for factor {factor} during factor-to-variable message update
                        The updated message shape is not correct
                        Updated message shape: {updated_message.shape}
                        Factor value shape: {factor.value.shape}
                        """

                        updated_message = torch.sum(
                            updated_message, dim=tuple(j for j in range(self.num_sources + 1) if i != j))

                        updated_message = updated_message / \
                            torch.sum(updated_message)

                        assert updated_message.shape == (self.num_latent_codes,), f"""
                        Assert error for factor {factor} during factor-to-variable message update
                        The updated message shape after the summation is not correct
                        Updated message shape: {updated_message.shape}, this should be {(self.num_latent_codes,)}
                        """

                        updated_message = torch.log(
                            updated_message)

                        assert not _check_if_nan_or_inf(updated_message), f"""
                        Assert error for factor {factor} during factor-to-variable message update
                        The updated message contains NaN or Inf values
                        Updated message before log operation: {updated_message}
                        Updated message: {updated_message}
                        """

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
                for i in range(self.num_sources if factor.type == FactorType.PRIOR else self.num_sources + 1):
                    incoming_messages = []
                    for fact, message in self.msg_fv.items():
                        if fact != factor:
                            incoming_messages.append(message)

                    stacked_incoming_messages = torch.cat(
                        incoming_messages, dim=0)

                    sum_incoming_messages = torch.sum(
                        stacked_incoming_messages, dim=0)

                    sum_incoming_messages = sum_incoming_messages / \
                        torch.sum(sum_incoming_messages)

                    assert sum_incoming_messages.shape == torch.Size([self.num_latent_codes, self.num_latent_codes]), f"""
                    Assert error for factor {factor} during variable-to-factor message update
                    The shape of the sum of the incoming messages is not correct
                    The shape of the sum of the incoming messages is {sum_incoming_messages.shape}, while it should be {torch.Size([self.num_latent_codes, self.num_latent_codes])}
                    """

                    self.msg_vf[factor][:, i] = sum_incoming_messages

                    assert torch.sum(self.msg_vf[factor], dim=0).shape == factor.value.shape, f"""
                    Assert error for factor {factor} during variable-to-factor message update
                    The shapes between the message shape and the factor shape are not compatible
                    Message sum shape: {torch.sum(self.msg_vf[factor], dim=0).shape}
                    Factor shape: {factor.value.shape}
                    """

        # Compute the marginals for all the variables
        # self._update_marginals()
        self.marginal_posterior = self._get_marginal_posterior()

        return

    def sample_sources(self, time_step: int) -> torch.Tensor:

        assert self.marginal_posterior.shape == torch.Size([self.num_sources, self.num_latent_codes, self.num_latent_codes]), f"""
        The shape of the marginals is not correct.
        The shape of the marginals is {self.marginal_posterior.shape}, while it should be {(self.num_sources, self.num_latent_codes, self.num_latent_codes)}
        """

        marginal_posterior_sliced = self.marginal_posterior[:,
                                                            self.mixture[time_step]]
        # top_k_probs = torch.topk(probs, k=10, dim=-1).values
        samples = torch.multinomial(
            marginal_posterior_sliced, num_samples=1, replacement=True)

        log.debug(f"Sampled shape is {samples.shape}")
        return samples

    def separate(self) -> torch.Tensor:
        # Autoregressive sampling until the sources reach the mixture length
        sources = torch.stack(self.pasts, dim=0).view(-1, 1)
        for t in tqdm(range(self.mixture_length - 1), desc="Separation"):
            # TODO: change according to the number of iterations
            self.belief_propagation(iterations=5)
            samples = self.sample_sources(time_step=t)
            sources = torch.cat((sources, samples), dim=1)
            log.debug(f"Sampled sources at step {t} are: {samples}")
            log.debug(f"Sources at step {t} are : {sources}")
            self.pasts[0] = sources[0].view(1, -1)
            self.pasts[1] = sources[1].view(1, -1)
        return sources


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

    factor_graph = FactorGraph(
        num_sources=2, mixture=mixture, likelihood=likelihood, transformer=transformer)

    sources = factor_graph.separate()

    print(f"The sampled sources are: {sources}")
