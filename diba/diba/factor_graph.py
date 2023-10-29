import time
from typing import Dict, List, Set, Tuple, Union
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


class FactorGraph:
    @timeit
    def __init__(self, num_sources: int, mixture: torch.Tensor, likelihood: DenseLikelihood) -> None:
        """
        Initialize the factor graph
        @params num_classes: number of sources to separate 
        @params mixture: the latent code of the mixture to separate
        """
        self.num_sources = num_sources  # 'k' in the equations
        self.mixture = mixture  # 'm' in the equations
        self.mixture_length = len(mixture)  # 'n' in the equations
        self.num_latent_codes = 256  # 256 in the case of MNIST
        # self.variables: Set[Variable] = set()
        # always None in the case of uncoditional priors
        self.pasts = [None for _ in range(self.num_sources)]
        self.likelihood = likelihood
        self.marginals: torch.Tensor = []

        log.debug(f"Length of the mixture is: {self.mixture_length}")

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
        UNCOND_BOS = 0
        self.priors = [UnconditionedTransformerPrior(
            transformer=self.transformer, sos=UNCOND_BOS) for _ in range(self.num_sources)]

        ##########################
        # Initialize the factors #
        ##########################

        self.factors: List[NewFactor] = []

        # add the marginal distribution of the mixture
        # p(m_1), p(m_2), ..., p(m_n)
        factor = NewFactor(type="mixture_marginal",
                           value=torch.full(size=(1, self.num_latent_codes),
                                            fill_value=1.0 / self.num_latent_codes))

        # TODO: I don't know if I should add this factor
        # self.factors.append(factor)

        # add the marginal distribution of the sources
        # p(z^j_1), p(z^j_2), ..., p(z^j_n)
        factor = NewFactor(type="source_marginal",
                           value=torch.full(size=(self.num_sources,
                                                  self.num_latent_codes),
                                            fill_value=1.0 / self.num_latent_codes))
        # TODO: I don't know if I should add this factor
        # self.factors.append(factor)

        # add the autoregressive priors
        # p(z^1_i | z^1_{i-1}), p(z^2_i | z^2_{i-1}), ..., p(z^k_i | z^k_{i-1})
        # the first prior is null
        # TODO: The shape should be (num_sources, num_latent_codes, num_latent_codes)
        # if the prior is conditioned to the class index, otherwise it should be (num_latent_codes, num_latent_codes)

        factor = NewFactor(type="prior",
                           value=torch.full(size=(self.num_sources,
                                                  self.num_latent_codes,
                                                  self.num_latent_codes),
                                            fill_value=1/self.num_latent_codes))
        self.factors.append(factor)

        # add the posterior factors
        # p(z^j_i | m_i)
        factor = NewFactor(type="posterior",
                           value=torch.full(size=(self.num_latent_codes,
                                                  self.num_latent_codes,
                                                  self.num_latent_codes),
                                            fill_value=1/self.num_latent_codes))
        self.factors.append(factor)

        # add the likelihood factors, which don't depend on the indices
        # p(m| z^1, z^2, ..., z^k)
        factor = NewFactor(type="likelihood",
                           value=torch.full(
                               size=tuple(self.num_latent_codes for _ in range(
                                   self.num_sources + 1)),
                               fill_value=1/self.num_latent_codes
                           ))
        self.factors.append(factor)

        ###########################
        # Initialize the messages #
        ###########################

        self.msg_fv: Dict[NewFactor, torch.Tensor] = {}
        self.msg_vf: Dict[NewFactor, torch.Tensor] = {}

        for factor in self.factors:
            self.msg_fv[factor] = torch.zeros(
                size=(factor.value.shape))
            self.msg_vf[factor] = torch.zeros(
                size=(factor.value.shape))

    def __repr__(self):
        return f"""FactorGraph(
            num_sources={self.num_sources},
            mixture={self.mixture},
            mixture_length={self.mixture_length}, 
            num_latent_codes={self.num_latent_codes},
            factors={self.factors}"""

    def _element_wise_prod_excluding_row(self, tensor1: torch.Tensor, tensor2: torch.Tensor, row: int) -> torch.Tensor:
        """
        Element-wise product of two tensors, excluding the given row
        """
        result = tensor1 * tensor2
        result = torch.cat(
            (result[:row], result[row+1:]), dim=0)
        return result

    def _compute_sparse_likelihood(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ll_coords, ll_data = self.likelihood.get_log_likelihood(
            self.mixture[idx])
        return ll_coords, ll_data

    def _compute_dense_likelihood(self) -> torch.Tensor:
        """
        Returns the discrete likelihood K x K x ... K (m+1 times) matrix, where m is the number of sources and K is the number of latent codes.
        This models the likelihood p(m|z^1, z^2, ..., z^m)
        """

        return self.likelihood.get_dense_log_likelihood()

    def _compute_dense_log_posterior(self, log_prior: torch.Tensor) -> torch.Tensor:
        """
        Returns the discrete posterior K x K x ... K (m times) matrix, where m is the number of sources and K is the number of latent codes.
        This models the posterior p(z^i| m) for all the is.
        """
        log.debug(
            f"Dense likelihood shape is: {self.likelihood.get_dense_log_likelihood().shape}")
        log.debug(f"Prior shape is: {log_prior.shape}")

        log_likelihood = self.likelihood.get_dense_log_likelihood()
        # TODO: i'm just doing this for debugging reasons, the posterior should be (2,256,256,256) if class conditioned
        source = 0
        log_posterior = log_likelihood + log_prior[source, :, :]
        # Iterate over each source and compute the log_posterior
        return log_posterior

    def _update_autoregressive_prior(self):
        """
        Updates the value for the factor that refers to the autoregressive priors 
        p(z^j_i | z^j_{i-1}) for all the j and i.
        """
        log.debug(f"Computing the autoregressive priors")
        prior_factor = [
            factor for factor in self.factors if factor.type == "prior"][0]
        for class_idx in range(self.num_sources):
            class_prior: UnconditionedTransformerPrior = self.priors[class_idx]
            for i in range(self.mixture_length):
                current_source = self.pasts[class_idx] if self.pasts[class_idx] is not None else class_prior.get_sos(
                ).unsqueeze(0)
                current_source = current_source.to(torch.long)

                log.debug(
                    f"During autoregressive prior update, the current source shape is {current_source.shape}")

                # Log priors shape should be torch.Size([1, 256])
                # Current source shape sould be torch.Size([1, n]) where n is the number of tokens samples so far
                log_priors, _ = class_prior.get_logits(
                    token_ids=current_source,
                    past_key_values=None
                )

                probs = torch.softmax(log_priors, dim=-1)
                log.debug(
                    f"During autoregressive prior update, the prior shape is {prior_factor.value.shape} \n the probs shape is {probs.shape} \n the log_priors shape is {log_priors.shape}")
                prior_factor.value[class_idx, i, :] = probs

    def _update_posterior(self):
        """
        Updates the value for the factor that refers to the posterior
        """
        log.debug(f"Computing the posterior")
        posterior_factor = [
            factor for factor in self.factors if factor.type == "posterior"][0]
        prior_factor = [
            factor for factor in self.factors if factor.type == "prior"][0]

        posterior_value = self._compute_dense_log_posterior(
            log_prior=prior_factor.value)

        log.debug(
            f"During the posterior update, the posterior value is: {posterior_value}")

        posterior_factor.value = posterior_value

    def _update_marginals(self):
        """
        Updates the marginals for all the variables
        """
        pass

    def _compute_posterior(self, ll_data: torch.Tensor, ll_coords: torch.Tensor, priors: torch.Tensor, idx: int) -> torch.Tensor:

        def _compute_log_posterior(
            nll_data: torch.Tensor,
            nll_coords: torch.LongTensor,
            log_priors: torch.Tensor,
        ):

            # Inside the posterior computation, there are 2 priors, with shapes: [torch.Size([256, 256]), torch.Size([256, 256])]                          | 0/64 [00:00<?, ?it/s]
            # Inside the posterior computation, there are 2 priors, with shapes: [torch.Size([1, 256]), torch.Size([1, 256])]
            log.debug(
                f"Inside the posterior computation the priors shape is: {log_priors.shape}")

            log.debug(
                f"Inside the posterior computation, nll_coords shape is {nll_coords.shape}, while nll_data shape is {nll_data.shape}")

            dense_log_likelihood = torch.sparse_coo_tensor(
                nll_coords, nll_data, size=(256, 256)).to_dense()

            return log_priors + dense_log_likelihood

        posterior = _compute_log_posterior(
            nll_data=ll_data,
            nll_coords=ll_coords,
            log_priors=priors,
        )
        probs = torch.softmax(posterior, dim=-1)

        return probs

    def aggregate_messages(self, messages, operation="sum"):
        """Aggregates incoming messages into a variable via the specified operation.

        Args:
            - messages: A list of incoming messages, each with a different shape.
            - operation: The operation to use for aggregation (only "sum" is supported at the moment)

        Returns:
            A tensor containing the aggregated messages.
        """

        # TODO: I don't know if this keeps the semantic of the messages
        # Reshape the incoming messages to the same shape.
        messages: List[torch.Tensor] = torch.broadcast_tensors(*messages)

        messages: torch.Tensor = torch.stack(messages, dim=0)

        if operation == "sum":
            # Sum the incoming messages.
            aggregated_messages = torch.sum(messages, dim=0)
        else:
            raise NotImplementedError(
                f"Aggreagtion via the operation {operation} is not implemented yet"
            )

        return aggregated_messages

    @timeit
    def belief_propagation(self, iterations: int = 30):
        """
        Run the sum-product algorithm for the given number of iterations in order to compute the marginal distributions.
        """

        for it in tqdm(range(iterations), desc="Belief Propagation"):
            # NOTE: Brainstorming: https://chat.openai.com/share/b9eabf18-3d2d-475b-9b97-c080c9341661
            # Computing the local factor values:
            log.debug(
                f"In order to debug, the shape of the variable-to-factor messages are {[message.shape for message in self.msg_vf.values()]}")
            for factor in self.factors:
                if factor.type == "prior":
                    self._update_autoregressive_prior()
                if factor.type == "posterior":
                    self._update_posterior()
            # Update all factor-to-variable messages
            for factor, message in self.msg_fv.items():
                log.debug(f"Updating the message for factor {factor}")
                # Non unary factors
                if factor.type == "prior":
                    log.debug(
                        f"Factor {factor} has a value with shape {factor.value.shape}")

                    # Note: msg_vf gets very small and so the exponentiation of it gets near to zero, so the result is 0, and the log of 0 is -inf
                    updates_message_other_v = factor.value * torch.exp(
                        self.msg_vf[factor]
                    )

                    log.debug(f"""
                    During the prior_factor-to-variable message update:
                    The factor value is {factor.value}
                    The incoming messages are {self.msg_vf[factor]}
                    The exponentiated incoming messages are {torch.exp(self.msg_vf[factor])}""")

                    updated_message = torch.log(
                        updates_message_other_v + 1e-10)

                    assert not _check_if_nan_or_inf(updated_message), f"""
                    Assert error for factor {factor} during factor-to-variable message update
                    The updated message contains NaN or Inf values
                    Updated message before log operation: {updates_message_other_v}
                    Updated message: {updated_message}
                    """

                    log.debug(
                        f"Final updated message for {factor} has shape {updated_message.shape}, where the original message had shape: {message.shape}")

                    assert updated_message.shape == message.shape == factor.value.shape, f"""
                    Assert error for factor {factor} during factor-to-variable message update
                    The shapes between the original message, the updated message and the factor value are not compatible
                    Original message shape: {message.shape}
                    Updated message shape: {updated_message.shape}
                    Factor value shape: {factor.value.shape}
                    """

                    self.msg_fv[factor] = updated_message

                    log.debug(
                        f"The message after the update has shape {self.msg_fv[factor].shape}")

                    if factor.type == "likelihood" or factor.type == "posterior":
                        # TODO: I don't know if I can update the prior and the posterior in the same way.

                        stacked_updated_messages = torch.zeros(
                            size=(self.num_latent_codes, self.num_latent_codes, self.num_latent_codes))

                        # TODO: change this when you will deal with more than 2 sources
                        for i in range(self.num_latent_codes):

                            log.debug(
                                f"The message from variable to factor {factor} has shape {self.msg_vf[factor].shape}")

                            updated_message_z1 = factor.value[i, :, :] * torch.exp(
                                self.msg_vf[factor][i, :, :] +
                                self.msg_vf[factor][:, i, :]
                            )
                            updated_message_z2 = factor.value[:, i, :] * torch.exp(
                                self.msg_vf[factor][i, :, :] +
                                self.msg_vf[factor][:, i, :]
                            )

                            updated_message = torch.log(
                                updated_message_z1 + updated_message_z2)

                            stacked_updated_messages[i, :, :] = updated_message

                        assert not _check_if_nan_or_inf(updated_message), f"""
                        Assert error for factor {factor} during factor-to-variable message update
                        The updated message contains NaN or Inf values
                        Updated message before log operation: {updates_message_other_v}
                        Updated message: {updated_message}
                        """

                        assert stacked_updated_messages.shape == message.shape == factor.value.shape, f"""
                        Assert error for factor {factor} during factor-to-variable message update
                        The shapes between the original message, the updated message and the factor value are not compatible
                        Original message shape: {message.shape}
                        Updated message shape: {updated_message.shape}
                        Factor value shape: {factor.value.shape}
                        """

                        self.msg_fv[factor] = stacked_updated_messages

                        log.debug(
                            f"Likelihood message after update has shape {self.msg_fv[factor].shape}, the original message had shape {message.shape}")
            log.debug(f"Updated unary factor-to-variable messages")

            # Update all variable-to-factor messages
            # NOTE: I just have to take the sum of the incoming messages, except for the one that is coming from the factor itself
            for factor, message in self.msg_vf.items():
                incoming_messages = [
                    message for fact in self.factors for message in self.msg_fv[fact] if fact != factor]

                log.debug(
                    f"There are {len(incoming_messages)} incoming messages (excluding the one from the factor {factor}), with shapes {set(message.shape for message in incoming_messages)}")

                sum_incoming_messags = self.aggregate_messages(
                    incoming_messages)

                log.debug(
                    f"There are {len(incoming_messages)} incoming messages (excluding the one from the factor {factor})")
                log.debug(
                    f"The shapes of the messages are {set(message.shape for message in incoming_messages)}")
                # TODO: there is an error here since the shape is always (256,256), while it should be the shape of the factor value
                log.debug(
                    f"The shape of sum_incoming_messags is {sum_incoming_messags.shape}")

                self.msg_vf[factor] = sum_incoming_messags

        # Compute the marginals for all the variables
        for factor, message in self.msg_vf.items():
            incoming_messages = [
                message for fact in self.factors for message in self.msg_fv[fact]]

            self.marginals = self.aggregate_messages(incoming_messages)

        log.debug(
            f"At the end of the belief propagation, the marginals are: {self.marginals}")

        return

    def sample_sources(self) -> torch.Tensor:
        # Sample mixture_length times from the marginal distribution of each variable
        # Sample a source for each index
        probs = torch.softmax(self.marginals, dim=-1)
        source_1 = torch.multinomial(probs[0], 1)
        source_2 = torch.multinomial(probs[1], 1)

        return torch.cat((source_1, source_2), dim=0)

    def separate(self) -> torch.Tensor:
        # TODO: AUTOREGRESSION: sample one, insert it in the source tensor, update the messages with the new value, and then sample the next one
        sources = None
        for t in tqdm(range(self.mixture_length), desc="Separation"):
            self.belief_propagation(iterations=10)
            partial_sources = self.sample_sources().reshape(2, 1)
            if sources is None:
                sources = partial_sources
            else:
                sources = torch.cat((sources, partial_sources), dim=1)
            log.debug(f"Partial sources at step {t} are: {partial_sources}")
            print(f"Sources so far are: {sources}")
            self.pasts[0] = sources[0].reshape(1, -1)
            self.pasts[1] = sources[1].reshape(1, -1)

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

    likelihood = DenseLikelihood(sums=sums)

    factor_graph = FactorGraph(
        num_sources=2, mixture=mixture, likelihood=likelihood)

    sources = factor_graph.separate()

    print(f"The sampled sources are: {sources}")
