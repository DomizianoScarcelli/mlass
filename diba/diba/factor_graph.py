import time
from typing import Dict, List, Set, Tuple, Union
import numpy as np
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


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def _check_if_nan_or_inf(tensor: torch.Tensor) -> bool:
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


class Variable:
    _variables = {}

    def __new__(cls, class_idx: Union[int, None], idx: int, mixture: bool = False):
        key = (class_idx, idx, mixture)
        if key in Variable._variables:
            return Variable._variables[key]
        else:
            instance = super(Variable, cls).__new__(cls)
            instance.class_idx = class_idx
            instance.idx = idx
            instance.mixture = mixture
            Variable._variables[key] = instance
            return instance

    def __init__(self, class_idx: Union[int, None], idx: int, mixture: bool = False):
        # Initialize only if it's a new instance
        if not hasattr(self, 'initialized'):
            self.neigh_factors = set()
            self.outgoing_messages: Set[Message] = set()
            self.incoming_messages: Set[Message] = set()
            self.class_idx = class_idx
            self.idx = idx
            self.mixture = mixture

            # TODO: do not hardcode this
            self.marginal = torch.full((256,), (1 / 256))
            self.initialized = True

    def __hash__(self):
        return hash((self.class_idx, self.idx, self.mixture))

    def __repr__(self):
        if self.mixture:
            return f"m_{self.idx}"
        return f"z^{self.class_idx}_{self.idx}"


class NewVariable:
    _variables = {}

    def __new__(cls, class_idx: Union[int, None], value: torch.Tensor):
        key = "mixture" if class_idx is None else class_idx
        if key in NewVariable._variables:
            return NewVariable._variables[key]
        else:
            instance = super(Variable, cls).__new__(cls)
            instance.class_idx = class_idx
            instance.value = value
            NewVariable._variables[key] = instance
            return instance

    def __init__(self, class_idx: Union[int, None], value: torch.Tensor):
        # Initialize only if it's a new instance
        self.class_idx = class_idx
        self.value = value
        self.neigh_factors = set()
        self.outgoing_messages: Set[Message] = set()
        self.incoming_messages: Set[Message] = set()

    def __hash__(self):
        return hash((self.class_idx))

    def __repr__(self):
        if self.class_idx is None:
            return f"m"
        return f"z^{self.class_idx}"


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
        self.incoming_messages: Set[Message] = set()
        self.outgoing_messages: Set[Message] = set()

    def __hash__(self):
        return hash((self.type))

    def __repr__(self):
        return f"NewFactor(type={self.type})"


class Factor:
    _connected_vars_instances = {}
    _factors = {}

    def __new__(cls, type: str, idx: int, connected_vars: List[Variable], value: torch.Tensor):
        key = (type, idx, tuple(connected_vars))
        if key in Factor._factors:
            return Factor._factors[key]
        else:
            instance = super(Factor, cls).__new__(cls)
            instance.type = type
            instance.idx = idx
            instance.connected_vars = connected_vars
            instance.value = value
            Factor._factors[key] = instance
            return instance

    def __init__(self, type: str, idx: int, connected_vars: List[Variable], value: torch.Tensor):
        # Initialize only if it's a new instance
        if not hasattr(self, 'initialized'):
            self.value = value
            self.incoming_messages: Set[Message] = set()
            self.outgoing_messages: Set[Message] = set()
            self.initialized = True
            self.connected_vars = connected_vars
            self.type = type
            self.idx = idx

    def __hash__(self):
        return hash((self.type, self.idx, tuple(self.connected_vars)))

    def __repr__(self):
        return f"Factor(type={self.type}, idx={self.idx}, connected_vars={self.connected_vars})"


class SparseFactor:
    _connected_vars_instances = {}
    _factors = {}

    def __new__(cls, type: str, idx: int, connected_vars: List[Variable], value_coords: torch.Tensor, value_data: torch.Tensor):
        key = (type, idx, tuple(connected_vars))
        if key in SparseFactor._factors:
            return SparseFactor._factors[key]
        else:
            instance = super(SparseFactor, cls).__new__(cls)
            instance.type = type
            instance.idx = idx
            instance.connected_vars = connected_vars
            instance.value_coords = value_coords
            instance.value_data = value_data
            SparseFactor._factors[key] = instance
            return instance

    def __init__(self, type: str, idx: int, connected_vars: List[Variable], value_coords: torch.Tensor, value_data: torch.Tensor):
        # Initialize only if it's a new instance
        if not hasattr(self, 'initialized'):
            self.value_coords = value_coords
            self.value_data = value_data
            self.incoming_messages = set()
            self.outgoing_messages = set()
            self.initialized = True
            self.connected_vars = connected_vars
            self.type = type
            self.idx = idx

    def __hash__(self):
        return hash((self.type, self.idx, tuple(self.connected_vars)))

    def __eq__(self, other):
        return isinstance(other, SparseFactor) and (
            self.type, self.idx, tuple(self.connected_vars)) == (other.type, other.idx, tuple(other.connected_vars))

    def __repr__(self):
        return f"SparseFactor(type={self.type}, idx={self.idx}, connected_vars={self.connected_vars})"


class Message:
    _messages = {}

    def __new__(cls, _from: Union[Factor, Variable, SparseFactor], _to: Union[Factor, Variable, SparseFactor], value: torch.Tensor):
        key = (_from, _to)
        if key in Message._messages:
            return Message._messages[key]
        else:
            instance = super(Message, cls).__new__(cls)
            instance._from = _from
            instance._to = _to
            instance.value = value
            Message._messages[key] = instance
            return instance

    def __init__(self, _from: Union[Factor, Variable, SparseFactor], _to: Union[Factor, Variable, SparseFactor], value: torch.Tensor):
        # Initialize only if it's a new instance
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._from = _from
            self._to = _to
            self.value = value

    def __hash__(self):
        return hash((self._from, self._to))

    def __repr__(self):
        return f"Message(from={self._from}, to={self._to}, value_shape={self.value.shape})"


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
        self.variables: Set[Variable] = set()
        # always None in the case of uncoditional priors
        self.pasts = [None for _ in range(self.num_sources)]
        self.likelihood = likelihood

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

        self.factors: List[Factor] = []

        # add the marginal distribution of the mixture
        # p(m_1), p(m_2), ..., p(m_n)
        factor = NewFactor(type="mixture_marginal",
                           value=torch.full(size=(1, self.num_latent_codes),
                                            fill_value=1.0 / self.num_latent_codes))

        self.factors.append(factor)

        # add the marginal distribution of the sources
        # p(z^j_1), p(z^j_2), ..., p(z^j_n)
        factor = NewFactor(type="source_marginal",
                           value=torch.full(size=(self.num_sources,
                                                  self.num_latent_codes),
                                            fill_value=1.0 / self.num_latent_codes))
        self.factors.append(factor)

        # add the autoregressive priors
        # p(z^1_i | z^1_{i-1}), p(z^2_i | z^2_{i-1}), ..., p(z^k_i | z^k_{i-1})
        # the first prior is null
        factor = NewFactor(type="prior",
                           value=torch.full(size=(self.num_sources,
                                                  self.num_latent_codes,
                                                  self.num_latent_codes),
                                            fill_value=1/self.num_latent_codes))

        self.factors.append(factor)

        # add the posterior factors
        # p(z^j_i | m_i)
        factor = NewFactor(type="posterior",
                           value=torch.full(size=(self.num_sources,
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

        self.msg_fv: Dict[Tuple[NewFactor, int], torch.Tensor] = {}
        self.msg_vf: Dict[Tuple[int, NewFactor], torch.Tensor] = {}

        for factor in self.factors:
            for class_idx in range(self.num_sources):
                self.msg_fv[(factor, class_idx)] = torch.zeros(
                    size=(factor.value.shape))
                self.msg_vf[(class_idx, factor)] = torch.zeros(
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

    def _compute_dense_log_posterior(self, prior: torch.Tensor) -> torch.Tensor:
        """
        Returns the discrete posterior K x K x ... K (m times) matrix, where m is the number of sources and K is the number of latent codes.
        This models the posterior p(z^i| m) for all the is.
        """
        return prior + self._compute_dense_likelihood()

    def _compute_autoregressive_prior(self):
        """
        Compute the autoregressive prior p(z^j_i | z^j_{i-1}) for all the j and i.
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

            # assert num_sources == nll_coords.shape[0], f"""
            #     The number of sources is not the same as the number of likelihoods.
            #     The number of sources is: {num_sources}
            #     The number of likelihoods is: {nll_coords.shape[0]}
            # """
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

    def aggregate_messages(self, messages):
        """Aggregates incoming messages into a variable via a summation to compute the marginals.

        Args:
            messages: A list of incoming messages, each with a different shape.

        Returns:
            A tensor containing the aggregated messages.
        """

        # TODO: I don't know if this keeps the semantic of the messages

        # Reshape the incoming messages to the same shape.
        messages: List[torch.Tensor] = torch.broadcast_tensors(*messages)

        log.debug(
            f"After the aggregatiojn, there are {len(messages)} messages, with shapes: {[message.shape for message in messages]}")

        messages: torch.Tensor = torch.stack(messages, dim=0)

        log.debug(
            f"After the stacking, the aggregated message shape is: {messages.shape}"
        )

        # Sum the incoming messages.
        aggregated_messages = torch.sum(messages, dim=0)

        return aggregated_messages

    @timeit
    def belief_propagation(self, iterations: int = 30):
        """
        Run the sum-product algorithm for the given number of iterations in order to compute the marginal distributions.
        """

        for it in tqdm(range(iterations), desc="Belief Propagation"):
            # Update all factor-to-variable messages
            for (factor, class_idx), message in self.msg_fv.items():
                # Unary factors
                if factor.type == "mixture_marginal" or factor.type == "source_marginal":
                    self.msg_fv[(factor, class_idx)] = torch.log(factor.value)
                # Non unary factors
                else:
                    if factor.type == "prior" or factor.type == "posterior":
                        # I don't know if I can update the prior and the posterior in the same way.
                        # I think the code for the prior is correct btw

                        log.debug(
                            f"Factor {factor} has a value with shape {factor.value.shape}")

                        other_v = 0 if class_idx == 1 else 1
                        updates_message_other_v = factor.value[other_v, :, :] * torch.exp(
                            self.msg_vf[(other_v, factor)][other_v, :, :]
                        )
                        updated_message = torch.log(
                            updates_message_other_v)

                        log.debug(
                            f"Updated message has shape {updated_message.shape}")

                        log.debug(
                            f"Final updated message for {factor} is: {updated_message} with shape {updated_message.shape}")

                        self.msg_fv[(factor, class_idx)
                                    ] = updated_message

                    if factor.type == "likelihood":
                        # You may remove this loop
                        stacked_updated_messages = torch.zeros(
                            size=(self.num_latent_codes, self.num_latent_codes, self.num_latent_codes))
                        for i in range(self.num_latent_codes):

                            updated_message_z1 = factor.value[i, :, :] * torch.exp(
                                self.msg_vf[(0, factor)][i, :, :] +
                                self.msg_vf[(1, factor)][:, i, :]
                            )
                            updated_message_z2 = factor.value[:, i, :] * torch.exp(
                                self.msg_vf[(0, factor)][i, :, :] +
                                self.msg_vf[(1, factor)][:, i, :]
                            )

                            updated_message = torch.log(
                                updated_message_z1 + updated_message_z2)

                            stacked_updated_messages[i, :, :] = updated_message

                            log.debug(
                                f"Final updated message for likelihood factor is: {updated_message} with shape {updated_message.shape}")

                        self.msg_fv[(factor, class_idx)
                                    ] = stacked_updated_messages

            log.debug(f"Updated unary factor-to-variable messages")
            raise Exception("STOPPP")

            # Update all factor-to-variable messages

            non_unary_factors = [
                fac for fac in self.factors if not (fac.type == "mixture_marginal" or fac.type == "source_marginal")]
            for factor in non_unary_factors:
                if factor.type == "prior":
                    for fac_var in factor.connected_vars:
                        # NOTE: in the case of uncoditional priors, the past is always None
                        class_prior: UnconditionedTransformerPrior = self.priors[fac_var.class_idx]

                        past = self.pasts[fac_var.class_idx]

                        log.debug(f"Factor value idx is: {fac_var.idx}")
                        log.debug(
                            f"Mixture is {self.mixture} with shape: {self.mixture.shape}")
                        observed_mixture = self.mixture[fac_var.idx].unsqueeze(
                            0).unsqueeze(0).to(torch.long)

                        log.debug(
                            f"Obseved mixture is: {observed_mixture}")

                        log_priors, new_past = class_prior.get_logits(
                            token_ids=observed_mixture,
                            past_key_values=past)

                        log.debug(
                            f"Log priors before normalization for {fac_var} are: {log_priors} with shape {log_priors.shape}")

                        # TODO: don't know if this is needed
                        # log_priors = normalize_logits(log_priors)

                        log.debug(
                            f"Log priors after normalization for {fac_var} are: {log_priors} with shape {log_priors.shape}")

                        self.pasts[fac_var.class_idx] = new_past

                        prob_priors = torch.softmax(log_priors, dim=-1)

                        updated_factor_value = prob_priors.squeeze(0)

                        assert not _check_if_nan_or_inf(
                            updated_factor_value), f"The updated factor value for {factor} value is nan during autoregressive prior computation"

                        factor.value[:,
                                     fac_var.class_idx] = updated_factor_value

                if factor.type == "likelihood":
                    # Compute the likelihood
                    ll_coords, ll_data = self._compute_sparse_likelihood(
                        factor.idx)
                    factor.value_coords = ll_coords
                    factor.value_data = ll_data

                if factor.type == "posterior":
                    # Compute the posterior

                    # TODO: I don't know if this is correct, since I'm grabbing the likelihood factor by index
                    likelihood_factor: SparseFactor = [
                        fac for fac in self.factors if fac.type == "likelihood" and fac.idx == factor.idx][0]

                    prior_factor: Factor = [
                        fac for fac in self.factors if fac.type == "prior" and fac.idx == factor.idx]

                    if prior_factor == []:
                        log.debug(f"Prior factor is empty")

                        priors = torch.full(
                            (self.num_latent_codes, self.num_latent_codes), 1.0 / self.num_latent_codes)
                    else:
                        log.debug(
                            f"Considering the prior factor {prior_factor}")
                        prior_factor = prior_factor[0]

                        log.debug(
                            f"Prior factor value shape is {prior_factor.value.shape}")
                        # TODO: don't know how to handle this, since I thought I had one prior for each source, but actually the prior is 256 x 256
                        # priors = [prior_factor.value[:, 0].unsqueeze(0),
                        #           prior_factor.value[:, 1].unsqueeze(0)]
                        priors = prior_factor.value

                    # Convert the probabilities to log probabilities
                    # priors = [torch.log(prior) for prior in priors]
                    priors = torch.log(priors)

                    ll_coords, ll_data = likelihood_factor.value_coords, likelihood_factor.value_data
                    updated_posterior = self._compute_posterior(
                        ll_data=ll_data, ll_coords=ll_coords, priors=priors, idx=factor.idx)

                    log.debug(
                        f"Updated posterior for idx: {factor.idx} is {updated_posterior}, with shape {updated_posterior.shape}")

                    factor.value = updated_posterior

                for outgoing_message in factor.outgoing_messages:
                    log.debug(f"-------------------")
                    variable: Variable = outgoing_message._to
                    log.debug(f"Factor: {factor}")
                    log.debug(f"Variable: {variable}")

                    assert variable in factor.connected_vars, f"""
                        The variable {variable} is not connected to the factor {factor}.
                        The connected variables are: {factor.connected_vars}
                    """

                    # TODO: don't know if this is correct
                    if isinstance(factor, SparseFactor):
                        value = torch.sparse_coo_tensor(
                            factor.value_coords, factor.value_data, size=(self.num_latent_codes, self.num_latent_codes))
                        value = value.to_dense()
                        value = torch.softmax(value, dim=-1)
                        log.debug(
                            f"Non-zero values for the factor {factor} are: {value[value != 0]}")
                    else:
                        value = factor.value

                    log.debug(
                        f"Neigh variables to factor {factor}: {factor.connected_vars}")

                    log.debug(
                        f"Factor value is {value} with shape {value.shape}"
                    )

                    variable_idx_in_factor = factor.connected_vars.index(
                        variable)

                    broadcasted_messages = torch.broadcast_tensors(
                        *[message.value for message in variable.incoming_messages])

                    messages_from_var_to_factor = torch.stack(
                        broadcasted_messages, dim=0)

                    log.debug(
                        f"Messages from variables to factor {factor}: {messages_from_var_to_factor} ")

                    messages_from_var_to_factor_sum = torch.sum(
                        messages_from_var_to_factor, dim=0)

                    # NOTE: If messagges are too big, the exp will return inf
                    assert (messages_from_var_to_factor_sum < 50).all(), f"""
                        The messages from variables to factor {factor} are too big.
                        The messages are: {messages_from_var_to_factor}
                        The big numbers are: {messages_from_var_to_factor_sum[messages_from_var_to_factor_sum > 50]}
                    """

                    log.debug(
                        f"Sum of the messages from variables to factor {factor}: {messages_from_var_to_factor_sum}")

                    sum_of_prod = torch.sum(self._element_wise_prod_excluding_row(value.T, torch.exp(
                        messages_from_var_to_factor_sum), variable_idx_in_factor), dim=0)

                    log.debug(
                        f"Sum of the product for factor {factor} is: {sum_of_prod}")

                    log_sum_prod = torch.log(sum_of_prod + 1e-7)

                    log.debug(
                        f"Log sum of the product for factor {factor} is: {log_sum_prod}, with a mean of {torch.mean(log_sum_prod)}")

                    updated_message = log_sum_prod - torch.mean(log_sum_prod)

                    log.debug(
                        f"Updated message shape: {updated_message.shape}")

                    assert not _check_if_nan_or_inf(
                        updated_message), f"""
                        The updated message for {outgoing_message} is nan during factor-to-variable message computation.

                        For debug: 
                        factor.value: {value}
                        messages_from_var_to_factor_sum: {messages_from_var_to_factor_sum}
                        exp: {torch.exp(messages_from_var_to_factor_sum)}
                        simple_prod: {value.T * torch.exp(messages_from_var_to_factor_sum)}
                        element_wise_prod_excluding_row: {self._element_wise_prod_excluding_row(value.T, torch.exp(messages_from_var_to_factor_sum), variable_idx_in_factor)}
                        sum: {torch.sum(self._element_wise_prod_excluding_row(value.T, torch.exp(messages_from_var_to_factor_sum), variable_idx_in_factor), dim=0)}
                        updated_message (log): {updated_message}
                        """

                    outgoing_message.value = updated_message

            # Update all variable-to-factor messages
            for variable in self.variables:
                log.debug(
                    f"Variable {variable} incoming messages are: {[variable.incoming_messages]}, outgoing messages are: {variable.outgoing_messages}")
                for outgoing_message in variable.outgoing_messages:
                    factor: Factor = outgoing_message._to

                    # In a certain case, the shapes of the incoming message are
                    # [torch.Size([256, 256]), torch.Size([256]), torch.Size([256]), torch.Size([256])]

                    incoming_messages = [
                        message.value for message in variable.incoming_messages if message._from != factor]

                    log.debug(
                        f"Incoming messages for {variable} are: {incoming_messages}")

                    incoming_messages_sum = self.aggregate_messages(
                        incoming_messages)

                    log.debug(
                        f"Sum of the incoming messages for {variable} is: {incoming_messages_sum}")

                    assert not _check_if_nan_or_inf(
                        incoming_messages_sum), f"The updated message for {outgoing_message} is nan during variable-to-factor message computation"

                    # Computing marginal for the variable
                    all_incoming_messages_stack = self.aggregate_messages(
                        [message.value for message in variable.incoming_messages]),

                    variable.marginal = all_incoming_messages_stack

                    # print(
                    #     f"Marginals for the variable {variable} after iteration {it} are: {variable.marginal} with shape {variable.marginal.shape}")

                    outgoing_message.value = incoming_messages_sum - \
                        torch.log(torch.sum(torch.exp(incoming_messages_sum)))

        return

    def sample_sources(self) -> torch.Tensor:
        sources = torch.zeros(self.num_sources, self.mixture_length)
        for variable in self.variables:
            if variable.class_idx is not None:
                logits = variable.marginal[0]

                # Create a Gumbel noise tensor
                gumbel_noise = torch.empty_like(logits).uniform_()

                # Add the Gumbel noise tensor to the logits tensor
                logits_with_noise = logits + gumbel_noise

                # Take the softmax of the resulting tensor
                probs = torch.softmax(logits_with_noise, dim=-1)

                # Sample from the tensor of probabilities
                samples = torch.multinomial(probs, num_samples=1)

                sample = samples[0]

                log.debug(f"Sample for variable {variable} is: {sample}")

                sources[variable.class_idx, variable.idx] = sample
        return torch.tensor(sources, dtype=torch.long)

    def separate(self) -> torch.Tensor:
        self.belief_propagation(iterations=20)
        sources = self.sample_sources()
        return sources


def _test_build_pattern():
    var1 = Variable(class_idx=0, idx=0)
    var2 = Variable(class_idx=0, idx=0)

    assert var1 == var2, f"{var1} is not equal to {var2}"

    fac1 = Factor(type="test", idx=0, connected_vars=[
                  var1], value=torch.zeros((1, 2)))
    fac2 = Factor(type="test", idx=0, connected_vars=[
                  var2], value=torch.zeros((1, 2)))

    assert fac1 == fac2, f"{fac1} is not equal to {fac2}"

    sparse_fac1 = SparseFactor(type="test", idx=0, connected_vars=[
        var1], value_coords=torch.zeros((1, 2)), value_data=torch.zeros((1,)))
    sparse_fac2 = SparseFactor(type="test", idx=0, connected_vars=[
                                    var2], value_coords=torch.zeros((1, 2)), value_data=torch.zeros((1,)))

    assert sparse_fac1 == sparse_fac2, f"{sparse_fac1} is not equal to {sparse_fac2}"

    message1 = Message(_from=fac1, _to=var1, value=torch.zeros((1, 2)))
    message2 = Message(_from=fac2, _to=var2, value=torch.zeros((1, 2)))

    assert message1 == message2, f"{message1} is not equal to {message2}"

    tricky_message_1 = Message(_from=fac1, _to=Variable(
        class_idx=0, idx=0), value=torch.zeros((1, 2)))

    assert tricky_message_1 == message1, f"{tricky_message_1} is not equal to {message1}"


if __name__ == "__main__":
    ###############################
    # Test the factor graph class #
    ###############################
    _test_build_pattern()

    #############
    # Main code #
    #############

    MIXTURE_LENGTH = 49
    mixture = torch.tensor([210, 135, 8, 202, 135, 8, 56, 39, 63, 168, 149, 119, 70, 56, 137, 149, 93, 217, 217, 217, 8, 210, 66,
                            254, 26, 9, 168, 135, 210, 29, 26, 88, 222, 75, 210, 56, 56, 88, 4, 34, 80, 8, 56, 137, 75, 7, 41, 8, 56])

    SUMS_CHECKPOINT_PATH = "lass_mnist/checkpoints/sum/256-sigmoid-big.pt"
    with open(SUMS_CHECKPOINT_PATH, 'rb') as f:
        sums = torch.load(f, map_location=torch.device('cpu'))

    likelihood = DenseLikelihood(sums=sums)

    factor_graph = FactorGraph(
        num_sources=2, mixture=mixture, likelihood=likelihood)

    sources = factor_graph.separate()

    print(f"The sampled sources are: {sources}")
