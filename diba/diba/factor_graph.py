import time
from typing import Dict, List, Set, Tuple, Union
import numpy as np
import torch
from tqdm import tqdm
from diba.diba.diba import _compute_log_posterior
from diba.diba.utils import normalize_logits

from lass_mnist.lass.diba_interaces import DenseLikelihood, UnconditionedTransformerPrior
from transformers import GPT2LMHeadModel, GPT2Config
from torch.nn.functional import pad

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
    logging.basicConfig(filename='last_run.log', level=logging.DEBUG,
                        format='%(name)s - %(levelname)s - %(message)s')


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

    def __new__(cls, classIdx: Union[int, None], idx: int, mixture: bool = False):
        key = (classIdx, idx, mixture)
        if key in Variable._variables:
            return Variable._variables[key]
        else:
            instance = super(Variable, cls).__new__(cls)
            instance.classIdx = classIdx
            instance.idx = idx
            instance.mixture = mixture
            Variable._variables[key] = instance
            return instance

    def __init__(self, classIdx: Union[int, None], idx: int, mixture: bool = False):
        # Initialize only if it's a new instance
        if not hasattr(self, 'initialized'):
            self.neigh_factors = set()
            self.outgoing_messages = set()
            self.incoming_messages = set()
            self.marginal = None
            self.initialized = True

    def __hash__(self):
        return hash((self.classIdx, self.idx, self.mixture))

    def __repr__(self):
        if self.mixture:
            return f"m_{self.idx}"
        return f"z^{self.classIdx}_{self.idx}"


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
            Factor._factors[key] = instance
            return instance

    def __init__(self, type: str, idx: int, connected_vars: List[Variable], value: torch.Tensor):
        # Initialize only if it's a new instance
        if not hasattr(self, 'initialized'):
            self.value = value
            self.incoming_messages = set()
            self.outgoing_messages = set()
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
                            value=torch.full((self.num_latent_codes,), 1.0 / self.num_latent_codes))

            self.factors.append(factor)
            self.variables = self.variables.union(set(factor.connected_vars))

        # add the marginal distribution of the sources
        # p(z^j_1), p(z^j_2), ..., p(z^j_n)
        for j in range(self.num_sources):
            for i in range(self.mixture_length):
                factor = Factor(type="source_marginal",
                                idx=i,
                                connected_vars=[
                                    Variable(classIdx=j, idx=i),
                                ],
                                value=torch.full((self.num_latent_codes,), 1.0 / self.num_latent_codes))
                self.factors.append(factor)
                self.variables = self.variables.union(
                    set(factor.connected_vars))
        # add the autoregressive priors
        # p(z^1_i | z^1_{i-1}), p(z^2_i | z^2_{i-1}), ..., p(z^k_i | z^k_{i-1})
        for j in range(self.num_sources):
            for i in range(self.mixture_length):
                # the first prior is null
                if i > 0:
                    factor = Factor(type="prior",
                                    idx=i,
                                    connected_vars=[Variable(classIdx=j, idx=i),
                                                    Variable(classIdx=j, idx=i-1)],
                                    value=torch.full(size=(self.num_latent_codes, 2), fill_value=1/self.num_latent_codes))
                    # value=torch.zeros((self.num_latent_codes, 2)))

                self.factors.append(factor)
                self.variables = self.variables.union(
                    set(factor.connected_vars))

        # add the posterior factors
        # p(z^j_i | m_i)
        for j in range(self.num_sources):
            for i in range(self.mixture_length):
                factor = Factor(type="posterior",
                                idx=i,
                                connected_vars=[
                                    Variable(classIdx=j, idx=i),
                                    Variable(classIdx=None, idx=i, mixture=True)],
                                value=torch.full(size=(self.mixture_length, self.mixture_length), fill_value=1/self.num_latent_codes))
                self.factors.append(factor)
                self.variables = self.variables.union(
                    set(factor.connected_vars))

        # add the likelihood factors, which don't depend on the indices
        # p(m| z^1, z^2, ..., z^k)
        for i in range(self.mixture_length):
            factor = SparseFactor(type="likelihood",
                                  idx=i,
                                  connected_vars=[
                                      Variable(classIdx=None,
                                               idx=i, mixture=True),
                                      *[Variable(classIdx=j, idx=i) for j in range(self.num_sources)]],
                                  value_coords=self._compute_sparse_likelihood(
                                      idx=i)[0],
                                  value_data=self._compute_sparse_likelihood(
                                      idx=i)[1])
            self.factors.append(factor)

        self.variables = self.variables.union(
            set(factor.connected_vars))

        assert self.variables == set(Variable._variables.values()), f"""
            The variables in the factor graph are not the same as the ones in the Variable class.
            The variables in the factor graph are: {self.variables}
            The variables in the Variable class are: {Variable._variables.values()}
        """

        ###########################
        # Initialize the messages #
        ###########################

        for factor in self.factors:

            for var in factor.connected_vars:
                message_in = Message(
                    _from=factor, _to=var, value=torch.zeros((self.num_latent_codes, )))
                message_out = Message(
                    _from=var, _to=factor, value=torch.zeros((self.num_latent_codes, )))

                # TODO: remove this trick
                for x in self.variables:
                    if x == var:
                        x.incoming_messages.add(message_in)
                        x.outgoing_messages.add(message_out)
                        x.neigh_factors.add(factor)

                for f in self.factors:
                    if f == factor:
                        f.incoming_messages.add(message_out)
                        f.outgoing_messages.add(message_in)

        # TODO: remove this trick
        # Trick for the variables
        # for factor in self.factors:
        #     log.debug(
        #         f"Factor {factor} has connected vars before the trick: {factor.connected_vars}")
        #     factor.connected_vars = [
        #         var for var in self.variables if var in factor.connected_vars]
        #     log.debug(
        #         f"Factor {factor} has connected vars after the trick: {factor.connected_vars}")

        total_connected_vars = set([
            var for fac in self.factors for var in fac.connected_vars])

        assert total_connected_vars == set(self.variables), f"""
            The variables in the factor graph are not the same as the ones in the Variable class.
            The variables in the factor graph are: {total_connected_vars}
            The variables in the Variable class are: {set(self.variables)}
        """

    def __repr__(self):
        return f"""FactorGraph(
            num_sources={self.num_sources},
            mixture={self.mixture},
            variables={self.variables},
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

    def _stack_with_padding(self, tensor_list: List[torch.Tensor]) -> torch.Tensor:

        assert len(tensor_list) > 0, "The list of tensors should not be empty"

        # Transform the tensors all to the same rank
        tensor_list = [t.unsqueeze(0) if len(t.shape) == 1 else t
                       for t in tensor_list]

        # Get the maximum shape in each dimension
        max_shape = torch.tensor([max([t.shape[i] for t in tensor_list])
                                  for i in range(len(tensor_list[0].shape))])

        # Create a list of padded tensors
        padded_tensors = [pad(t, (0, max_shape[1] - t.shape[1],
                              0, max_shape[0] - t.shape[0])) for t in tensor_list]

        # Stack the padded tensors along a new axis (if needed)
        return torch.stack(padded_tensors)

    def _compute_sparse_likelihood(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ll_coords, ll_data = self.likelihood.get_log_likelihood(
            self.mixture[idx])
        return ll_coords, ll_data

    def _compute_posterior(self, ll_data: torch.Tensor, ll_coords: torch.Tensor, priors: List[torch.Tensor]) -> torch.Tensor:

        prior_0, prior_1 = tuple(priors)

        posterior = _compute_log_posterior(
            log_p0=prior_0,
            log_p1=prior_1,
            nll_data=ll_data,
            nll_coords=ll_coords,
        )

        num_tokens = self.likelihood.get_tokens_count()  # 256 in the case of MNIST

        # Convert to shape (num samples, num_tokens*num_tokens)
        coords0, coords1 = ll_coords
        num_samples, _ = posterior.shape
        logits = torch.full(size=(num_samples, num_tokens**2),
                            fill_value=np.log(1e-16), device=torch.device("cpu"))
        logits.index_copy_(-1, coords0 * num_tokens + coords1, posterior)

        probs = torch.softmax(logits, dim=-1).unsqueeze(0)

        probs = probs.reshape(num_tokens, num_tokens)

        return probs

    # def _from_padded_stack_to_tensors(shapes: List[torch.Size], padded_stack: torch.Tensor) -> List[torch.Tensor]:
    #     extracted_tensors: List[torch.Tensor] = []
    #     for shape in shapes:
    #         # Get the original height and width
    #         original_h, original_w = shape
    #         # Extract the corresponding portion from the 'stacked_tensor'
    #         extracted = padded_stack[:, :original_h, :original_w]
    #         extracted_tensors.append(extracted)
    #     return extracted_tensors

    @timeit
    def belief_propagation(self, iterations: int = 30):
        """
        Run the sum-product algorithm for the given number of iterations in order to compute the marginal distributions.
        """

        for it in tqdm(range(iterations), desc="Belief Propagation"):
            # Update all unary factor-to-variable messages
            for variable in self.variables:
                for incoming_message in variable.incoming_messages:
                    factor = incoming_message._from
                    if factor.type == "mixture_marginal" or factor.type == "source_marginal":
                        incoming_message.value = factor.value

            # Update all factor-to-variable messages
            non_unary_factors = [
                fac for fac in self.factors if not (fac.type == "mixture_marginal" or fac.type == "source_marginal")]
            for factor in tqdm(non_unary_factors, desc="Factor-to-Variable messages"):
                if factor.type == "prior":
                    for fac_var in factor.connected_vars:
                        # NOTE: in the case of uncoditional priors, the past is always None
                        class_prior: UnconditionedTransformerPrior = self.priors[fac_var.classIdx]

                        past = self.pasts[fac_var.classIdx]

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

                        log_priors = normalize_logits(log_priors)

                        log.debug(
                            f"Log priors for {fac_var} are: {log_priors} with shape {log_priors.shape}")

                        self.pasts[fac_var.classIdx] = new_past

                        prob_priors = torch.softmax(log_priors, dim=-1)

                        updated_factor_value = prob_priors.squeeze(0)

                        assert not _check_if_nan_or_inf(
                            updated_factor_value), f"The updated factor value for {factor} value is nan during autoregressive prior computation"

                        factor.value[:,
                                     fac_var.classIdx] = updated_factor_value

                if factor.type == "likelihood":
                    # Compute the likelihood
                    ll_coords, ll_data = self._compute_sparse_likelihood(
                        factor.idx)
                    factor.value_coords = ll_coords
                    factor.value_data = ll_data

                if factor.type == "posterior":
                    # Compute the posterior
                    likelihood_factor: SparseFactor = [
                        fac for fac in self.factors if fac.type == "likelihood" and fac.idx == factor.idx][0]

                    prior_factor: Factor = [
                        fac for fac in self.factors if fac.type == "prior" and fac.idx == factor.idx]

                    if prior_factor == []:
                        priors = [torch.full((1, self.num_latent_codes), 1.0 / self.num_latent_codes),
                                  torch.full((1, self.num_latent_codes), 1.0 / self.num_latent_codes)]
                    else:
                        prior_factor = prior_factor[0]
                        priors = [prior_factor.value[:, 0].unsqueeze(0),
                                  prior_factor.value[:, 1].unsqueeze(0)]

                    # Convert the probabilities to log probabilities
                    priors = [torch.log(prior) for prior in priors]

                    ll_coords, ll_data = likelihood_factor.value_coords, likelihood_factor.value_data
                    updated_posterior = self._compute_posterior(
                        ll_data=ll_data, ll_coords=ll_coords, priors=priors)

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

                    log.debug(
                        f"Neigh variables to factor {factor}: {factor.connected_vars}")

                    variable_idx_in_factor = factor.connected_vars.index(
                        variable)

                    messages_from_var_to_factor = torch.stack([
                        message.value for variable in factor.connected_vars for message in variable.outgoing_messages if message._to == factor], dim=0)

                    # TODO: the messages are always all zeros
                    log.debug(
                        f"Messages from variables to factor {factor}: {messages_from_var_to_factor}")

                    messages_from_var_to_factor_sum = torch.sum(
                        messages_from_var_to_factor, dim=0)

                    updated_message = torch.log(
                        torch.sum(self._element_wise_prod_excluding_row(factor.value.T, torch.exp(messages_from_var_to_factor_sum), variable_idx_in_factor), dim=0) + 1e-10)

                    log.debug(
                        f"Updated message shape: {updated_message.shape}")

                    assert not _check_if_nan_or_inf(
                        updated_message), f"""
                        The updated message for {outgoing_message} is nan during factor-to-variable message computation.

                        For debug: 
                        factor.value: {factor.value}
                        messages_from_var_to_factor_sum: {messages_from_var_to_factor_sum}
                        exp: {torch.exp(messages_from_var_to_factor_sum)}
                        simple_prod: {factor.value.T * torch.exp(messages_from_var_to_factor_sum)}
                        element_wise_prod_excluding_row: {self._element_wise_prod_excluding_row(factor.value.T, torch.exp(messages_from_var_to_factor_sum), variable_idx_in_factor)}
                        sum: {torch.sum(self._element_wise_prod_excluding_row(factor.value.T, torch.exp(messages_from_var_to_factor_sum), variable_idx_in_factor), dim=0)}
                        updated_message (log): {updated_message}
                        """

                    outgoing_message.value = updated_message

                    _var_message_from_factor = [
                        message for message in variable.incoming_messages if message._from == factor]

                    assert (_var_message_from_factor[0].value == updated_message).all(), f"""
                    The updated message for {outgoing_message} is not the same as the incoming message for {variable} from {factor}.
                    """

            # Update all variable-to-factor messages
            for variable in self.variables:
                for outgoing_message in variable.outgoing_messages:
                    factor: Factor = outgoing_message._to

                    incoming_messages = torch.stack(
                        [message.value for message in variable.incoming_messages if message._from != factor], dim=0)

                    incoming_messages_sum = torch.sum(incoming_messages, dim=0)

                    assert not _check_if_nan_or_inf(
                        incoming_messages_sum), f"The updated message for {outgoing_message} is nan during variable-to-factor message computation"

                    outgoing_message.value = incoming_messages_sum

            # Calculate Marginals for each iteration
            for variable in self.variables:

                incoming_messages_stack = torch.stack(
                    [message.value for message in variable.incoming_messages])

                sum_incoming_messages_stack = torch.sum(
                    incoming_messages_stack, dim=0)

                variable.marginal = torch.softmax(
                    sum_incoming_messages_stack, dim=-1)

                log.debug(
                    f"Variable {variable} incoming messages are: {incoming_messages_stack} with shape: {incoming_messages_stack.shape}. The sum of the messages is {torch.sum(incoming_messages_stack, dim=0)}")

                print(
                    f"Marginals for the variable {variable} after iteration {it} are: {variable.marginal}")
        return


def _test_build_pattern():
    var1 = Variable(classIdx=0, idx=0)
    var2 = Variable(classIdx=0, idx=0)

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
        classIdx=0, idx=0), value=torch.zeros((1, 2)))

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

    factor_graph.belief_propagation(iterations=100)
    marginals: Dict[Variable, torch.Tensor] = {
        var: torch.softmax(var.marginal, dim=-1) for var in factor_graph.variables}
    print(marginals)
