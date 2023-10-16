import time
from typing import Dict, List, Set, Union
import torch
from tqdm import tqdm

from lass_mnist.lass.diba_interaces import UnconditionedTransformerPrior
from transformers import GPT2LMHeadModel, GPT2Config
from torch.nn.functional import pad

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


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        dprint(f"{func.__name__} took {end_time - start_time} seconds.")
        return result
    return wrapper


DEBUG = False


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def _check_if_nan(tensor: torch.Tensor) -> bool:
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


class Variable:
    def __init__(self, classIdx: Union[int, None], idx: int, mixture: bool = False):
        self.classIdx = classIdx
        self.idx = idx
        self.mixture = mixture
        self.neigh_factors: Set[Factor] = set()
        self.outgoing_messages: Set[Message] = set()
        self.incoming_messages: Set[Message] = set()
        self.marginal: torch.Tensor = None

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Variable):
            return self.classIdx == __value.classIdx and self.idx == __value.idx
        return False

    def __hash__(self):
        return hash((self.classIdx, self.idx, self.mixture))

    def __repr__(self):
        if self.mixture:
            return f"m_{self.idx}"
        return f"z^{self.classIdx}_{self.idx}"


class Factor:
    def __init__(self, type: str, idx: int, connected_vars: List[Variable], value: torch.Tensor):
        self.type = type
        self.idx = idx
        self.connected_vars = connected_vars
        self.value = value
        self.incoming_messages: Set[Message] = set()
        self.outgoing_messages: Set[Message] = set()

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Factor):
            return self.type == __value.type and self.idx == __value.idx and self.connected_vars == __value.connected_vars
        return False

    def __hash__(self):
        return hash((self.type, self.idx, tuple(self.connected_vars)))

    def __repr__(self):
        return f"Factor(type={self.type}, idx={self.idx}, connected_vars={self.connected_vars})"


class Message:
    def __init__(self, _from: Union[Factor, Variable], _to: Union[Factor, Variable], value: torch.Tensor):
        self._from = _from
        self._to = _to
        self.value = value

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Message):
            return self._from == __value._from and self._to == __value._to
        return False

    def __hash__(self):
        return hash((self._from, self._to))

    def __repr__(self):
        return f"Message(from={self._from}, to={self._to}, value_shape={self.value.shape})"


class FactorGraph:
    @timeit
    def __init__(self, num_sources: int, mixture: torch.Tensor) -> None:
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
        self.pasts = [None for _ in range(self.num_sources)]

        print(f"Length of the mixture is: {self.mixture_length}")

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
            transformer=self.transformer, sos=0) for _ in range(self.num_sources)]

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
                                value=torch.full(size=(self.num_latent_codes, 2), fill_value=1/self.num_latent_codes))
                self.factors.append(factor)
                self.variables = self.variables.union(
                    set(factor.connected_vars))

        # add the likelihood factors, which don't depend on the indices
        # p(m| z^1, z^2, ..., z^k)
        likelihood_factor = Factor(type="likelihood",
                                   idx=None,
                                   connected_vars=[
                                       *[Variable(classIdx=None,
                                                  idx=i, mixture=True) for i in range(self.mixture_length)],
                                       *[Variable(classIdx=j, idx=i) for j in range(self.num_sources) for i in range(self.mixture_length)]],
                                   value=None)
        likelihood_factor.value = torch.full(
            size=(self.num_latent_codes, len(likelihood_factor.connected_vars)), fill_value=1/self.num_latent_codes)
        self.factors.append(likelihood_factor)
        self.variables = self.variables.union(
            set(likelihood_factor.connected_vars))

        ###########################
        # Initialize the messages #
        ###########################

        for factor in self.factors:

            for var in factor.connected_vars:
                message_in = Message(
                    _from=factor, _to=var, value=torch.zeros((self.num_latent_codes, )))
                message_out = Message(
                    _from=var, _to=factor, value=torch.zeros((self.num_latent_codes, )))

                for x in self.variables:
                    if x == var:
                        x.incoming_messages.add(message_in)
                        x.outgoing_messages.add(message_out)
                        x.neigh_factors.add(factor)

                for f in self.factors:
                    if f == factor:
                        f.incoming_messages.add(message_out)
                        f.outgoing_messages.add(message_in)

        # Trick for the variables
        for factor in self.factors:
            factor.connected_vars = [
                var for var in self.variables if var in factor.connected_vars]

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

    def _from_padded_stack_to_tensors(shapes: List[torch.Size], padded_stack: torch.Tensor) -> List[torch.Tensor]:
        extracted_tensors: List[torch.Tensor] = []
        for shape in shapes:
            # Get the original height and width
            original_h, original_w = shape
            # Extract the corresponding portion from the 'stacked_tensor'
            extracted = padded_stack[:, :original_h, :original_w]
            extracted_tensors.append(extracted)
        return extracted_tensors

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

                        dprint(f"Factor value idx is: {fac_var.idx}")
                        dprint(
                            f"Mixture is {self.mixture} with shape: {self.mixture.shape}")
                        observed_mixture = self.mixture[fac_var.idx].unsqueeze(
                            0).unsqueeze(0).to(torch.long)

                        dprint(
                            f"Obseved mixture shape is: {observed_mixture.shape}")

                        log_priors, new_past = class_prior.get_logits(
                            token_ids=observed_mixture,
                            past_key_values=past)

                        # TODO: there is a problem here, since new past is always None
                        if new_past is not None:
                            dprint(f"Past is: {past}")
                            dprint(f"New past is: {new_past}")
                            raise Exception("Stop here")

                        self.pasts[fac_var.classIdx] = new_past

                        prob_priors = torch.softmax(log_priors, dim=-1)

                        updated_factor_value = prob_priors.squeeze(0)

                        assert not _check_if_nan(
                            updated_factor_value), f"The updated factor value for {factor} value is nan during autoregressive prior computation"

                        factor.value[:,
                                     fac_var.classIdx] = updated_factor_value

                for outgoing_message in factor.outgoing_messages:
                    dprint(f"-------------------")
                    # TODO: strange trick, I should find a better way to do this
                    temp_var: Variable = outgoing_message._to
                    variable = [
                        var for var in self.variables if var == temp_var][0]
                    dprint(f"Factor: {factor}")
                    dprint(f"Temp var: {temp_var}")
                    dprint(f"Variable: {variable}")

                    assert variable in factor.connected_vars, f"""
                        The variable {variable} is not connected to the factor {factor}.
                        The connected variables are: {factor.connected_vars}
                    """

                    neigh_variables = factor.connected_vars

                    dprint(
                        f"Neigh variables to factor {factor}: {neigh_variables}")

                    variable_idx_in_factor = neigh_variables.index(variable)

                    messages_from_var_to_factor = torch.stack([
                        message.value for variable in neigh_variables for message in variable.outgoing_messages if message._to == factor], dim=0)

                    messages_from_var_to_factor_sum = torch.sum(
                        messages_from_var_to_factor, dim=0)

                    updated_message = torch.log(
                        torch.sum(self._element_wise_prod_excluding_row(factor.value.T, torch.exp(messages_from_var_to_factor_sum), variable_idx_in_factor), dim=0) + 1e-10)

                    dprint(
                        f"Updates message shape: {updated_message.shape}")

                    assert not _check_if_nan(
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

                    # TODO: solve this problem
                    assert len(_var_message_from_factor) != 0, f"""
                        There is no message from {factor} to {variable}.
                        Even though {outgoing_message} exists
                    """

                    assert (_var_message_from_factor[0].value == updated_message).all(), f"""
                    The updated message for {outgoing_message} is not the same as the incoming message for {variable} from {factor}.
                    """

            # Update all variable-to-factor messages
            for variable in self.variables:
                for outgoing_message in variable.outgoing_messages:
                    temp_factor: Factor = outgoing_message._to
                    factor = [fac for fac in self.factors if fac == temp_factor]

                    incoming_messages = torch.stack(
                        [message.value for message in variable.incoming_messages if message._from != factor], dim=0)

                    incoming_messages_sum = torch.sum(incoming_messages, dim=0)

                    assert not _check_if_nan(
                        incoming_messages_sum), f"The updated message for {outgoing_message} is nan during variable-to-factor message computation"

                    outgoing_message.value = incoming_messages_sum

            # Calculate Marginals for each iteration
            for variable in self.variables:
                variable.marginal = torch.sum(
                    torch.stack([message.value for message in variable.incoming_messages]), dim=0)

            print(
                f"""Marginals for a chosen variable z_0_1 after iteration {it} are: {
                    [variable for variable in self.variables if variable.classIdx == 0 and variable.idx == 1][0].marginal
                }""")
        return


if __name__ == "__main__":
    # MIXTURE_LENGTH = 49
    mixture = torch.tensor([210, 135, 8, 202, 135, 8, 56, 39, 63, 168, 149, 119, 70, 56, 137, 149, 93, 217, 217, 217, 8, 210, 66,
                            254, 26, 9, 168, 135, 210, 29, 26, 88, 222, 75, 210, 56, 56, 88, 4, 34, 80, 8, 56, 137, 75, 7, 41, 8, 56])
    factor_graph = FactorGraph(
        num_sources=2, mixture=mixture)
    factor_graph.belief_propagation(iterations=100)
    marginals: Dict[Variable, torch.Tensor] = {
        var: torch.softmax(var.marginal, dim=-1) for var in factor_graph.variables}
    print(marginals)
