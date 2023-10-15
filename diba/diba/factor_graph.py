import time
from pathlib import Path
from typing import Dict, List, Set, Union
import torch
from tqdm import tqdm

from lass_mnist.lass.diba_interaces import UnconditionedTransformerPrior
from transformers import GPT2LMHeadModel, GPT2Config
from torch.nn.functional import pad


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

        # factor->variables messages list
        self.msg_fv: Set[Message] = set()
        # variables->factor messages list
        self.msg_vf: Set[Message] = set()

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

                self.msg_fv.add(message_in)
                self.msg_vf.add(message_out)

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
            factors={self.factors},
            msg_fv={self.msg_fv}, 
            msg_vf={self.msg_vf})"""

    def element_wise_prod_excluding_row(self, tensor1: torch.Tensor, tensor2: torch.Tensor, row: int) -> torch.Tensor:
        """
        Element-wise product of two tensors, excluding the given row
        """
        # Specify the row to exclude (let's say the first row, which is index 0)

        # Multiply the tensors element-wise
        result = tensor1 * tensor2

        # Remove the specified row using slicing
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
        for _ in tqdm(range(iterations), desc="Belief Propagation"):
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
                for outgoing_message in factor.outgoing_messages:
                    dprint(f"-------------------")
                    variable: Variable = outgoing_message._to
                    dprint(f"Factor: {factor}")
                    dprint(f"Variable: {variable}")
                    # TODO: try to avoid this for loop
                    neigh_variables = factor.connected_vars

                    dprint(
                        f"Neigh variables to factor {factor}: {neigh_variables}")

                    variable_idx_in_factor: int = factor.connected_vars.index(
                        variable)

                    messages_from_var_to_factor = torch.stack([
                        message.value for variable in neigh_variables for message in variable.outgoing_messages if message._to == factor], dim=0)

                    messages_from_var_to_factor_sum = torch.sum(
                        messages_from_var_to_factor, dim=0)

                    updated_message = torch.log(
                        torch.sum(self.element_wise_prod_excluding_row(factor.value.T, torch.exp(messages_from_var_to_factor_sum), variable_idx_in_factor), dim=0))

                    dprint(
                        f"Updates message shape: {updated_message.shape}")

            # Update all variable-to-factor messages
            for variable in self.variables:
                for outgoing_message in variable.outgoing_messages:
                    factor: Factor = outgoing_message._to

                    incoming_messages = torch.stack(
                        [message.value for message in variable.incoming_messages if message._from != factor], dim=0)

                    incoming_messages_sum = torch.sum(incoming_messages, dim=0)
                    outgoing_message.value = incoming_messages_sum

        # Calculate Marginals
        for variable in self.variables:
            variable.marginal = torch.sum(
                [message.value for message in variable.incoming_messages], dim=0)

        return


if __name__ == "__main__":
    factor_graph = FactorGraph(num_sources=2, mixture=torch.randn((256)))
    factor_graph.belief_propagation()
    marginals: Dict[Variable, torch.Tensor] = {
        var: var.marginal for var in factor_graph.variables}
    print(marginals)
    # logits_0, past = factor_graph.priors[0].get_logits(
    #     token_ids=torch.ones((256, 1), dtype=torch.long), past_key_values=None)

    # dprint(factor_graph.msg_fv)
