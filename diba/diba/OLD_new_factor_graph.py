from __future__ import annotations
from enum import Enum
import time
from typing import Dict, List, Set, Tuple, NamedTuple
import torch
from tqdm import tqdm

from lass_mnist.lass.diba_interaces import DenseLikelihood, UnconditionedTransformerPrior
from transformers import GPT2LMHeadModel, GPT2Config

import logging
from abc import abstractmethod

# TODO: THIS IS TOO MEMORY DEMANDING


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
# Nodes (Variables and Factors)
#########################################


class Node:
    def __init__(self, name):
        self.name = name
        self.neighbors = []

    def __repr__(self):
        return "{classname}({name}, [{neighbors}])".format(
            classname=type(self).__name__,
            name=self.name,
            neighbors=', '.join([n.name for n in self.neighbors])
        )

    @abstractmethod
    def is_valid_neighbor(self, neighbor: Node) -> bool:
        return

    def add_neighbor(self, neighbor: Node):
        assert self.is_valid_neighbor(neighbor)
        self.neighbors.append(neighbor)


class Variable(Node):
    def is_valid_neighbor(self, factor: Factor):
        # Variables can only neighbor Factors
        return isinstance(factor, Factor)


class FactorType(Enum):
    PRIOR = "prior"
    POSTERIOR = "posterior"
    LIKELIHOOD = "likelihood"


class Factor(Node):
    def __init__(self, name: str, type: FactorType):
        super(Factor, self).__init__(name)
        self.data: LabeledTensor = None
        self.type = type

    def is_valid_neighbor(self, variable: Variable):
        # Factors can only neighbor Variables
        return isinstance(variable, Variable)


#########################################
# Factor Graph
#########################################

class FactorGraph:
    def __init__(self, num_sources: int, mixture: torch.Tensor, likelihood: DenseLikelihood):
        '''
        num_sources (int): number of sources
        mixture (torch.Tensor): shape (mixture_length)
        likelihood (DenseLikelihood): likelihood function
        '''
        self.num_sources = num_sources
        self.mixture = mixture
        self.likelihood = likelihood
        self.num_latent_variables = likelihood.get_tokens_count()
        self._factors: Dict[str, Factor] = {}
        self._variables: Dict[str, Variable] = {}
        self._create_graph()

    def variable_from_name(self, var_name: str) -> Variable:
        return self._variables[var_name]

    def _create_graph(self) -> Tuple[List[Factor], List[Variable]]:
        '''
        Returns:
            factors (List[Factor]): list of factors
            variables (List[Variable]): list of variables
        '''
        # Create variables
        for i in tqdm(range(self.num_latent_variables), desc="Creating z variables"):
            for j in range(self.num_sources):
                variable = Variable(f'z^{j}_{i}')
                self._variables[variable.name] = variable

        for i in tqdm(range(self.num_latent_variables), desc="Creating m variables"):
            variable = Variable(f'm_{i}')
            self._variables[variable.name] = variable

        # Create factors
        for i in tqdm(range(self.num_latent_variables), desc="Creating factors"):
            for j in range(self.num_sources):
                # Create prior factors
                if i == 0:
                    continue

                prior = Factor(
                    f"p(z^{j}_{i}) | z^{j}_{i-1}",
                    type=FactorType.PRIOR)

                prior_tensor = torch.ones(size=(self.num_latent_variables,
                                                self.num_latent_variables))
                prior.data = LabeledTensor(
                    tensor=prior_tensor / prior_tensor.sum(dim=0),
                    axes_labels=(
                        f'z^{j}_{i-1}', f'z^{j}_{i}'))

                variables = [self.variable_from_name(
                    f'z^{j}_{i-1}'), self.variable_from_name(f'z^{j}_{i}')]

                for variable in variables:
                    prior.add_neighbor(variable)
                    variable.add_neighbor(prior)

                assert is_conditional_prob(prior.data, f'z^{j}_{i}')
                self._factors[prior.name] = prior

                # TODO: Create posterior factors
                posterior = Factor(
                    f"p(z^{j}_{i}) | z^{j}_{i-1}, m_{i}",
                    type=FactorType.POSTERIOR)

                posterior_tensor = torch.ones(size=(self.num_latent_variables,
                                                    self.num_latent_variables,
                                                    self.num_latent_variables))
                posterior.data = LabeledTensor(
                    tensor=posterior_tensor / posterior_tensor.sum(dim=0),
                    axes_labels=(
                        f"m_{i}", f'z^{j}_{i}',  f'z^{j}_{i-1}',))

                variables = [
                    self.variable_from_name(f'z^{j}_{i-1}'),
                    self.variable_from_name(f'z^{j}_{i}'),
                    self.variable_from_name(f'm_{i}')]

                for variable in variables:
                    posterior.add_neighbor(variable)
                    variable.add_neighbor(prior)

                assert is_conditional_prob(prior.data, f'z^{j}_{i}')
                self._factors[posterior.name] = posterior

            # Create likelihood factors
            all_sources_names = [f'z^{j}_{i}' for j in range(
                self.num_sources)]
            likelihood = Factor(
                f"p(m_{i}) | {','.join(all_sources_names)}",
                type=FactorType.LIKELIHOOD)

            likelihood_tensor = torch.ones(size=tuple(
                self.num_latent_variables for _ in range(self.num_sources + 1)))

            likelihood.data = LabeledTensor(
                tensor=likelihood_tensor / likelihood_tensor.sum(dim=0),
                axes_labels=(
                    f'm_{i}', *all_sources_names))

            variables = [self.variable_from_name(f'm_{i}'), *[
                self.variable_from_name(f'z^{j}_{i}') for j in range(self.num_sources)]]

            for variable in variables:
                likelihood.add_neighbor(variable)
                variable.add_neighbor(likelihood)

            assert is_conditional_prob(likelihood.data, f'm_{i}')
            self._factors[likelihood.name] = likelihood

    def _get_factor_to_variable_message(self, factor: Factor, variable: Variable) -> torch.Tensor:
        '''
        factor (Factor)
        variable (Variable)
        '''
        pass


#########################################
# Main code
#########################################

if __name__ == "__main__":
    mixture = torch.tensor([210, 135, 8, 202, 135, 8, 56, 39, 63, 168, 149, 119, 70, 56, 137, 149, 93, 217, 217, 217, 8, 210, 66,
                            254, 26, 9, 168, 135, 210, 29, 26, 88, 222, 75, 210, 56, 56, 88, 4, 34, 80, 8, 56, 137, 75, 7, 41, 8, 56])
    # MIXTURE_LENGTH = 49

    SUMS_CHECKPOINT_PATH = "lass_mnist/checkpoints/sum/256-sigmoid-big.pt"
    with open(SUMS_CHECKPOINT_PATH, 'rb') as f:
        sums = torch.load(f, map_location=torch.device('cpu'))

    likelihood = DenseLikelihood(sums=sums)

    factor_graph = FactorGraph(
        num_sources=2, mixture=mixture, likelihood=likelihood)
