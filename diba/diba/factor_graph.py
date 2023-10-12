from typing import Dict, List, Mapping
import numpy as np


class FactorGraph():
    def __init__(self, num_classes: int, mixture: np.ndarray) -> None:
        """
        Initialize the factor graph
        @params num_classes: number of sources to separate 
        @params mixture: the latent code of the mixture to separate
        """
        self.num_classes = num_classes  # 'k' in the equations
        self.mixture = mixture  # 'm' in the equations
        self.mixture_length = len(mixture)  # 'n' in the equations
        self.num_latent_codes = 256  # 256 in the case of MNIST

        # list of dictionaries, each dictionary is a factor
        self.factors = []

        # initialize the factors
        # add the marginal distribution of the mixture

        # p(m_1), p(m_2), ..., p(m_n)
        for i in range(self.mixture_length):
            self.factors.append({
                'type': 'mixture_marginal',
                'index': i,
                "class": None,
                'value': np.zeros((1, self.num_latent_codes)),
            })

        # add the marginal distribution of the sources
        # p(z^j_1), p(z^j_2), ..., p(z^j_n)
        for j in range(self.num_classes):
            for i in range(self.mixture_length):
                self.factors.append({
                    'type': 'source_marginal',
                    'index': i,
                    "class": j,
                    'value': np.zeros((1, self.num_latent_codes)),
                })

        # add the autoregressive priors
        # p(z^1_i | z^1_{i-1}), p(z^2_i | z^2_{i-1}), ..., p(z^k_i | z^k_{i-1})
        for i in range(self.mixture_length):
            self.factors.append({
                'type': 'prior',
                'index': i,
                'value': np.zeros((1, self.num_latent_codes)),
            })

        # add the likelihood factors, which don't depend on the indices
        # p(m| z^1, z^2, ..., z^k)
        self.factors.append({
            'type': 'likelihood',
            'index': None,
            "value": np.zeros(tuple(self.num_latent_codes for _ in range(self.num_classes))),
        })


if __name__ == "__main__":
    fg = FactorGraph(num_classes=2, mixture=np.array([1, 2, 3, 4, 5]))

    print(fg.factors)
