import torch
import numpy as np

# Source: https://github.com/lorenlugosch/pytorch_HMM/blob/master/HMM.ipynb


class HMM(torch.nn.Module):
    """
    Hidden Markov Model with discrete observations.
    """

    def __init__(self, M, N, priors):
        super(HMM, self).__init__()
        self.M = M  # number of possible observations (time steps)
        # number of states (all possible latent sequences in the codebook)
        self.N = N

        self.priors = priors

        # A (transition matrix)
        self.transition_model = TransitionModel(self.N)

        # b(x_t) P(m_t | z_t) (emission matrix)
        self.emission_model = EmissionModel(self.N, self.M)

        # TODO: for now we have only 2 sources, but this has to be generalized to a list of n sources.
        prior_0, prior_1 = self.priors

        # pi (initial state distribution)
        self.unnormalized_state_priors = torch.nn.Parameter(
            torch.randn(self.N))

    def forward(self, x):
        pass


class TransitionModel(torch.nn.Module):
    def __init__(self, N):
        super(TransitionModel, self).__init__()
        self.N = N
        self.unnormalized_transition_matrix = torch.nn.Parameter(
            torch.zeros(N, N))


class EmissionModel(torch.nn.Module):
    def __init__(self, N, M):
        super(EmissionModel, self).__init__()
        self.N = N
        self.M = M
        self.unnormalized_emission_matrix = torch.nn.Parameter(
            torch.zeros(N, M))
