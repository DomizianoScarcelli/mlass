from typing import Dict, List, Mapping
import torch
import numpy as np
from diba.diba.diba import _compute_log_posterior
from diba.diba.utils import normalize_logits, unravel_indices
from lass_audio.jukebox.vqvae.bottleneck import BottleneckBlock

from lass_audio.jukebox.vqvae.vqvae import VQVAE
from lass_audio.lass.diba_interfaces import JukeboxPrior, SparseLikelihood

# Source: https://github.com/lorenlugosch/pytorch_HMM/blob/master/HMM.ipynb


class HMM(torch.nn.Module):
    """
    Hidden Markov Model with discrete observations.
    """

    def __init__(self, M, num_sources, priors: List[JukeboxPrior], vqvae: VQVAE, likelihood: SparseLikelihood, mixture_codes: List[int]):
        super(HMM, self).__init__()
        self.M = M  # number of possible observations (time steps)
        # number of states (all possible latent sequences in the codebook)
        self.num_sources = num_sources

        # List of JukeBoxPrior objects (one for each source)
        # that will be used to perform inference.
        self.priors = priors
        self.likelihood = likelihood
        self.mixture_codes = mixture_codes

        # A (transition matrix)
        self.transition_model = TransitionModel(self.num_sources)

        # b(x_t) P(m_t | z_t) (emission matrix)
        self.emission_model = EmissionModel(self.num_sources, self.M)

        inner_bottleneck: BottleneckBlock = vqvae.bottleneck.level_blocks[-1]

        self.discrete_codes: torch.Tensor = inner_bottleneck.k

        # This is (2048, 64), so everything's ok.
        print(f"Discrete codes shape: {self.discrete_codes.shape}")

        print(f"Discrete codes: {self.discrete_codes}")

        emission_matrix = torch.zeros(self.num_sources, self.M)

        logits_dict = {}

        for i, prior in enumerate(self.priors):
            logits, _ = prior.get_logits(torch.zeros((1, 1), dtype=torch.long))
            print(f"Logits shape for prior {i}: {logits.shape}")
            print(f"Logits for prior {i}: {logits}")
            # probabilities = torch.softmax(logits, dim=1)
            # print(f"Probabilities shape for prior {i}: {probabilities.shape}")
            # print(f"Probabilities for prior {i}: {probabilities}")

            # print(
            #     f"Argmax for prior {i}: {torch.argmax(probabilities)} with value {torch.max(probabilities)}")
            # emission_matrix[:, i] = probabilities.squeeze()
            # Mixture code is a list of length 1024

            normalized_logits = normalize_logits(logits)

            logits_dict[i] = normalized_logits

        # TODO: here it should be a for loop that iterates over the t time steps
        # NOTE: likelihood is p(m_t | z_0_t, z_1_t, z_n_t), while posterior is p(z_0_t, z_1_t, ..., z_n_t | m_t)
        ll_coords, ll_data = likelihood._get_log_likelihood(
            self.mixture_codes[0])  # TODO: make it depend on t instead of 0

        log_posterior = self.compute_log_posterior(
            ll_data, ll_coords, logits_dict)

        # NOTE: I think by doing this I am computing the naive generalization of the likelihood function P(m | z_0, z_1, ..., z_m), which I have to do in order to compare the performances.

        # NOTE: the indices the flattened indices from 0 to log_posterior. In top k they just get the indices of the topk elements
        indices = unravel_indices(torch.arange(
            0, log_posterior.shape[-1]), log_posterior.shape)[1]

        print(f"Indices are: {indices}")

        # sample = torch.distributions.Categorical(logits=log_posterior).sample()
        # print(f"Sample is {sample} with shape: {sample.shape}")

        x_0, x_1 = ll_coords[:, indices]

        print(f"x_0 is {x_0} with shape {x_0.shape}")
        print(f"x_1 is {x_1} with shape {x_1.shape}")

        # pi (initial state distribution)
        self.unnormalized_state_priors = torch.nn.Parameter(
            torch.randn(self.num_sources))

    def forward(self, x):
        pass

    def compute_log_posterior(
        self,
        nll_data: torch.Tensor,
        nll_coords: torch.LongTensor,
        logits_dict: Dict[int, torch.Tensor],
    ):
        coords_p0, coords_p1 = nll_coords
        # TODO: generalize for more than 2 sources
        return nll_data + logits_dict[0][:, coords_p0] + logits_dict[1][:, coords_p1]


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
