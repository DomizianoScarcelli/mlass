import torch
from diba.diba.interfaces import Likelihood, SeparationPrior
from typing import Sequence

from lass_mnist.lass.diba_interaces import DenseLikelihood, UnconditionedTransformerPrior

# TODO: move this inside diba.diba, for DEBUG reasons I put this here


def graphical_model_separation(
    priors: Sequence[SeparationPrior],
    likelihood: Likelihood,
    mixture: Sequence[int],
    latent_length: int,
    num_sources: int,
):
    """Separate the input mixture using graphical models.

    Args:
        priors: List of priors to use for separation
        likelihood: Likelihood to use for separation
        mixture: Sequence of tokens to separate
        latent_length: Length of the latent space
        num_sources: Number of sources to separate

    Returns:
        List of separated sources, one for each source
    """
    pass
