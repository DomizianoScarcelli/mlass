from typing import Dict, List, NamedTuple, Set, Tuple, Union
import numpy as np
import torch
from tqdm import tqdm
from diba.diba.utils import normalize_logits
from torchvision.utils import save_image

from lass_mnist.lass.diba_interaces import SparseLikelihood, UnconditionedTransformerPrior
from transformers import GPT2LMHeadModel, GPT2Config

import logging

from lass_mnist.lass.utils import ROOT_DIR


log = logging.getLogger("logger")
DEBUG = True
torch.set_printoptions(precision=2, sci_mode=False)

if DEBUG:
    logging.basicConfig(filename='factor_graph.log', level=logging.DEBUG,
                        format='%(name)s - %(levelname)s - %(message)s',
                        filemode='w')


def _check_if_nan_or_inf(tensor: torch.Tensor) -> bool:
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


def sample(mixture_t: int,
           priors: List[UnconditionedTransformerPrior],
           sources: int, past_z: torch.Tensor,
           partial_likelihoods: List[torch.Tensor],
           m_marginal: torch.Tensor,
           time_step: int = 0):
    samples_t = []
    for i in range(sources):
        partial_likelihood = partial_likelihoods[i]

        log_likelihood = partial_likelihood[:, mixture_t].flatten()

        assert not _check_if_nan_or_inf(log_likelihood), f"""
        Log likelihood is nan or inf
        Log likelihood is {log_likelihood}
        """

        curr_past = past_z[i].to(torch.long)

        log.debug(f"Curr past for i={i} is {curr_past}")

        assert curr_past.shape[0] == 256 and len(curr_past.shape) == 2, f"""
        Past z has wrong shape
        Past z has shape {curr_past.shape}
        """

        # print(
        #     f"For time_step t={time_step} shape of curr past is {curr_past.shape}")

        log_prior, _ = priors[i]._get_logits(
            x=curr_past,
            past_key_value=None,
        )

# og prior before normalization is tensor([[ -3.36, -17.99, -17.99,  ..., -18.01,  -5.29, -18.00],                                                                                           | 0/4 [00:00<?, ?it/s]
#         [ -1.87, -16.76, -16.76,  ..., -16.76,  -0.16, -16.75],
#         [ -0.65, -16.63, -16.60,  ..., -16.62,   0.47, -16.62],
#         ...,
#         [  2.29, -16.56, -16.58,  ..., -16.57,   2.23, -16.58],
#         [ -2.96, -18.59, -18.56,  ..., -18.56,   1.71, -18.56],
#         [  1.83, -16.62, -16.62,  ..., -16.58,   4.47, -16.62]])
# Log prior after normalization is tensor([[-10.16, -24.79, -24.79,  ..., -24.80, -12.09, -24.80],
#         [ -8.00, -22.88, -22.88,  ..., -22.88,  -6.28, -22.87],
#         [ -6.11, -22.09, -22.05,  ..., -22.08,  -4.98, -22.07],
#         ...,
#         [ -7.07, -25.92, -25.94,  ..., -25.93,  -7.13, -25.94],
#         [-11.31, -26.94, -26.91,  ..., -26.91,  -6.64, -26.91],
#         [ -9.50, -27.96, -27.95,  ..., -27.91,  -6.87, -27.95]])
        # if time_step == 10:
        # print(f"Log prior before normalization is {log_prior}")

        log_prior = normalize_logits(log_prior, temperature=1.0)

        # if time_step == 10:
        # print(f"Log prior after normalization is {log_prior}")
        # raise Exception("Stop here")
        assert log_prior.shape == torch.Size([256, 256]), f"""
        Log prior has wrong shape
        Log prior has shape {log_prior.shape}
        """

        LAMBDA = 1

        log_posterior = LAMBDA * log_likelihood + log_prior

        K = 32
        top_k_posterior, top_k_indices = torch.topk(log_posterior, k=K, dim=-1)
        softmaxed = torch.softmax(top_k_posterior, dim=-1)

        sample = []
        for i in range(log_posterior.shape[0]):
            sample.append(torch.tensor(
                np.random.choice(
                    top_k_indices[i].numpy(),
                    p=softmaxed[i].numpy())))  # TODO: I cannot insert logits here, but I don't know if softmax is correct
        # sample = torch.distributions.Categorical(logits=log_posterior).sample()
        # sample.append(curr_sample)

        sample = torch.stack(sample, dim=0)

        samples_t.append(sample)

    final_samples = torch.stack(samples_t, dim=0).unsqueeze(-1)

    return final_samples


def separate(mixture: torch.Tensor,
             likelihood: SparseLikelihood,
             transformer: GPT2LMHeadModel,
             sources: int):
    past = torch.zeros((sources, 256, 1))
    mixture_length = len(mixture)
    all_samples = past.detach().clone()
    partial_likelihoods = []
    priors = []
    for i in range(sources):
        partial_likelihood = likelihood.get_marginal_likelihood(
            source_idx=i)
        partial_likelihoods.append(partial_likelihood)
        priors.append(UnconditionedTransformerPrior(
            transformer=transformer, sos=0))

    m_marginal = likelihood.get_m_marginal_likelihood()
    for t in range(mixture_length):
        samples = sample(mixture_t=mixture[t],
                         priors=priors,
                         sources=sources,
                         past_z=past,
                         partial_likelihoods=partial_likelihoods,
                         m_marginal=m_marginal,
                         time_step=t)
        all_samples = torch.cat((all_samples, samples), dim=2)

        log.debug(f"New all samples shape is {all_samples.shape}")
        past = all_samples.detach().clone()

    log.debug(
        f"final all samples are {all_samples} with shape {all_samples.shape}")

    all_samples = all_samples[..., 1:].to(torch.long)
    if sources == 2:
        return all_samples[0], all_samples[1]
    if sources == 3:
        return all_samples[0], all_samples[1], all_samples[2]
    raise NotImplementedError(
        f"Separation for {sources} sources is not implemented")


if __name__ == "__main__":
    #############
    # Main code #
    #############
    mixture = torch.tensor([210, 135, 8, 202, 135, 8, 56, 39, 63, 168, 149, 119, 70, 56, 137, 149, 93, 217, 217, 217, 8, 210, 66,
                            254, 26, 9, 168, 135, 210, 29, 26, 88, 222, 75, 210, 56, 56, 88, 4, 34, 80, 8, 56, 137, 75, 7, 41, 8, 56])
    # MIXTURE_LENGTH = 49

    SUMS_CHECKPOINT_PATH = "lass_mnist/checkpoints/sum/3_sources_sums_epoch_430.pt"

    log.debug(f"Loading sparse sums")
    with open(SUMS_CHECKPOINT_PATH, 'rb') as f:
        sums = torch.load(f, map_location=torch.device('cpu'))

    transformer_config = GPT2Config(
        vocab_size=256,
        n_positions=len(mixture),
        n_embd=128,
        n_layer=3,
        n_head=2,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=511,)

    transformer = GPT2LMHeadModel(
        config=transformer_config)

    AUTOREGRESSIVE_CHECKPOINT_PATH = "lass_mnist/checkpoints/unconditioned/256-sigmoid-big.pt"
    with open(AUTOREGRESSIVE_CHECKPOINT_PATH, 'rb') as f:
        transformer.load_state_dict(
            torch.load(f, map_location=torch.device('cpu')))

    transformer.eval()

    likelihood = SparseLikelihood(sums=sums)

    # all_samples = separate(mixture=mixture, likelihood=likelihood,
    #                        transformer=transformer, sources=3)

    # sliced_likelihood = likelihood.get_log_likelihood(mixture[0])

    result_dir = ROOT_DIR / "multi-separated-images"

    z1, z2, z3 = separate(mixture=mixture, likelihood=likelihood,
                          transformer=transformer, sources=3)

    # save_image(z1, result_dir / f"sep/{0}-1.png")
    # save_image(z2, result_dir / f"sep/{1}-2.png")
    # save_image(z2, result_dir / f"sep/{2}-2.png")

    # log.debug(f"All samples are {all_samples} with shape {all_samples.shape}")
