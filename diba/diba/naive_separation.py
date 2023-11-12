from typing import Dict, List, NamedTuple, Set, Tuple, Union
import numpy as np
import torch
from tqdm import tqdm
from diba.diba.utils import normalize_logits

from lass_mnist.lass.diba_interaces import DenseMarginalLikelihood, UnconditionedTransformerPrior
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
           likelihood: torch.Tensor):
    samples_t = []
    for i in range(sources):
        partial_likelihood = likelihood[i]

        log_likelihood = partial_likelihood[:, mixture_t].view(-1)

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

        log_prior, _ = priors[i].get_logits(
            token_ids=curr_past,
            past_key_values=None,
        )

        log_prior = normalize_logits(log_prior)

        assert log_prior.shape == torch.Size([256, 256]), f"""
        Log prior has wrong shape
        Log prior has shape {log_prior.shape}
        """

        LAMBDA = 1

        log_posterior = LAMBDA * log_likelihood + log_prior

        K = 32
        top_k_posterior, top_k_indices = torch.topk(log_posterior, k=K, dim=-1)

        softmaxed_top_k_posterior = torch.softmax(top_k_posterior, dim=-1)

        sample = []
        for i in range(log_posterior.shape[0]):
            sample.append(torch.tensor(
                np.random.choice(
                    top_k_indices[i].detach().numpy(),
                    p=softmaxed_top_k_posterior[i].detach().numpy())))

        sample = torch.stack(sample, dim=0)

        samples_t.append(sample)

    final_samples = torch.stack(samples_t, dim=1).unsqueeze(-1)

    log.debug(f"Final samples shape is {final_samples.shape}")

    return final_samples.permute(1, 0, 2)


def separate(mixture: torch.Tensor,
             likelihood: torch.Tensor,
             transformer: GPT2LMHeadModel,
             sources: int):
    past = torch.zeros((sources, 256, 1))
    mixture_length = len(mixture)
    all_samples = past
    priors = []
    for _ in range(sources):
        priors.append(UnconditionedTransformerPrior(
            transformer=transformer, sos=0))

    for t in tqdm(range(mixture_length), desc="Separating mixture"):
        samples = sample(mixture_t=mixture[t],
                         priors=priors,
                         sources=sources,
                         past_z=past,
                         likelihood=likelihood)
        all_samples = torch.cat((all_samples, samples), dim=2)

        log.debug(f"New all samples shape is {all_samples.shape}")
        past = all_samples

    log.debug(
        f"final all samples are {all_samples} with shape {all_samples.shape}")

    all_samples = all_samples[..., 1:].to(torch.long)
    if sources == 2:
        return all_samples[0], all_samples[1]
    if sources == 3:
        return all_samples[0], all_samples[1], all_samples[2]
    raise NotImplementedError(
        f"Separation for {sources} sources is not implemented")


# if __name__ == "__main__":
#     #############
#     # Main code #
#     #############
#     mixture = torch.tensor([210, 135, 8, 202, 135, 8, 56, 39, 63, 168, 149, 119, 70, 56, 137, 149, 93, 217, 217, 217, 8, 210, 66,
#                             254, 26, 9, 168, 135, 210, 29, 26, 88, 222, 75, 210, 56, 56, 88, 4, 34, 80, 8, 56, 137, 75, 7, 41, 8, 56])
#     # MIXTURE_LENGTH = 49

#     SUMS_CHECKPOINT_PATH = "lass_mnist/checkpoints/sum/best_3_sources.pt"

#     log.debug(f"Loading sparse sums")
#     with open(SUMS_CHECKPOINT_PATH, 'rb') as f:
#         sums = torch.load(f, map_location=torch.device('cpu'))

#     transformer_config = GPT2Config(
#         vocab_size=256,
#         n_positions=len(mixture),
#         n_embd=128,
#         n_layer=3,
#         n_head=2,
#         use_cache=False,
#         bos_token_id=0,
#         eos_token_id=511,)

#     transformer = GPT2LMHeadModel(
#         config=transformer_config)

#     AUTOREGRESSIVE_CHECKPOINT_PATH = "lass_mnist/checkpoints/unconditioned/256-sigmoid-big.pt"
#     with open(AUTOREGRESSIVE_CHECKPOINT_PATH, 'rb') as f:
#         transformer.load_state_dict(
#             torch.load(f, map_location=torch.device('cpu')))

#     transformer.eval()

#     likelihood = DenseMarginalLikelihood(sums=sums)

#     result_dir = ROOT_DIR / "multi-separated-images"

#     z1, z2, z3 = separate(mixture=mixture, likelihood=likelihood,
#                           transformer=transformer, sources=3)

#     print(f"z1 is {z1}")
#     print(f"z2 is {z2}")
#     print(f"z3 is {z3}")
