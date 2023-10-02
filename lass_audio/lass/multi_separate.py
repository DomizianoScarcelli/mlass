import abc
import functools
from pathlib import Path
from typing import Callable, List, Mapping, Dict
import numpy as np

import torch
import torchaudio
from tqdm import tqdm
from diba.diba import Likelihood
from torch.utils.data import DataLoader
from diba.diba.diba import _ancestral_sample, _compute_log_posterior, _sample
from diba.diba.interfaces import SeparationPrior
from diba.diba.utils import get_topk, normalize_logits, unravel_indices

from lass_audio.lass.datasets import SeparationDataset
from lass_audio.lass.datasets import SeparationSubset
from lass_audio.lass.diba_interfaces import JukeboxPrior, SparseLikelihood
from lass_audio.lass.hidden_markov_model import HMM
from lass_audio.lass.utils import assert_is_audio, decode_latent_codes, get_dataset_subsample, get_raw_to_tokens, setup_priors, setup_vqvae
from lass_audio.lass.datasets import ChunkedMultipleDataset


audio_root = Path(__file__).parent.parent


class Separator(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def separate(mixture) -> Mapping[str, torch.Tensor]:
        ...


class SumProductSeparator(Separator):
    def __init__(
        self,
        encode_fn: Callable,
        decode_fn: Callable,
        priors: Mapping[str, SeparationPrior],
        likelihood: SparseLikelihood,
    ):
        super().__init__()
        self.likelihood = likelihood
        self.source_types = list(priors)
        self.priors = list(priors.values())

        # lambda x: vqvae.encode(x.unsqueeze(-1), vqvae_level, vqvae_level + 1).view(-1).tolist()
        self.encode_fn = encode_fn
        # lambda x: decode_latent_codes(vqvae, x.squeeze(0), level=vqvae_level)
        self.decode_fn = decode_fn
        self.device = torch.device("cpu")

    def initialize_graphical_model(self,
                                   dataset: SeparationDataset,
                                   num_sources: int) -> torch.Tensor:
        """
        Given the dataset and the number of sources to separate, it creates the Hidden Markov Model that models the problem.

        The HMM is represented as an adjacency list, where each node is a latent variable and each edge is a dependency between two latent variables.
        """

        # Step 1: Define the Directed Graphical Model
        # model = BayesianNetwork()

        loader = DataLoader(dataset, batch_size=1, num_workers=0)

        # models: Dict[int, BayesianNetwork] = {}

        batch: List[torch.Tensor]
        for batch_idx, batch in enumerate(tqdm(loader)):
            # Here I have two choices:
            # - One graph per tuple of sources, meaning num_batches total graphs, but small. This is good since the mixture only depends on the tuple of sources.
            # - One graph per dataset, meaning 1 single graph but huge. This is good in order to capture patterns.
            # Since the source influece only the mixture, and the priors capture other pieces of information needed, we will choose the first option.

            # Step 1: initialize the graphical model with the nodes.
            # These are used just to create the mixture, and then they correspond to the hidden variables, so they cannot be observed.
            sources = batch
            weight = 1 / num_sources

            self.mixture = torch.tensor(
                [(source * weight).tolist() for source in sources]).sum(dim=0).squeeze(0)

            # TODO: this is hard coded, but it corresponds to the max_sample_tokens in the separate, it's also present in the dataset
            # I think it's done in order to have batches of tokens of audio
            time_steps = 1024

            hmm = HMM(M=time_steps, N=2048)

            # STEP 2: For each single time step, compute the prior and the likelihood in order to compute the posterior probability.
            # TODO: generalize for more than 2 sources, now this is just for debug reasons

            # self.priors is a list of a neural network that expose a _get_logits method
            prior_0, prior_1 = self.priors

            # I encode the mixture to get its latent code
            mixture_codes = self.encode_fn(self.mixture)

            DEBUG_NUM_SOURCES = 2
            # TODO: this is used inside of beam_search and it corresponds to the number of beans to keep,
            # IDK what should be its value in this context. For now I will put it to 1, then I'll see if I can generalize it.
            DEBUG_NUM_CURRENT_BEAMS = 1

            # I initialize the latent codes for the two sources.
            # TODO: I think this whole procedure has to be done inside of the HMM class
            xs_0, xs_1 = torch.full((DEBUG_NUM_SOURCES, DEBUG_NUM_CURRENT_BEAMS, time_steps + 1),
                                    fill_value=-1, dtype=torch.long, device=self.device)

            # NOTE: in this case p.get_sos() returns the value 0
            xs_0[:, 0], xs_1[:, 0] = [p.get_sos() for p in self.priors]
            past_0, past_1 = (None, None)

            # NOTE: Xs_0 is: tensor([[ 0, -1, -1,  ..., -1, -1, -1]]) with shape torch.Size([DEBUG_NUM_CURRENT_BEAMS, 1025])
            # NOTE: Xs_1 is: tensor([[ 0, -1, -1,  ..., -1, -1, -1]]) with shape torch.Size([DEBUG_NUM_CURRENT_BEAMS, 1025])

            print(f"Xs_0 is: {xs_0} with shape {xs_0.shape}")
            print(f"Xs_0 is: {xs_1} with shape {xs_1.shape}")

            # TODO: Actually computing the prior and likelihood
            for t in tqdm(range(time_steps), desc="Loop over all the time steps"):
                CONTENT = xs_0[:DEBUG_NUM_CURRENT_BEAMS, : t + 1]
                PARSED_CONTENT = CONTENT[:, -1:]

                # TODO: DEBUG: The code breaks because I don't ever update xs_0 and xs_1, which are automatically updated in the beam search because of the auto-regressive nature of the model
                print(
                    f"At time step t={t} the content is: {CONTENT} with shape: {CONTENT.shape} \n The parsed content is {PARSED_CONTENT} with shape {PARSED_CONTENT.shape}")

                # compute log priors
                log_p_0, past_0 = prior_0._get_logits(
                    xs_0[:DEBUG_NUM_CURRENT_BEAMS, : t + 1], past_0)
                log_p_1, past_1 = prior_1._get_logits(
                    xs_1[:DEBUG_NUM_CURRENT_BEAMS, : t + 1], past_1)

                print(f"past_0 is: {past_0}")
                print(f"past_1 is: {past_1}")

                TEMPERATURE = 0.7

                # normalize priors and apply temperature
                log_p_0 = normalize_logits(log_p_0, TEMPERATURE)
                log_p_1 = normalize_logits(log_p_1, TEMPERATURE)

                # log likelihood in sparse COO format
                assert isinstance(self.mixture[t], int)
                ll_coords, ll_data = self.likelihood._get_log_likelihood(
                    self.mixture[t])

                # compute log posterior

                # TODO: make your changes here!
                if ll_coords.numel() > 0:
                    # Note: posterior_data has shape (n_samples, nonzeros)
                    posterior_data = _compute_log_posterior(
                        ll_data, ll_coords, log_p_0, log_p_1)

                    # TODO: here happens the sampling
                    log_post_sum, (beams, coords_idx) = get_topk(
                        log_post_sum + posterior_data, DEBUG_NUM_CURRENT_BEAMS)

                    log_post_sum = log_post_sum.unsqueeze(-1)
                    x_0, x_1 = ll_coords[:, coords_idx]
                else:
                    raise RuntimeError(
                        f"Code {self.mixture[t]} is not available in likelihood!")

            # Add the model to the list of models
            # models[batch_idx] = model

            # print(
            #     f"Length of the dataset and number of batches: {len(dataset)}")

            # print(
            #     f"model after step 1 is: {model}")

            # raise RuntimeError("STOP HERE MAN!")

            # TODO:
            # Step 4: Perform Inference and Sample each source z_i from P(z_i | m)
            # inference = VariableElimination(model)

            # for i in range(num_sources):  # Assuming you know the number of sources
            #     # Assuming the evidence is observed as 1
            #     evidence = {f"z_{i+1}": 1}
            #     result = inference.map_query(
            #         variables=[f"z_{i+1}"], evidence=evidence)
            #     sampled_z_i = result[f"z_{i+1}"]

            # Now 'sampled_z_i' contains the sampled value of 'z_i' from P(z_i | m) while marginalizing out other z_j's.

            # You can repeat this step for all sources to sample each one individually.

            # self._print_graph(model)

    @torch.no_grad()
    def separate(self) -> Mapping[str, torch.Tensor]:

        mixture_codes = self.encode_fn(self.mixture)

        print(
            f"Mixture codes are: {mixture_codes} with shape: {torch.tensor(mixture_codes).shape}")

        mixture_len = len(self.mixture)

        # Prior = P(z_i)

        # Conditional Probability = P(z_i | m)

        # Likelihood = P(m | z_i)

        # Since len(z_i) and len(m) = 1024, all the probabilities are also torch.Tensors of length 1024

        CONSIDERED_TOKEN_IDX = 1
        # This is the frequency count tensor of rank K ^ {num_sources + 1} (3 in case of 2 sources).
        ll_coords, ll_data = self.likelihood._get_log_likelihood(
            x=self.mixture[0, CONSIDERED_TOKEN_IDX])  # TODO: don't know what this is lol


# -----------------------------------------------------------------------------


@torch.no_grad()
def separate_dataset(
    dataset: SeparationDataset,
    separator: Separator,
    save_path: str,
    save_fn: Callable,
    resume: bool = False,
    num_workers: int = 0,
):
    # convert paths
    save_path = Path(save_path)
    if not resume and save_path.exists() and not len(list(save_path.glob("*"))) == 0:
        raise ValueError(f"Path {save_path} already exists!")

    # get samples
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

    # main loop
    save_path.mkdir(exist_ok=True)
    for batch_idx, batch in enumerate(tqdm.tqdm(loader)):
        chunk_path = save_path / f"{batch_idx}"
        if chunk_path.exists():
            print(f"Skipping path: {chunk_path}")
            continue

        # load audio tracks
        origs = batch
        ori_1, ori_2 = origs
        print(f"chunk {batch_idx+1} out of {len(dataset)}")

        # generate mixture
        mixture = 0.5 * ori_1 + 0.5 * ori_2
        mixture = mixture.squeeze(0)  # shape: [1 , sample-length]
        seps = separator.separate(mixture=mixture)
        chunk_path.mkdir(parents=True)

        # save separated audio
        save_fn(
            separated_signals=[sep.unsqueeze(0) for sep in seps.values()],
            original_signals=[ori.squeeze(0) for ori in origs],
            path=chunk_path,
        )
        print(f"chunk {batch_idx+1} saved!")
        del seps, origs


# -----------------------------------------------------------------------------


def save_separation(
    separated_signals: List[torch.Tensor],
    original_signals: List[torch.Tensor],
    sample_rate: int,
    path: Path,
):
    assert_is_audio(*original_signals, *separated_signals)
    # assert original_1.shape == original_2.shape == separation_1.shape == separation_2.shape
    assert len(original_signals) == len(separated_signals)
    for i, (ori, sep) in enumerate(zip(original_signals, separated_signals)):
        print(ori.shape, sep.shape)
        torchaudio.save(str(path / f"ori{i+1}.wav"),
                        ori.cpu(), sample_rate=sample_rate)
        torchaudio.save(str(path / f"sep{i+1}.wav"),
                        sep.cpu(), sample_rate=sample_rate)
        break


def main(
    audio_dir_1: str = audio_root / "data/bass",
    audio_dir_2: str = audio_root / "data/drums",
    vqvae_path: str = audio_root / "checkpoints/vqvae.pth.tar",
    prior_1_path: str = audio_root / "checkpoints/prior_bass_44100.pth.tar",
    prior_2_path: str = audio_root / "checkpoints/prior_drums_44100.pth.tar",
    sum_frequencies_path: str = audio_root / "checkpoints/sum_frequencies.npz",
    vqvae_type: str = "vqvae",
    prior_1_type: str = "small_prior",
    prior_2_type: str = "small_prior",
    max_sample_tokens: int = 1024,
    sample_rate: int = 44100,
    save_path: str = audio_root / "separated-audio",
    resume: bool = True,
    num_pairs: int = 100,
    seed: int = 0,
    **kwargs,
):
    # convert paths
    save_path = Path(save_path)
    audio_dir_1 = Path(audio_dir_1)
    audio_dir_2 = Path(audio_dir_2)

    # if not resume and save_path.exists():
    #    raise ValueError(f"Path {save_path} already exists!")

    # rank, local_rank, device = setup_dist_from_mpi(port=29533, verbose=True)
    device = torch.device("cpu")

    # setup models
    vqvae = setup_vqvae(
        vqvae_path=vqvae_path,
        vqvae_type=vqvae_type,
        sample_rate=sample_rate,
        sample_tokens=max_sample_tokens,
        device=device,
    )

    priors = setup_priors(
        prior_paths=[prior_1_path, prior_2_path],
        prior_types=[prior_1_type, prior_2_type],
        vqvae=vqvae,
        fp16=True,
        device=device,
    )
    priors = {
        Path(prior_1_path).stem: priors[0],
        Path(prior_2_path).stem: priors[1],
    }

    # create separator
    level = vqvae.levels - 1
    separator = SumProductSeparator(
        encode_fn=lambda x: vqvae.encode(
            x.unsqueeze(-1).to(device), level, level + 1)[-1].squeeze(0).tolist(),  # TODO: check if correct
        decode_fn=lambda x: decode_latent_codes(
            vqvae, x.squeeze(0), level=level),
        priors={k: JukeboxPrior(p.prior, torch.zeros(
            (), dtype=torch.float32, device=device)) for k, p in priors.items()},
        likelihood=SparseLikelihood(sum_frequencies_path, device, 3.0),
        **kwargs,
    )

    graph = separator.initialize_graphical_model()
    # separator._print_graph(graph)

    # TODO: Early return for debugging
    return

    # setup dataset
    raw_to_tokens = get_raw_to_tokens(vqvae.strides_t, vqvae.downs_t)
    dataset = ChunkedMultipleDataset(
        instruments_audio_dir=[audio_dir_1, audio_dir_2],
        sample_rate=sample_rate,
        max_chunk_size=raw_to_tokens * max_sample_tokens,
        min_chunk_size=raw_to_tokens,
    )

    # subsample the test dataset
    indices = get_dataset_subsample(len(dataset), num_pairs, seed=seed)
    subdataset = SeparationSubset(dataset, indices=indices)

    # separate subsample
    separate_dataset(
        dataset=subdataset,
        separator=separator,
        save_path=save_path,
        save_fn=functools.partial(save_separation, sample_rate=sample_rate),
        resume=resume,
    )


if __name__ == "__main__":
    main()
