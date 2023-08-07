import abc
import functools
from pathlib import Path
from typing import Callable, List, Mapping

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from diba.diba import Likelihood
from torch.utils.data import DataLoader
from diba.diba.interfaces import SeparationPrior

from lass_audio.lass.datasets import SeparationDataset
from lass_audio.lass.datasets import SeparationSubset
from lass_audio.lass.diba_interfaces import JukeboxPrior, SparseLikelihood
from lass_audio.lass.utils import assert_is_audio, decode_latent_codes, get_dataset_subsample, get_raw_to_tokens, setup_priors, setup_vqvae
from lass_audio.lass.datasets import ChunkedMultipleDataset


from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

import pandas as pd

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
        likelihood: Likelihood,
    ):
        super().__init__()
        self.likelihood = likelihood
        self.source_types = list(priors)
        self.priors = list(priors.values())

        # lambda x: vqvae.encode(x.unsqueeze(-1), vqvae_level, vqvae_level + 1).view(-1).tolist()
        self.encode_fn = encode_fn
        # lambda x: decode_latent_codes(vqvae, x.squeeze(0), level=vqvae_level)
        self.decode_fn = decode_fn

    def _print_graph(self, graph: DBN):
        print("Nodes in the graph:")
        print(graph.nodes())

        print("\nEdges in the graph:")
        print(graph.edges())

        # Print the conditional probability distributions (CPDs) for each node in the graph
        for node in graph.nodes():
            cpd = graph.get_cpds(node)
            print(f"\nCPD for node '{node}':")
            print(cpd)

    def initialize_graphical_model(self,
                                   dataset: SeparationDataset,
                                   num_sources: int,
                                   time_steps: int) -> DBN:

        # Step 1: Define the Directed Graphical Model
        model = BayesianNetwork()

        loader = DataLoader(dataset, batch_size=1, num_workers=0)

        for batch_idx, batch in enumerate(tqdm(loader)):
            origs = batch
            # TODO: generalize for more than 2 sources, using the MultiDatasets
            ori_1, ori_2 = origs
            mixture = (0.5 * ori_1 + 0.5 * ori_2).squeeze(0)

            # TODO: add nodes and edges for the graph here

        # Step 2: Add nodes and edges to the model based on the dependencies between variables

        # In our case, we will connect each z_i^{t-1} to z_i^{t}, and each z_i to m.
        sources_nodes = [f"z_{i}_{t}" for i in range(
            num_sources) for t in range(time_steps)]
        mixture_nodes = [f"m_{t}" for t in range(time_steps)]

        model.add_nodes_from(sources_nodes + mixture_nodes)

        # Add edges for t_i_{t-1} -> t_i_{t} (Temporal dependency for the latent sources)
        for i in range(num_sources):
            for t in range(time_steps):
                if t == time_steps:
                    break
                start = f"z_{i}_{t}"
                end = f"z_{i}_{t+1}"

                model.add_edge(start, end)

        # print("Model edges: ", model.edges())

        # Add edges for m_{t-1} -> m_{t} (Temporal dependency for the mixture signal)
        for t in range(time_steps):
            if t == time_steps:
                break
            start = f"m_{t}"
            end = f"m_{t+1}"

            model.add_edge(start, end)

        # TODO:
        # Step 3: Parameterize the graph with prior and likelihood values
        # You'll need to define Conditional Probability Distributions (CPDs) for each node in the graph
        # based on your prior probabilities and likelihood values. Use MaximumLikelihoodEstimator or other
        # methods to estimate CPDs from your data.

        # TODO:
        # Step 4: Perform Inference and Sample each source z_i from P(z_i | m)
        inference = VariableElimination(model)

        for i in range(num_sources):  # Assuming you know the number of sources
            # Assuming the evidence is observed as 1
            evidence = {f"z_{i+1}": 1}
            result = inference.map_query(
                variables=[f"z_{i+1}"], evidence=evidence)
            sampled_z_i = result[f"z_{i+1}"]

            # Now 'sampled_z_i' contains the sampled value of 'z_i' from P(z_i | m) while marginalizing out other z_j's.

            # You can repeat this step for all sources to sample each one individually.

        self._print_graph(model)

    @torch.no_grad()
    def separate(self, mixture: torch.Tensor) -> Mapping[str, torch.Tensor]:
        device = "cpu"

        mixture_codes = self.encode_fn(mixture)

        mixture_len = len(mixture)

        # Prior = P(z_i)

        # Conditional Probability = P(z_i | m)

        # Likelihood = P(m | z_i)

        # Since len(z_i) and len(m) = 1024, all the probabilities are also torch.Tensors of length 1024

        CONSIDERED_TOKEN_IDX = 1
        # This is the frequency count tensor of rank K ^ {num_sources + 1} (3 in case of 2 sources).
        ll_coords, ll_data = self.likelihood._get_log_likelihood(
            x=mixture[CONSIDERED_TOKEN_IDX])


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
