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
from diba.diba.diba import _compute_log_posterior
from diba.diba.interfaces import SeparationPrior

from lass_audio.lass.datasets import SeparationDataset
from lass_audio.lass.datasets import SeparationSubset
from lass_audio.lass.diba_interfaces import JukeboxPrior, SparseLikelihood
from lass_audio.lass.utils import assert_is_audio, decode_latent_codes, get_dataset_subsample, get_raw_to_tokens, setup_priors, setup_vqvae
from lass_audio.lass.datasets import ChunkedMultipleDataset


from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD


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
        self.device = torch.device("cpu")

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
                                   num_sources: int) -> Dict[int, BayesianNetwork]:

        # Step 1: Define the Directed Graphical Model
        # model = BayesianNetwork()

        loader = DataLoader(dataset, batch_size=1, num_workers=0)

        models: Dict[int, BayesianNetwork] = {}

        batch: List[torch.Tensor]
        for batch_idx, batch in enumerate(tqdm(loader)):
            # Here I have two choices:
            # - One graph per tuple of sources, meaning num_batches total graphs, but small. This is good since the mixture only depends on the tuple of sources.
            # - One graph per dataset, meaning 1 single graph but huge. This is good in order to capture patterns.
            # Since the source influece only the mixture, and the priors capture other pieces of information needed, we will choose the first option.
            sources = batch
            weight = 1 / num_sources

            print(f"The source has shape: {sources[0].shape}")
            print(f"The weight is: {weight}")

            self.mixture = torch.tensor(
                [(source * weight).tolist() for source in sources]).sum(dim=0).squeeze(0)

            # TODO: this is hard coded, but it corresponds to the max_sample_tokens in the separate, it's also present in the dataset
            # I think it's done in order to have batches of tokens of audio
            time_steps = 1024

            model = BayesianNetwork()

            sources_nodes = [f"z_{i}_{t}" for i in range(
                num_sources) for t in range(time_steps)]
            mixture_nodes = [f"m_{t}" for t in range(time_steps)]

            model.add_nodes_from(sources_nodes + mixture_nodes)

            # Add edges for t_i_{t-1} -> t_i_{t} (Temporal dependency for the latent sources)
            for i in range(num_sources):
                for t in tqdm(range(time_steps), desc=f"Temporal dependency for the latent source z_{i}"):
                    if t == time_steps - 1:  # Skip the last time step
                        break
                    start = f"z_{i}_{t}"
                    end = f"z_{i}_{t+1}"
                    mixture_node = f"m_{t}"

                    model.add_edge(start, mixture_node)
                    model.add_edge(start, end)

            # Add edges for m_{t-1} -> m_{t} (Temporal dependency for the mixture signal)
            for t in tqdm(range(time_steps), desc="Temporal dependency for the mixture signal"):
                if t == time_steps - 1:  # Skip the last time step
                    break
                start = f"m_{t}"
                end = f"m_{t+1}"

                model.add_edge(start, end)

            # TODO: verify the correctness of this step
            # STEP 2: Add CPDs considering priors and likelihood
            # TODO: generalize for more than 2 sources, now this is just for debug reasons

            # self.priors is a list of a neural network that expose a _get_logits method
            prior_0, prior_1 = self.priors

            # TODO: in order to compute the logits, you can see how it's done in `diba._ancestral_sample``
            mixture_codes = self.encode_fn(self.mixture)

            DEBUG_NUM_SOURCES = 2
            # TODO: this is used inside of beam_search and it corresponds to the number of beans to keep, IDK what should be its value in this context
            DEBUG_NUM_CURRENT_BEAMS = 1
            xs_0, xs_1 = torch.full((DEBUG_NUM_SOURCES, DEBUG_NUM_CURRENT_BEAMS, time_steps + 1),
                                    fill_value=-1, dtype=torch.long, device=self.device)

            print(
                f"xs_0 configuration is: num_sources = {DEBUG_NUM_SOURCES}, beams = {DEBUG_NUM_CURRENT_BEAMS}, time_steps={time_steps+1} ")

            # NOTE: in this case p.get_sos() returns the value 0
            xs_0[:, 0], xs_1[:, 0] = [p.get_sos() for p in self.priors]
            past_0, past_1 = (None, None)

            # NOTE: Xs_0 is: tensor([[ 0, -1, -1,  ..., -1, -1, -1]]) with shape torch.Size([1, 1025])
            # NOTE: Xs_1 is: tensor([[ 0, -1, -1,  ..., -1, -1, -1]]) with shape torch.Size([1, 1025])
            # TODO: this is wrong since in separate xs_0 has shape  (10, 1025)

            print(f"Xs_0 is: {xs_0} with shape {xs_0.shape}")
            print(f"Xs_0 is: {xs_1} with shape {xs_1.shape}")

            for t in tqdm(range(time_steps), desc="Loop over all the time steps"):
                # compute log priors
                log_p_0, past_0 = prior_0._get_logits(
                    xs_0[:DEBUG_NUM_CURRENT_BEAMS, : t + 1], past_0)
                log_p_1, past_1 = prior_1._get_logits(
                    xs_1[:DEBUG_NUM_CURRENT_BEAMS, : t + 1], past_1)

                prior_probs_0 = torch.softmax(log_p_0, dim=-1)
                prior_probs_1 = torch.softmax(log_p_1, dim=-1)

                print(
                    f"Prior probs 0: {prior_probs_0} with shape {prior_probs_0.T.shape}")
                print(
                    f"Prior probs 1: {prior_probs_1} with shape {prior_probs_1.T.shape}")

                # TODO: in case t = 0, I just add the prior to the CPD, otherwise I should compute the CPD
                # using both the prior and the likelihood (?)
                z_0_t = f"z_0_{t}"
                z_1_t = f"z_1_{t}"

                DISCRETE_VALUES = 2048
                model.add_cpds(
                    TabularCPD(
                        variable=z_0_t, variable_card=DISCRETE_VALUES, values=prior_probs_0.T.numpy())
                )
                model.add_cpds(
                    TabularCPD(
                        variable=z_1_t, variable_card=DISCRETE_VALUES, values=prior_probs_1.T.numpy())
                )

                print(f"The model after computing the priors is: {model}")

                current_mixture_token = mixture_codes[t]

                # Likelihood for the current mixture token in sparse tensor format
                # print("current mixture token: ", current_mixture_token)
                # ll_coords, ll_data = self.likelihood._get_log_likelihood(
                #     current_mixture_token)

            # print(f"""
            #     Logits 1: {logits_1} with shape {logits_1.shape}
            #     Logits 2: {logits_2} with shape {logits_2.shape}
            #       """)

            # Add the model to the list of models
            models[batch_idx] = model

            print(
                f"Length of the dataset and number of batches: {len(dataset)}")

            print(
                f"model after step 1 is: {model}")

            raise RuntimeError("STOP HERE MAN!")

            # TODO:
            # Step 3: Parameterize the graph with prior and likelihood values
            # You'll need to define Conditional Probability Distributions (CPDs) for each node in the graph
            # based on your prior probabilities and likelihood values.

            # Iterate through each latent source and time step

            # (GPT help)[https://chat.openai.com/share/a20bcbfd-5970-449f-b9f4-baa2f97e61a6]

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
