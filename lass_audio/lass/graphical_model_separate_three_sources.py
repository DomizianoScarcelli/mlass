import abc
import functools
from pathlib import Path
from typing import Callable, List, Mapping, Optional


import numpy as np
import sparse
import torch
import torchaudio
import tqdm
from torch.utils.data import DataLoader
from diba.diba.interfaces import SeparationPrior
from diba.diba.sparse_graphical_model import SparseDirectedGraphicalModel

from diba.diba.sparse_utils import convert_sparse_coo_to_torch_coo
from lass_audio.lass.datasets import ChunkedMultipleDataset, SeparationDataset
from lass_audio.lass.datasets import SeparationSubset
from lass_audio.lass.diba_interfaces import JukeboxPrior 
from lass_audio.lass.utils import assert_is_audio, decode_latent_codes, get_dataset_subsample, get_raw_to_tokens, setup_priors, setup_vqvae
from diba.diba.evaluation import save_sdr, compute_sdr


audio_root = Path(__file__).parent.parent

class Separator(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def separate(self, mixture) -> Mapping[str, torch.Tensor]:
        ...

class SparseDirectedGraphicalSeparator(Separator):
    def __init__(
            self,
            encode_fn: Callable,
            decode_fn: Callable,
            priors: Mapping[str, SeparationPrior],
            sums: torch.Tensor,
            topk: Optional[int] = None 
            ):
        super().__init__()
        self.source_types = list(priors)
        self.priors = list(priors.values())
        print(f"Prior list size: ", len(self.priors))
        self.gm = SparseDirectedGraphicalModel(
                priors = list(priors.values()),
                sums=sums,
                num_tokens=sums.shape[-1],
                num_sources=3,
                topk=topk) 

        # lambda x: vqvae.encode(x.unsqueeze(-1), vqvae_level, vqvae_level + 1).view(-1).tolist()
        self.encode_fn = encode_fn
        # lambda x: decode_latent_codes(vqvae, x.squeeze(0), level=vqvae_level)
        self.decode_fn = decode_fn

    @torch.no_grad()
    def separate(self, mixture: torch.Tensor) -> Mapping[str, torch.Tensor]:
        # convert signal to codes
        mixture_codes = self.encode_fn(mixture)

        # separate mixture (x has shape [2, num. tokens])
        x = self.gm.separate(mixture=mixture_codes).to(self.sums.device)
        print(f"x shape: {x.shape}")

        x_0, x_1, x_2 = x[0], x[1], x[2]
        dec_bs = 1
        num_batches = int(np.ceil(self.gm.num_beams / dec_bs))
        res_0, res_1, res_2 = [None]*num_batches, [None]*num_batches, [None]*num_batches

        print(x_0.shape, x_1.shape, x_2.shape)

        for i in range(num_batches):
            res_0[i] = self.decode_fn(x_0[i * dec_bs:(i + 1) * dec_bs].flatten())
            res_1[i] = self.decode_fn(x_1[i * dec_bs:(i + 1) * dec_bs].flatten())
            res_2[i] = self.decode_fn(x_2[i * dec_bs:(i + 1) * dec_bs].flatten())

        res_0 = torch.cat(res_0, dim=0).view(self.gm.num_beams, -1)
        res_1 = torch.cat(res_1, dim=0).view(self.gm.num_beams, -1)
        res_2 = torch.cat(res_2, dim=0).view(self.gm.num_beams, -1)
        
        print(res_0.shape, res_1.shape, res_2.shape)
        mean = torch.mean(torch.stack([res_0, res_1, res_2], dim=0), dim=0).cpu()
        print(mean.shape)
        # select best
        best_idx = (mean - torch.tensor(mixture).view(1, -1)).norm(p=2, dim=-1).argmin()
        return {source: self.decode_fn(xi.flatten()) for source, xi in zip(self.source_types, [x_0[best_idx], x_1[best_idx], x_2[best_idx]])}

# -----------------------------------------------------------------------------

@torch.no_grad()
def separate_dataset(
    dataset: SeparationDataset,
    separator: Separator,
    save_path: Path,
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
        print(f"chunk {batch_idx+1} out of {len(dataset)}")

        # generate mixture
        mixture = torch.mean(torch.stack(origs), dim=0).squeeze(0) # shape: [1 , sample-length]
        print("Mixture shape: ", mixture.shape)
        seps = separator.separate(mixture=mixture)
        chunk_path.mkdir(parents=True)

        # save separated audio
        print("Number of separated signals: ", len(seps.values()))
        print("Number of original signals: ", len(origs))
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
    SDR_PATH = audio_root / "sdr.json"
    assert_is_audio(*original_signals, *separated_signals)
    # assert original_1.shape == original_2.shape == separation_1.shape == separation_2.shape
    assert len(original_signals) == len(separated_signals)
    for i, (ori, sep) in enumerate(zip(original_signals, separated_signals)):
        print(ori.shape, sep.shape)
        sdr = compute_sdr(ori.cpu(), sep.cpu())
        save_sdr(sdr=sdr, path=SDR_PATH)
        print(f"SDR is: ", sdr)
        torchaudio.save(str(path / f"ori{i+1}.wav"),
                        ori.cpu(), sample_rate=sample_rate)
        torchaudio.save(str(path / f"sep{i+1}.wav"),
                        sep.cpu(), sample_rate=sample_rate)


def main(
    audio_dirs: List[Path] = [audio_root / "data/test/piano", audio_root / "data/test/bass", audio_root / "data/test/drums"],
    vqvae_path: Path = audio_root / "checkpoints/vqvae.pth.tar",
    prior_paths: List[Path] = [audio_root / "checkpoints/prior_piano_44100.pth.tar",audio_root / "checkpoints/prior_bass_44100.pth.tar", audio_root / "checkpoints/prior_drums_44100.pth.tar"],
    sum_frequencies_path: Path = audio_root / "checkpoints/sum_dist_43500.npz",
    vqvae_type: str = "vqvae",
    prior_types: List[str] = ["small_prior", "small_prior", "small_prior"],
    max_sample_tokens: int = 1024,
    sample_rate: int = 44100,
    save_path: Path = audio_root / "three-separated-audio",
    resume: bool = True,
    num_pairs: int = 100,
    seed: int = 0,
    **kwargs,
):
    # convert paths
    save_path = Path(save_path)
    audio_dirs = [Path(dir) for dir in audio_dirs]

    # if not resume and save_path.exists():
    #    raise ValueError(f"Path {save_path} already exists!")

    # rank, local_rank, device = setup_dist_from_mpi(port=29533, verbose=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # setup models
    vqvae = setup_vqvae(
        vqvae_path=vqvae_path,
        vqvae_type=vqvae_type,
        sample_rate=sample_rate,
        sample_tokens=max_sample_tokens,
        device=device,
    )
    print("VQVAE setup completed")
    priors = setup_priors(
        prior_paths=prior_paths,
        prior_types=prior_types,
        vqvae=vqvae,
        fp16=True,
        device=device,
    )
    priors = {Path(prior_path.stem): prior for prior_path, prior in zip(prior_paths, priors)}
    print("Priors setup completed, size is: ", len(priors))

    # setup dataset
    raw_to_tokens = get_raw_to_tokens(vqvae.strides_t, vqvae.downs_t)
    dataset = ChunkedMultipleDataset(
        instruments_audio_dir=audio_dirs,
        sample_rate=sample_rate,
        max_chunk_size=raw_to_tokens * max_sample_tokens,
        min_chunk_size=raw_to_tokens,
    )

    # subsample the test dataset
    indices = get_dataset_subsample(len(dataset), num_pairs, seed=seed)
    subdataset = SeparationSubset(dataset, indices=list(indices))

    with open(sum_frequencies_path, "rb") as f:
        sums_coo: sparse.COO = sparse.load_npz(sum_frequencies_path)
        sums = convert_sparse_coo_to_torch_coo(sums_coo, device).to(device)
        print("Sums: ", sums)

    # create separator
    level = vqvae.levels - 1
    print("Creating separator")
    separator = SparseDirectedGraphicalSeparator(
        encode_fn=lambda x: vqvae.encode(
            x.unsqueeze(-1).to(device), level, level + 1)[-1].squeeze(0).tolist(), 
        decode_fn=lambda x: decode_latent_codes(
            vqvae, x.squeeze(0), level=level),
        priors={k: JukeboxPrior(p.prior, torch.zeros(
            (), dtype=torch.float32, device=device)) for k, p in priors.items()},
        sums=sums,
        topk=64,
        **kwargs,
    )
    print(f"Separator setup completed")

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
