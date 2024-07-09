import argparse
import re
from pathlib import Path 
from typing import Sequence, Tuple, Union, List

import numpy as np
import sparse
import torch
from lass_audio.jukebox.data.files_dataset import FilesAudioDataset
from lass_audio.jukebox.hparams import Hyperparams, setup_hparams
from lass_audio.jukebox.make_models import make_vqvae
from lass_audio.jukebox.utils.audio_utils import audio_preprocess
from lass_audio.jukebox.utils.dist_utils import setup_dist_from_mpi
from lass_audio.jukebox.vqvae.vqvae import VQVAE
from .utils import ROOT_DIRECTORY
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_SOURCES = 3

def split_datapoints_by_step(datapoints: torch.Tensor, batch_size: int, step: int) -> torch.Tensor:
    batches = []
    batched_size = batch_size // step
    for i in range(step):
        batches.append(
            datapoints[i * batched_size: (i + 1) * batched_size])
    result = torch.stack(batches)
    return result

def get_mixtures(datapoints: torch.Tensor) -> torch.Tensor:
    mixtures = []
    for i in range(2, len(datapoints)+1):
        # Build the mixtures of the images up to index i
        mixture = torch.mean(datapoints[:i], dim=0)
        mixtures.append(mixture)
    return torch.stack(mixtures)

def load_checkpoint(checkpoint_path: Union[Path, str]) -> Tuple[sparse.COO, int]:
    """Load ceckpoint containing the distribution of sum of the VQ-VAE

    Args:
        checkpoint_path: path to checkpoint

    Returns:
        sum_dist: sparse tensor containing the frequencies of sums
        iterations: how many iterations were used to compute the matrix
    """
    # get number of iterations from filename
    checkpoint_filename = Path(checkpoint_path).name
    match = re.fullmatch(
        r"sum_dist_(?P<iterations>[0-9]+)\.npz", checkpoint_filename)
    if match is None:
        raise RuntimeError(
            f"The filename {checkpoint_filename} is not in the correct format!"
            "It must be of format 'sum_dist_[NUM_ITER].npz'!"
        )
    iterations = int(match.group("iterations"))

    # load sparse matrix
    sum_dist = sparse.load_npz(checkpoint_path)
    assert isinstance(sum_dist, sparse.COO)
    return sum_dist, iterations

def collate_fn(batch):
    return torch.stack([torch.from_numpy(b) for b in batch], 0),

def estimate_distribution(
    hps: Hyperparams,
    output_dir: Union[Path, str],
    epochs: int,
    batch_size: int,
    alpha: Tuple[float],
    save_iters: int,
    vqvae_level: int,
    checkpoint_path: Union[Path, str, None] = None,
):
    """Estimate frequencies of sum-result pairs in pretrained VQ-VAE

    Args:
        hps: Hyper-parameters used by lass_audio
        output_dir: Directory that will contain the estiamted distribution file
        epochs: Number of epoch used for estimation
        batch_size: Batch size used during estimation
        alpha: Convex coefficients for the mixture
        save_iters: Number of iterations before storing of estiamtion checkpoint
        checkpoint_path: Path used for resuming estimation

    """

    def compute_latent(x: torch.Tensor, vqvae: VQVAE, level: int) -> Sequence[int]:
        z = vqvae.encode(x, start_level=level, end_level=level +
                         1, bs_chunks=1)[0].cpu().numpy()
        return np.reshape(z, newshape=-1, order="F").tolist()

    rank, local_rank, device = setup_dist_from_mpi(port=29540)

    # get number of latent codes
    latent_bins = hps.l_bins

    # instantiate the vqvae model and audio dataset
    vqvae = make_vqvae(hps, device)
    audio_dataset = FilesAudioDataset(hps)

    # prepare data-loader
    dataset_loader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    # VQ-VAE arithmetic statistics generation
    # load checkpoint if available
    if checkpoint_path is not None:
        sum_dist, iterations = load_checkpoint(checkpoint_path)
        _, prefix_i, prefix_j, prefix_k = sum_dist.coords.tolist()
        prefix_data = sum_dist.data.tolist()
        del sum_dist
    else:
        prefix_i, prefix_j, prefix_k, prefix_data = [], [], [], []
        iterations = 0

    buffer_adds: List[List[int]] = [[] for _ in range(NUM_SOURCES)]
    buffer_sums: List[List[int]] = [[] for _ in range(NUM_SOURCES-1)]

    # run estimation loop
    with torch.no_grad():
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(tqdm(dataset_loader, desc=f"Epoch: {epoch}/{epochs} | Separating audio...")):
                x = audio_preprocess(batch, hps=hps)

                # get tracks from batch
                xs = split_datapoints_by_step(x, batch_size, step=NUM_SOURCES)
                print("xs shape: ", xs.shape)
                mixtures = get_mixtures(datapoints=xs)
                print("mixtures shape: ", mixtures.shape)

                #TODO: continue

                # compute latent vectors
                for i, x_i in enumerate(xs):
                    buffer_adds[i].extend(compute_latent(x_i, vqvae, vqvae_level))
                for i, m_i in enumerate(mixtures):
                    buffer_sums[i].extend(compute_latent(m_i, vqvae, vqvae_level))

                # save output
                iterations += 1
                #NOTE: save_iters defaults to 500
                if iterations % save_iters == 0: 
                    if NUM_SOURCES not in [2,3]:
                        raise NotImplementedError("Train sums implemeneted just for 2 and 3 sources")
                    for i in range(NUM_SOURCES-1):
                        if i == 0:
                            sum_dist = sparse.COO(
                                coords=[
                                    i,
                                    prefix_i + buffer_adds[i] + buffer_adds[i+1],
                                    prefix_j + buffer_adds[i+1] + buffer_adds[i],
                                    prefix_k + buffer_sums[i] + buffer_sums[i],
                                ],
                                data=prefix_data + [1] * (len(buffer_adds[i]) * 2),
                                shape=[NUM_SOURCES-1, latent_bins, latent_bins, latent_bins],
                            )
                        else:
                            sum_dist = sparse.COO(
                                coords=[
                                    i,
                                    prefix_i + buffer_sums[i-1] + buffer_adds[i+1],
                                    prefix_j + buffer_adds[i+1] + buffer_sums[i-1],
                                    prefix_k + buffer_sums[i] + buffer_sums[i],
                                ],
                                data=prefix_data + [1] * (len(buffer_adds[i]) * 2),
                                shape=[NUM_SOURCES-1, latent_bins, latent_bins, latent_bins],
                            )

                        # store results
                        output_dir = Path(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_path = output_dir / f"sum_dist_{iterations}.npz"
                        sparse.save_npz(str(checkpoint_path), sum_dist)

                        # load compressed matrix
                        sum_dist, iterations = load_checkpoint(checkpoint_path)
                        _, prefix_i, prefix_j, prefix_k = sum_dist.coords.tolist()
                        prefix_data = sum_dist.data.tolist()
                        del sum_dist

                        # reset loop variables
                        buffer_adds: List[List[int]] = [[] for _ in range(NUM_SOURCES)]
                        buffer_sums: List[List[int]] = [[] for _ in range(NUM_SOURCES-1)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute an approximation of distribution of sums of latent codes in a VQ-VAE"
    )
    parser.add_argument(
        "--vqvae-path",
        type=str,
        help="Archive containing the VQ-VAE parameters. You can download it"
        "at the url https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar",
        default=str(ROOT_DIRECTORY / "checkpoints/vqvae.pth.tar")
    )
    parser.add_argument(
        "--audio-files-dir",
        type=str,
        help="Directory containing the audio files",
        default=str(ROOT_DIRECTORY / "data/train"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory in which the output estimation will be stored",
        default=str(ROOT_DIRECTORY / "logs/vqvae_sum_distribution"),
    )
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs", default=100)

    parser.add_argument(
        "--sample-length",
        type=float,
        help="Size in seconds of audio chunk used during estimation",
        default=11.88862,
    )
    parser.add_argument("--sample-rate", type=int,
                        help="Sample rate", default=44100)
    parser.add_argument(
        "--alpha",
        type=float,
        nargs=2,
        help="Convex coefficients for the mixture",
        default=(0.5, 0.5),
        metavar=("ALPHA_1", "ALPHA_2"),
    )

    parser.add_argument("--batch-size", type=int, help="Batch size", default=8)
    parser.add_argument(
        "--save-iters",
        type=int,
        help="Interval of steps before saving partial results",
        default=500,
    )
    parser.add_argument(
        "--checkpoint-path", type=str, help="Checkpoint path", default=None
    )
    parser.add_argument("--vqvae-level", type=int,
                        help="VQVAE Level", default=2)

    args = vars(parser.parse_args())
    vqvae_path = args.pop("vqvae_path")
    audio_files_dir = args.pop("audio_files_dir")

    # validate input
    msg = "Please specify a path containing the VQ-VAE parameters. "
    "You can download it at the url "
    "https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar"

    if vqvae_path is None:
        raise ValueError("Parameter --vqvae_path is required." + msg)

    if not Path(vqvae_path).exists():
        raise ValueError(f"The Path '{vqvae_path}' doesn't exist." + msg)

    if not Path(audio_files_dir).exists():
        raise ValueError(f"Directory {audio_files_dir} doesn't exists.")

    sr = args.pop("sample_rate")
    sl = args.pop("sample_length")
    hps = setup_hparams(
        "vqvae",
        #TODO: commented for now since those keys do not exist in the args dict
        dict(
            # vqvae hps
            # l_bins=args["latent_bins"],
            # downs_t=args.pop("downs_t"),
            # sr=sr,
            # commit=args.pop("commit"),
            restore_vqvae=vqvae_path,
            # data hps
            sample_length=int(sl * sr),
            audio_files_dir=audio_files_dir,
            min_duration=sl + 0.01,  # add a small epsilon
            labels=False,
            aug_shift=True,
            aug_blend=True,
        ),
    )

    # execute distribution estimation
    estimate_distribution(hps=hps, **args)
