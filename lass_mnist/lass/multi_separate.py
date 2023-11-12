"""
This will contain the code in order to perform the separation with more than 2 sources.

NOTE: for now, since I don't want to modify the dataset loader to have multiple sources, I will use only 2 sources, but use the graphical model
to separate.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Any, Optional, Mapping, Callable

import shutil
import hydra
import torch
import torchmetrics
import tqdm
from omegaconf import MISSING
from transformers import GPT2LMHeadModel, PreTrainedModel
from torchvision.utils import save_image
import numpy as np
import random

from diba.diba.naive_separation import separate

from ..modules import VectorQuantizedVAE
from .utils import refine_latents, CONFIG_DIR, ROOT_DIR, CONFIG_STORE
from .diba_interaces import DenseMarginalLikelihood, SparseLikelihood, UnconditionedTransformerPrior, DenseLikelihood
import multiprocessing as mp
from typing import Sequence
from numpy.random import default_rng
from torch.utils.data import Dataset


class PairsDataset(Dataset):
    def __init__(self, dataset: Sequence, seed: int = 0):
        super().__init__()
        self._rng = default_rng(seed=seed)
        self._dataset = dataset
        self._data_permutation = self._rng.permutation(len(dataset))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        data1 = self._dataset[item]
        data2 = self._dataset[self._data_permutation[item]]
        return dict(first=data1, second=data2)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def psnr_grayscale(target, preds, reduction="elementwise_mean", dim=None):
    return torchmetrics.functional.peak_signal_noise_ratio(
        preds, target, data_range=1.0, reduction=reduction, dim=dim)


def batched_psnr_unconditional(gts: List[torch.Tensor], gens: List[torch.Tensor]) -> float:
    (gt1, gt2), (gen1, gen2) = gts, gens
    dims = list(range(1, len(gt1.shape)))
    batched_psnr_12 = (
        0.5 * psnr_grayscale(gt1, gen1, reduction=None, dim=dims) +
        0.5 * psnr_grayscale(gt2, gen2, reduction=None, dim=dims)
    )

    batched_psnr_21 = (
        0.5 * psnr_grayscale(gt1, gen2, reduction=None, dim=dims) +
        0.5 * psnr_grayscale(gt2, gen1, reduction=None, dim=dims)
    )

    bpsnr = torch.stack([batched_psnr_12, batched_psnr_21], dim=-1)
    bpsnr_max, _ = bpsnr.max(dim=-1)
    return bpsnr_max.mean(dim=0).item()


def select_closest_to_mixture(
        vqvae: VectorQuantizedVAE,
        gen1: torch.LongTensor,
        gen2: torch.LongTensor,
        gt_mixture: torch.Tensor,
):
    gen1_o = vqvae.codes_to_latents(gen1).detach().clone()
    gen2_o = vqvae.codes_to_latents(gen2).detach().clone()

    # SELECT BEST
    geni1 = vqvae.decode_latents(gen1_o).detach().clone()
    geni2 = vqvae.decode_latents(gen2_o).detach().clone()

    rec_error = ((gt_mixture - (geni1 + geni2) * 0.5) ** 2).sum([1, 2, 3])
    sel = rec_error.argmin()

    return (geni1[sel], geni2[sel]), (gen1_o[sel], gen2_o[sel]), sel


@torch.no_grad()
def generate_samples(
    model: VectorQuantizedVAE,
    transformer: GPT2LMHeadModel,
    sums: torch.Tensor,
    gts: Tuple[torch.Tensor, torch.Tensor],
    bos: Tuple[int, int],
    latent_length: int,
):
    gt1, gt2 = gts
    gtm = 0.5 * gt1 + 0.5 * gt2

    # check input shape
    assert gt1.shape == gt2.shape
    batch_size = gt1.shape[0]

    _, z_e_x_mixture, _ = model(gtm)
    codes_mixture = model.codeBook(z_e_x_mixture)
    codes_mixture = codes_mixture.view(
        batch_size, latent_length ** 2).tolist()  # (B, H**2)

    label0, label1 = bos
    p0 = UnconditionedTransformerPrior(transformer=transformer, sos=label0)
    p1 = UnconditionedTransformerPrior(transformer=transformer, sos=label1)

    gen1ims, gen2ims = [], []
    gen1lats, gen2lats = [], []

    for bi in tqdm.tqdm(range(batch_size), desc="separating"):
        mixture = codes_mixture[bi]

        z0, z1 = separate(
            mixture=mixture, likelihood=sums, transformer=transformer, sources=2)

        (x0, x1), (x0lat, x1lat), _ = select_closest_to_mixture(
            vqvae=model,
            gen1=z0.reshape(-1, latent_length, latent_length),
            gen2=z1.reshape(-1, latent_length, latent_length),
            gt_mixture=gtm[bi:bi+1],
        )

        gen1ims.append(x0)
        gen2ims.append(x1)
        gen1lats.append(x0lat)
        gen2lats.append(x1lat)

    # Shapes are: torch.Size([2, (1), 1, 28, 28]), torch.Size([2, (1), 1, 28, 28]), torch.Size([2, (1), 128, 7, 7]), torch.Size([2, (1), 128, 7, 7])
    # the (1) means that the size has been squeezed
    gen1im = torch.stack(gen1ims, dim=0)
    gen2im = torch.stack(gen2ims, dim=0)
    gen1lat = torch.stack(gen1lats, dim=0)
    gen2lat = torch.stack(gen2lats, dim=0)

    return (gen1im, gen2im), (gen1lat, gen2lat)


@dataclass
class CheckpointsConfig:
    vqvae: str = MISSING
    autoregressive: str = MISSING
    sums: str = MISSING


@dataclass
class SeparationMethodConfig:
    do_sample: Optional[bool] = None
    num_beams: Optional[int] = None
    num_beams_groups: Optional[int] = None
    num_return_sequences: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None


@dataclass
class EvaluateSeparationConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"/dataset": MISSING},
            {"/vqvae": MISSING},
            {"/autoregressive": MISSING},
            {"/separation_method": "sampling"},
            "_self_",
        ]
    )

    latent_length: int = MISSING
    vocab_size: int = MISSING
    batch_size: int = 4
    # TODO: change it back to 64
    # batch_size: int = 16
    class_conditioned: bool = False
    num_workers: int = mp.cpu_count() - 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints: CheckpointsConfig = field(default_factory=CheckpointsConfig)

    # method: SeparationMethodConfig = SeparationMethodConfig()


CONFIG_STORE.store(
    group="separation", name="base_separation", node=EvaluateSeparationConfig
)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="separation/mnist.yaml")
def main(cfg):
    torch.manual_seed(0)
    cfg: EvaluateSeparationConfig = cfg.separation

    # instantiate models
    model = hydra.utils.instantiate(cfg.vqvae).to(cfg.device)
    transformer = hydra.utils.instantiate(cfg.autoregressive).to(cfg.device)

    assert isinstance(transformer, PreTrainedModel)

    # create output directory
    result_dir = ROOT_DIR / "multi-separated-images"
    if result_dir.exists():
        shutil.rmtree(result_dir)

    result_dir.mkdir(parents=True)
    (result_dir / "sep").mkdir()
    (result_dir / "ori").mkdir()

    # Define the train & test dataSets
    test_set = hydra.utils.instantiate(cfg.dataset)

    # Define the data loaders
    test_loader = torch.utils.data.DataLoader(
        PairsDataset(test_set, seed=200),
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
    )

    # load models
    with open(cfg.checkpoints.vqvae, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=cfg.device))

    with open(cfg.checkpoints.autoregressive, 'rb') as f:
        transformer.load_state_dict(torch.load(f, map_location=cfg.device))

    SUMS_CHECKPOINT_PATH = "lass_mnist/checkpoints/sum/best_3_sources.pt"
    with open(SUMS_CHECKPOINT_PATH, 'rb') as f:
        sums = torch.load(f, map_location=cfg.device)

    likelihood = DenseMarginalLikelihood(sums=sums)
    # set models to eval
    model.eval()
    transformer.eval()

    uncond_bos = 0

    print("Start separation")

    psnrs = []

    # main separation loop
    for i, batch in enumerate(tqdm.tqdm(test_loader)):

        gt1, labels1 = batch["first"]
        gt2, labels2 = batch["second"]

        # prepare the data
        gt1 = gt1.to(cfg.device)
        gt2 = gt2.to(cfg.device)

        labels1 = labels1 if cfg.class_conditioned else uncond_bos
        labels2 = labels2 if cfg.class_conditioned else uncond_bos

        (gen1, gen2), (gen1lat, gen2lat) = generate_samples(
            model=model,
            transformer=transformer,
            sums=likelihood.sums[:2],
            gts=[gt1, gt2],
            bos=[labels1, labels2],
            latent_length=cfg.latent_length,
        )

        gtm = (gt1 + gt2) / 2.0

        psnr = batched_psnr_unconditional(gts=[gt1, gt2], gens=[gen1, gen2])
        print(
            f"The psnr before refining for batch {i} is {psnr}")

        # print(f"Refining latents for batch {i}")
        # gen1, gen2 = refine_latents(
        #     model,
        #     gen1lat,
        #     gen2lat,
        #     gtm,
        #     n_iterations=500,
        #     learning_rate=1e-1,
        # )

        for j in range(len(gen1)):
            img_idx = i * cfg.batch_size + j
            save_image(gen1[j], result_dir / f"sep/{img_idx}-1.png")
            save_image(gen2[j], result_dir / f"sep/{img_idx}-2.png")
            save_image(gt1[j], result_dir / f"ori/{img_idx}-1.png")
            save_image(gt2[j],  result_dir / f"ori/{img_idx}-2.png")

        psnr = batched_psnr_unconditional(gts=[gt1, gt2], gens=[gen1, gen2])
        print(
            f"The psnr after refining for batch {i} is {psnr}")
        psnrs.append(psnr)

    print(f"Final psnr is {np.mean(psnrs)}")


if __name__ == '__main__':
    main()
