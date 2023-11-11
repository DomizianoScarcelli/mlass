from __future__ import annotations
import json
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import hydra
import numpy as np
import torch
from omegaconf import MISSING
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from .utils import CONFIG_DIR, CONFIG_STORE, ROOT_DIR
import pickle
import sys

NUM_SOURCES = 2
RESUME = True
RESUME_FROM = 46


class SparseTensor:
    def __init__(self,
                 indices: Union[torch.Tensor, None],
                 values: Union[torch.Tensor, None],
                 shape: tuple,
                 data: Union[Dict[torch.Tensor, torch.Tensor], None] = None,
                 dtype: torch.dtype = torch.long):

        self.dtype = dtype
        self.shape = shape
        if data is not None:
            if indices is not None or values is not None:
                raise ValueError(
                    "Either data or indices and values must be provided.")

            self.data: Dict[torch.Tensor, torch.Tensor] = data
        else:
            if indices is None or values is None:
                raise ValueError(
                    "Either data or indices and values must be provided.")
            self.data: Dict[torch.Tensor, torch.Tensor] = {}

            assert indices.dtype == self.dtype and values.dtype == self.dtype

            assert indices.shape[0] == 0 or indices.shape[1] == NUM_SOURCES + 1

            assert indices.shape[0] == values.shape[
                0], f"The number of indices {indices.shape[0]} must be equal to the number of values {values.shape[0]}"

            for index, value in zip(indices, values):
                self.data[index] = value

    def index(self, index: torch.Tensor):
        assert index.dtype == self.dtype
        assert index.shape == (NUM_SOURCES + 1,)
        if index in self.data:
            return self.data[index]
        return torch.tensor(0, dtype=torch.long)

    def multi_index(self, indices: torch.Tensor):
        assert indices.dtype == self.dtype
        assert indices.shape[1] == NUM_SOURCES + 1 and indices.shape[0] > 0
        return torch.stack([self.data[index] for index in indices])

    def add_value(self, index: torch.Tensor, value: int, reduction: str = "sum") -> None:
        assert index.dtype == self.dtype
        assert index.shape == (NUM_SOURCES + 1,)
        if reduction == "sum":
            if index in self.data:
                self.data[index] += value
            else:
                self.data[index] = value
        else:
            raise NotImplementedError

    def normalize(self, reduction: str = "sum") -> None:
        value_sum = torch.sum(self.values())
        if reduction == "sum":
            for index in self.data:
                self.data[index] /= value_sum
        else:
            raise NotImplementedError

    def save(self, path: str):
        print(f"Dict size in MB is {sys.getsizeof(self.data)/ (1024 * 1024)}")
        torch.save(self.data.copy(), path)

    @staticmethod
    def load(path: str):
        data = torch.load(path)
        return SparseTensor(shape=tuple(256 for _ in range(NUM_SOURCES + 1)), data=data, indices=None, values=None)

    def values(self) -> torch.Tensor:
        return torch.tensor([value for value in self.data.values()])

    def copy(self) -> SparseTensor:
        return SparseTensor(shape=self.shape, data=self.data.copy(), indices=None, values=None)


def roll(x, n):
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)


def split_images_by_step(images: torch.Tensor, batch_size: int, step: int) -> List[torch.Tensor]:
    image_batches = []

    batched_images_size = batch_size // step

    for i in range(len(images) // batched_images_size):
        image_batches.append(
            images[i * batched_images_size: (i + 1) * batched_images_size])

    return image_batches


def train(data_loader, sums: SparseTensor, model, args, writer, step):
    for images, _ in tqdm(data_loader, desc="Training sums"):
        images = images.to(args.device)
        sources_images = split_images_by_step(
            images, args.batch_size, step=NUM_SOURCES)
        images_mixture = torch.mean(torch.stack(sources_images), dim=0)

        with torch.no_grad():
            z_e_xs = []
            for images in sources_images:
                _, z_e_x, _ = model(images)
                z_e_xs.append(z_e_x)
            _, z_e_x_mixture, _ = model(images_mixture)
            codes: List[torch.Tensor] = [
                model.codeBook(z_e_x) for z_e_x in z_e_xs]
            codes_mixture: torch.Tensor = model.codeBook(z_e_x_mixture)

        codes = torch.stack([code.flatten()
                            for code in codes], dim=0)
        codes_mixture = codes_mixture.flatten()

        assert len(codes) == NUM_SOURCES

        codes_indices = torch.stack(
            (*codes, codes_mixture), dim=0)

        ###########################
        # Update the sparse sum tensor by adding 1 to the indices of the codes
        ###########################

        # Remove duplicated from codes_indices
        codes_indices, indices = torch.unique(
            codes_indices, dim=1, return_inverse=True)

        codes_indices = codes_indices.t().to(torch.long)

        for i in range(codes_indices.shape[0]):
            sums.add_value(codes_indices[i], 1)

        # values = torch.tensor(
        #     [1 for _ in range(codes_indices.shape[1])], dtype=torch.long)

        # update = torch.sparse_coo_tensor(
        #     codes_indices, values, size=sums.shape)

        # coalesced_update = update.coalesce()

        # assert torch.all(coalesced_update.values() == 1)

        # sums += update

        # assert torch.any(sums.to_dense() == 1)

        step += 1
    return sums, step


def evaluate(data_loader, sums: SparseTensor, model, args, writer, step):
    sums_test = sums.copy()
    sums_test.normalize()

    batch_size = 16
    with torch.no_grad():
        loss = 0.0
        for images, _ in tqdm(data_loader, desc="Evaluating sums"):
            images = images.to(args.device)
            sources_images = split_images_by_step(
                images, batch_size, step=NUM_SOURCES)
            images_mixture = torch.mean(torch.stack(sources_images), dim=0)

            with torch.no_grad():
                z_e_xs = []
                for images in sources_images:
                    _, z_e_x, _ = model(images)
                    z_e_xs.append(z_e_x)
                _, z_e_x_mixture, _ = model(images_mixture)
                codes: List[torch.Tensor] = [
                    model.codeBook(z_e_x) for z_e_x in z_e_xs]

                assert len(codes) == NUM_SOURCES

                codes_mixture = model.codeBook(z_e_x_mixture)

                codes = torch.stack([code.flatten() for code in codes], dim=0)
                codes_mixture = codes_mixture.flatten()

                codes_indices = torch.stack((*codes, codes_mixture), dim=0)

                # Remove duplicated from codes_indices
                codes_indices, indices = torch.unique(
                    codes_indices, dim=1, return_inverse=True)

                codes_indices = codes_indices.t()

                # TODO: make this more efficient, remove for loop
                test_sum = torch.tensor([])
                for i in range(codes_indices.shape[0]):
                    indexed = sums_test.index(codes_indices[i]).unsqueeze(0)
                    test_sum = torch.cat((test_sum, indexed))

                # assert torch.allclose(test_sum, sums_test.to_dense()[
                #                       codes[0], codes[1], codes_mixture].flatten()), f"""
                # test_sum is {test_sum}
                # sums_test is {sums_test.to_dense()[codes[0], codes[1], codes_mixture]}
                # """

                loss += torch.mean(-torch.log(test_sum + 1e-16))

        loss /= len(data_loader)
    # Logs
    writer.add_scalar("loss/test/loss", loss.item(), step)

    return loss.item()


@dataclass
class SumsEstimationConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"/dataset@train_dataset": MISSING},
            {"/dataset@test_dataset": MISSING},
            {"/vqvae": MISSING},
            "_self_",
        ]
    )
    vqvae_checkpoint: str = "lass_mnist/checkpoints/vqvae/256-sigmoid-big.pt"
    num_codes: int = MISSING
    batch_size: int = 64
    num_epochs: int = 500
    output_folder: str = "sums"
    num_workers: int = mp.cpu_count() - 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG_STORE.store(group="sums_estimation",
                   name="base_sums_estimation", node=SumsEstimationConfig)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="sums_estimation/mnist.yaml")
def main(cfg):
    torch.manual_seed(0)
    cfg: SumsEstimationConfig = cfg.sums_estimation
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    (ROOT_DIR / "logs").mkdir(exist_ok=True)
    (ROOT_DIR / "models").mkdir(exist_ok=True)
    (ROOT_DIR / "models" / cfg.output_folder).mkdir(exist_ok=True)

    writer = SummaryWriter("./logs/{0}".format(cfg.output_folder))
    save_filename = "./models/{0}".format(cfg.output_folder)

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(
        hydra.utils.instantiate(cfg.train_dataset),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        hydra.utils.instantiate(cfg.test_dataset),
        num_workers=cfg.num_workers,
        batch_size=16,
        shuffle=False,
        drop_last=True,
    )

    # Fixed images for TensorBoard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8,
                           value_range=(-1, 1), normalize=True)
    writer.add_image("original", fixed_grid, 0)

    # load vqvae
    model = hydra.utils.instantiate(cfg.vqvae).to(cfg.device)
    with open(cfg.vqvae_checkpoint, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=cfg.device))

    # freeze vqvae parameters
    for param in model.parameters():
        param.requires_grad = False

    if RESUME:
        print(
            f"Loading sums ./best_{NUM_SOURCES}_sources.pt from epoch {RESUME_FROM}")
        sums = SparseTensor.load(f"./best_{NUM_SOURCES}_sources.pt")
        print(f"Sums loaded correctly")
    else:
        sums = SparseTensor(indices=torch.tensor([], dtype=torch.long),
                            values=torch.tensor([], dtype=torch.long),
                            shape=tuple(cfg.num_codes for _ in range(NUM_SOURCES + 1)))

    assert sums.shape == tuple(cfg.num_codes for _ in range(NUM_SOURCES + 1))

    step = 0
    best_loss = -1.0
    for epoch in tqdm(range(RESUME_FROM if RESUME else 0, cfg.num_epochs), desc="Training sums epochs"):
        sums, step = train(train_loader, sums, model, cfg, writer, step)

        # if (epoch) % 20 == 0 or epoch == cfg.num_epochs - 1:
        loss = evaluate(test_loader, sums, model, cfg, writer, step)

        print("loss = {:f} at epoch {:f}".format(loss, epoch + 1))
        writer.add_scalar("loss/testing_loss", loss, epoch + 1)

        if (epoch == 0) or (loss <= best_loss):
            best_loss = loss
            model_path = f"./best_{NUM_SOURCES}_sources.pt"
            sums.save(model_path)

            torch.save(torch.tensor([]),
                       f"./last_saved_epoch_is_{epoch}.pt")

            print(f"Saved at epoch {epoch} with loss {loss}")
        model_path = f"./sums_{NUM_SOURCES}_sources.pt"
        sums.save(model_path)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == "__main__":
    main()
