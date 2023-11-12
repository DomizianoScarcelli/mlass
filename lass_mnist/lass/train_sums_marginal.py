import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List

import hydra
import torch
from omegaconf import MISSING
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from .utils import CONFIG_DIR, CONFIG_STORE, ROOT_DIR

NUM_SOURCES = 4
RESUME = False
RESUME_FROM = 0


def roll(x, n):
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)


def split_images_by_step(images: torch.Tensor, batch_size: int, step: int) -> List[torch.Tensor]:
    image_batches = []

    batched_images_size = batch_size // step

    for i in range(len(images) // batched_images_size):
        image_batches.append(
            images[i * batched_images_size: (i + 1) * batched_images_size])

    return image_batches


def train(data_loader, sums, model, args, writer, step):
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
            codes_mixture = model.codeBook(z_e_x_mixture)

        codes: torch.Tensor = torch.stack(
            [code.flatten() for code in codes], dim=0)
        codes_mixture = codes_mixture.flatten()

        assert len(codes) == NUM_SOURCES

        for i in range(NUM_SOURCES):
            for j, code in enumerate(codes[i]):
                sums[i, code, codes_mixture[j]] += 1

            # TODO: the marginalization has to be done at the end
            # marginal = torch.sum(sums[i], dim=0)
            # sums[i] /= (marginal + 1e-9

        step += 1
    return step


def evaluate(data_loader, sums, model, args, writer, step):
    sums_test = sums / (torch.sum(sums) + 1e-16)
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

                stacked_test = []
                for i in range(NUM_SOURCES):
                    stacked_test.append(sums_test[i, codes[i], codes_mixture])
                stacked_test = torch.stack(stacked_test, dim=0)

                # NOTE: this is hardcoded because i don't know how to do it dynamically
                loss += torch.mean(-torch.log(stacked_test.flatten() + 1e-16))

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
        sums = torch.load(f"./best_{NUM_SOURCES}_sources.pt")
        print(f"Sums loaded correctly")
    else:
        sums = torch.zeros(
            (NUM_SOURCES, cfg.num_codes, cfg.num_codes)).to(cfg.device)

    step = 0
    best_loss = -1
    for epoch in tqdm(range(RESUME_FROM if RESUME else 0, cfg.num_epochs), desc="Training sums epochs"):
        step = train(train_loader, sums, model, cfg, writer, step)
        loss = evaluate(test_loader, sums, model, cfg, writer, step)

        print("loss = {:f} at epoch {:f}".format(loss, epoch + 1))
        writer.add_scalar("loss/testing_loss", loss, epoch + 1)

        if (epoch == 0) or (RESUME and epoch == RESUME_FROM) or (loss < best_loss):
            best_loss = loss
            model_path = f"./best_{NUM_SOURCES}_sources.pt"
            torch.save(sums, model_path)

            torch.save(torch.tensor([]),
                       f"./last_saved_epoch_is_{epoch}.pt")

            print(f"Saved at epoch {epoch} with loss {loss}")
        model_path = f"./sums_{NUM_SOURCES}_sources.pt"
        torch.save(sums, model_path)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == "__main__":
    main()
