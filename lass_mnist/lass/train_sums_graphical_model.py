import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List
from tqdm import tqdm
import hydra
import torch
from omegaconf import MISSING
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from .utils import CONFIG_DIR, CONFIG_STORE, ROOT_DIR

NUM_SOURCES = 3

def roll(x, n):
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)

def split_images_by_step(images: torch.Tensor, batch_size: int, step: int) -> torch.Tensor:
    image_batches = []
    batched_images_size = batch_size // step
    for i in range(len(images) // batched_images_size):
        image_batches.append(
            images[i * batched_images_size: (i + 1) * batched_images_size])
    return torch.stack(image_batches)

def get_mixtures(images: torch.Tensor) -> torch.Tensor:
    mixtures = []
    for i in range(len(images)):
        # Build the mixtures of the images up to index i
        mixture = torch.mean(images[:i], dim=0)
        mixtures.append(mixture)
    return torch.stack(mixtures)


def train(data_loader, sums, model, args, writer, step):
    for images, _ in tqdm(data_loader, desc="Training sums"):
        images = images.to(args.device)
        # Shape: torch.Size([3, 21, 1, 28, 28] for NUM_SOURCES=3
        source_images = split_images_by_step(images, args.batch_size, step=NUM_SOURCES)
        # List of n tensors, each one is $m_i$, the mixture at step i, meaning
        # the mixture of the images 0,1,...,i-1
        # Shape: torch.Size([3, 21, 1, 28, 28] for NUM_SOURCES=3
        images_mixtures = get_mixtures(images=source_images)

        with torch.no_grad():
            z_e_xs = [model(images)[1] for images in source_images]
            z_e_mixtures = [model(image_mixture)[1] for image_mixture in images_mixtures]
            # z_e_xs = []
            # z_e_mixtures = []
            # for images in source_images:
            #     _, z_e_x, _ = model(images)
            #     z_e_xs.append(z_e_x)

            # for image_mixture in images_mixtures:
            #     _, z_e_mixture, _ = model(image_mixture)
            #     z_e_mixtures.append(z_e_mixture)
            
        codes = [model.codeBook(z_e_x) for z_e_x in z_e_xs]
        codes_mixtures = [model.codeBook(z_e_x_mixture) for z_e_x_mixture in z_e_mixtures]

        codes = torch.stack([code.flatten() for code in codes], dim=0)
        codes_mixtures = torch.stack([code.flatten() for code in codes_mixtures], dim=0)

        for i in range(NUM_SOURCES):
            if i == 0:
                sums[i, codes_mixtures[i], torch.zeros_like(codes[i]), codes[i]] += 1
            else:
                sums[i, codes_mixtures[i], codes_mixtures[i-1], codes[i]] += 1
        

        step += 1
    return step


def evaluate(data_loader, sums, model, args, writer, step):
    sums_test = sums / (torch.sum(sums) + 1e-16)
    with torch.no_grad():
        loss = 0.0
        for images, _ in data_loader:
            images = images.to(args.device)
            images1 = images[: 16 // 2]
            images2 = images[16 // 2 :]
            images_mixture = 0.5 * images1 + 0.5 * images2

            with torch.no_grad():
                _, z_e_x1, _ = model(images1)
                _, z_e_x2, _ = model(images2)
                _, z_e_x_mixture, _ = model(images_mixture)
                codes1 = model.codeBook(z_e_x1)
                codes2 = model.codeBook(z_e_x2)
                codes_mixture = model.codeBook(z_e_x_mixture)
                loss += torch.mean(-torch.log(sums_test[codes1, codes2, codes_mixture] + 1e-16))

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
    save_filename = "./lass_mnist/models/{0}".format(cfg.output_folder)

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
    fixed_grid = make_grid(fixed_images, nrow=8, value_range=(-1, 1), normalize=True)
    writer.add_image("original", fixed_grid, 0)

    # load vqvae
    model = hydra.utils.instantiate(cfg.vqvae).to(cfg.device)
    with open(cfg.vqvae_checkpoint, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=cfg.device))

    # freeze vqvae parameters
    for param in model.parameters():
        param.requires_grad = False

    sums = torch.zeros(NUM_SOURCES, cfg.num_codes, cfg.num_codes, cfg.num_codes).to(cfg.device)

    step = 0
    best_loss = -1.0
    for epoch in tqdm(range(cfg.num_epochs)):
        step = train(train_loader, sums, model, cfg, writer, step)
        loss = evaluate(test_loader, sums, model, cfg, writer, step)

        print("loss = {:f} at epoch {:f}".format(loss, epoch + 1))
        writer.add_scalar("loss/testing_loss", loss, epoch + 1)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open("{0}/best.pt".format(save_filename), "wb") as f:
                torch.save(sums, f)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == "__main__":
    main()
