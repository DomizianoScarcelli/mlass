import os
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List
import torch
from tqdm import tqdm
import hydra
import torch
from omegaconf import MISSING
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from .utils import CONFIG_DIR, CONFIG_STORE, ROOT_DIR

NUM_SOURCES = 3
RESTORE_FROM = 82

def roll(x, n):
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)

def split_images_by_step(images: torch.Tensor, batch_size: int, step: int) -> torch.Tensor:
    image_batches = []
    batched_images_size = batch_size // step
    for i in range(len(images) // batched_images_size):
        image_batches.append(
            images[i * batched_images_size: (i + 1) * batched_images_size])
    result = torch.stack(image_batches)
    return result

def get_mixtures(images: torch.Tensor) -> torch.Tensor:
    mixtures = []
    for i in range(2, len(images)+1):
        # Build the mixtures of the images up to index i
        mixture = torch.mean(images[:i], dim=0)
        mixtures.append(mixture)
    return torch.stack(mixtures)


def train(data_loader, sums, model, args, writer, step):
    for images, _ in tqdm(data_loader, desc="Training sums"):
        images = images.to(args.device)
        #torch.Size([3, 21, 1, 28, 28]) for NUM_SOURCES 3
        source_images = split_images_by_step(images, args.batch_size, step=NUM_SOURCES)
        # List of n tensors, each one is $m_i$, the mixture at step i, meaning
        # the mixture of the images 0,1,...,i-1
        # torch.Size([3, 21, 1, 28, 28]) for NUM_SOURCES 3
        images_mixtures = get_mixtures(images=source_images)

        with torch.no_grad():
            z_e_xs = [model(images)[1] for images in source_images]
            z_e_mixtures = [model(image_mixture)[1] for image_mixture in images_mixtures]

        codes = [model.codeBook(z_e_x) for z_e_x in z_e_xs]
        codes_mixtures = [model.codeBook(z_e_x_mixture) for z_e_x_mixture in z_e_mixtures]
        
        codes = torch.stack([code.flatten() for code in codes], dim=0)
        codes_mixtures = torch.stack([code.flatten() for code in codes_mixtures], dim=0)
        
        for i in range(NUM_SOURCES-1):
            if i == 0:
                sums[i, codes_mixtures[i], codes[i], codes[i+1]] += 1
                sums[i, codes_mixtures[i], codes[i+1], codes[i]] += 1
            else:
                sums[i, codes_mixtures[i], codes_mixtures[i-1], codes[i+1]] += 1
                sums[i, codes_mixtures[i], codes[i+1], codes_mixtures[i-1]] += 1
        step += 1
    return step


def evaluate(data_loader, sums, model, args, writer, step):
    with torch.no_grad():
        loss = 0.0
        sums_test = sums / torch.sum(sums) + 1e-16
        for images, _ in tqdm(data_loader, desc="Evaluating sums"):
            images = images.to(args.device)
            # Shape: torch.Size([3, 21, 1, 28, 28]) for NUM_SOURCES=3
            source_images = split_images_by_step(images, 16, step=NUM_SOURCES)
            # List of n tensors, each one is $m_i$, the mixture at step i, meaning
            # the mixture of the images 0,1,...,i-1
            # Shape: torch.Size([3, 21, 1, 28, 28]) for NUM_SOURCES=3
            # print(results)
            images_mixtures = get_mixtures(images=source_images)

            z_e_xs = [model(images)[1] for images in source_images]
            z_e_mixtures = [model(image_mixture)[1] for image_mixture in images_mixtures]

            codes = [model.codeBook(z_e_x) for z_e_x in z_e_xs]
            codes_mixtures = [model.codeBook(z_e_x_mixture) for z_e_x_mixture in z_e_mixtures]

            codes = torch.stack([code.flatten() for code in codes], dim=0)
            codes_mixtures = torch.stack([code.flatten() for code in codes_mixtures], dim=0)
                            
            for i in range(NUM_SOURCES-1):
                if i == 0:
                    loss += torch.mean(-torch.log(sums_test[i, codes_mixtures[i], codes[i], codes[i+1]] + 1e-16))
                    loss += torch.mean(-torch.log(sums_test[i, codes_mixtures[i], codes[i+1], codes[i]]) + 1e-16)
                else:
                    loss += torch.mean(-torch.log(sums_test[i, codes_mixtures[i], codes_mixtures[i-1], codes[i+1]] + 1e-16))
                    loss += torch.mean(-torch.log(sums_test[i, codes_mixtures[i], codes[i+1], codes_mixtures[i-1]] + 1e-16))

        loss /= len(data_loader)         
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
    batch_size: int = 128
    num_epochs: int = 500
    output_folder: str = "sums-MNIST-gm"
    num_workers: int = mp.cpu_count() - 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG_STORE.store(group="sums_estimation",
                   name="base_sums_estimation", node=SumsEstimationConfig)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="sums_estimation/mnist-gm.yaml")
def main(cfg):
    cfg: SumsEstimationConfig = cfg.sums_estimation
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    (ROOT_DIR / "logs").mkdir(exist_ok=True)
    (ROOT_DIR / "models").mkdir(exist_ok=True)
    (ROOT_DIR / "models" / cfg.output_folder).mkdir(exist_ok=True)

    writer = SummaryWriter("./logs/{0}".format(cfg.output_folder))
    save_filename = f"./lass_mnist/models/{cfg.output_folder}"

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
    
    if RESTORE_FROM is None:
        print(f"Creating empyy sums")
        sums = torch.zeros(NUM_SOURCES-1, cfg.num_codes, cfg.num_codes, cfg.num_codes).to(cfg.device)
    else:
        print(f"Loaded sums from epoch {RESTORE_FROM}")
        sums_path = f"./lass_mnist/models/sums-MNIST-gm/best_{RESTORE_FROM}.pt"
        if not os.path.exists(sums_path):
            raise ValueError(f"Path for restoring sums does not exist: {sums_path}")
        with open(sums_path, "rb") as f:
            sums = torch.load(f, map_location=cfg.device)
            if sums.shape[0]+1 != NUM_SOURCES:
                raise ValueError(f"The restored sums was generated for a different number of sources: {sums.shape[0] + 1}, which is different from the current {NUM_SOURCES} sources")

    step = 0
    best_loss = -1.0
    for epoch in tqdm(range(cfg.num_epochs - RESTORE_FROM if RESTORE_FROM is not None else 0)):
        step = train(train_loader, sums, model, cfg, writer, step)
        loss = evaluate(test_loader, sums, model, cfg, writer, step)

        print("loss = {:f} at epoch {:f}".format(loss, epoch + 1))
        writer.add_scalar("loss/testing_loss", loss, epoch + 1)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open(f"{save_filename}/best.pt", "wb") as f:
                torch.save(sums, f)
                print(f"Saved in {save_filename} with loss: {loss}")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == "__main__":
    main()
