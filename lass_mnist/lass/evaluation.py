from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def create_image_grid(path: Path, max_images: int = None):
    ori_path = path / 'ori'
    sep_path = path / 'sep'
    
    ori_images = sorted(ori_path.glob('*.png'))
    sep_images = sorted(sep_path.glob('*.png'))
    
    # Check if there are any images to process
    if not ori_images or not sep_images:
        raise ValueError("No images found in the specified directories.")
    
    if max_images:
        ori_images = ori_images[:max_images]
        sep_images = sep_images[:max_images]
    
    assert len(ori_images) == len(sep_images), "Mismatch between ori and sep images"
    
    num_images = len(ori_images)
    
    # Ensure grid size is a positive integer
    grid_size = int(np.ceil(np.sqrt(num_images)))
    if grid_size == 0:
        raise ValueError("Number of images to display is zero, cannot create a grid.")

    # Create and save the grid for original images
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    # fig.suptitle('Original Images')
    
    for ax, img_path in zip(axes.flatten(), ori_images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
    
    # Hide any empty subplots
    for ax in axes.flatten()[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    
    # Save the figure in the parent directory
    ori_grid_path = path / 'ori_grid.png'
    fig.savefig(ori_grid_path)
    plt.close(fig)

    # Create and save the grid for separated images
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    # fig.suptitle('Separation Results')
    
    for ax, img_path in zip(axes.flatten(), sep_images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')

    # Hide any empty subplots
    for ax in axes.flatten()[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    
    # Save the figure in the parent directory
    sep_grid_path = path / 'sep_grid.png'
    fig.savefig(sep_grid_path)
    plt.close(fig)

    print(f"Original images grid saved to {ori_grid_path}")
    print(f"Separation results grid saved to {sep_grid_path}")

if __name__ == "__main__":
    create_image_grid(Path('lass_mnist/results/separation/gm-separated-images'), max_images=16)
