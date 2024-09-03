from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def create_image_grid(path: Path, num_col: int, num_row: int):
    ori_path = path / 'ori'
    sep_path = path / 'sep'
    
    ori_images = sorted(ori_path.glob('*.png'))
    sep_images = sorted(sep_path.glob('*.png'))
    
    # Check if there are any images to process
    if not ori_images or not sep_images:
        raise ValueError("No images found in the specified directories.")
    
    num_images = num_col * num_row
    
    # Limit the images to the number that can fit in the grid
    ori_images = ori_images[:num_images]
    sep_images = sep_images[:num_images]
    
    assert len(ori_images) == len(sep_images), "Mismatch between ori and sep images"
    
    # Adjust figure size based on the grid dimensions
    aspect_ratio = num_col / num_row
    fig_width = 15  # Fixed width
    fig_height = fig_width / aspect_ratio  # Adjust height based on aspect ratio
    
    # Create and save the grid for original images
    fig, axes = plt.subplots(num_row, num_col, figsize=(fig_width, fig_height))
    
    for ax, img_path in zip(axes.flatten(), ori_images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
    
    # Hide any empty subplots
    for ax in axes.flatten()[len(ori_images):]:
        ax.axis('off')

    plt.tight_layout()
    
    # Save the figure in the parent directory
    ori_grid_path = path / 'ori_grid.png'
    fig.savefig(ori_grid_path)
    plt.close(fig)

    # Create and save the grid for separated images
    fig, axes = plt.subplots(num_row, num_col, figsize=(fig_width, fig_height))
    
    for ax, img_path in zip(axes.flatten(), sep_images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')

    # Hide any empty subplots
    for ax in axes.flatten()[len(sep_images):]:
        ax.axis('off')

    plt.tight_layout()
    
    # Save the figure in the parent directory
    sep_grid_path = path / 'sep_grid.png'
    fig.savefig(sep_grid_path)
    plt.close(fig)

    print(f"Original images grid saved to {ori_grid_path}")
    print(f"Separation results grid saved to {sep_grid_path}")

if __name__ == "__main__":
    create_image_grid(Path('lass_mnist/results/separation/gm-separated-images'), num_col=2, num_row=4)
    create_image_grid(Path('lass_mnist/results/separation/gm-three-separated-images'), num_col=3, num_row=4)
    create_image_grid(Path('lass_mnist/results/separation/pe-separated-images'), num_col=2, num_row=4)
    create_image_grid(Path('lass_mnist/results/separation/pe-three-separated-images'), num_col=3, num_row=4)
