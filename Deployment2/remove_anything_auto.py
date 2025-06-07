import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from checkpoint_downloader import download_all_checkpoints
from utils import (
    load_img_to_array,
    save_array_to_img,
    dilate_mask,
    show_mask,
    show_points,
    get_clicked_point,
)

CHECKPOINTS_DIR = "checkpoints"


def inpaint_and_save(image_path, mask_path, config_p, ckpt_p, output_dir="output/", mask_idx=0):
    """
    Load an image and mask, perform inpainting using LaMa, and save the output.
    
    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the input mask.
        config_p (str): Path to the LaMa model configuration file.
        ckpt_p (str): Path to the LaMa model checkpoint directory.
        output_dir (str): Directory to save the output image.
        mask_idx (int): Index of the mask used for unique filenames.
    """
    # Load image and mask
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Resize the mask to match the image size (if needed)
    mask = mask.resize(img.size, resample=Image.NEAREST)

    img = np.array(img)
    mask = np.array(mask)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Perform inpainting using LaMa
    inpainted_img = inpaint_img_with_lama(img, mask, config_p, ckpt_p)

    # Save the inpainted image with a unique filename
    base_name = os.path.basename(image_path)
    output_filename = f"{os.path.splitext(base_name)[0]}_inpainted_mask_{mask_idx}.jpg"
    output_path = os.path.join(output_dir, output_filename)

    Image.fromarray(inpainted_img).save(output_path)
    print(f"Inpainted image saved at: {output_path}")


def mask_and_inpaint(img_path, latest_coords, point_labels=1,
                     config_p=r"lama\configs\prediction\default.yaml",
                     ckpt_p=r"checkpoints\big-lama",
                     output_dir=r"output", dilate_kernel_size=15):
    """
    Generate masks using SAM, perform inpainting for each mask, and save the outputs.
    
    Args:
        img_path (str): Path to the input image.
        latest_coords (torch.Tensor): Tensor of coordinates for SAM input.
        point_labels (torch.Tensor): Tensor of point labels.
        config_p (str): Path to LaMa model configuration file.
        ckpt_p (str): Path to LaMa model checkpoint directory.
        output_dir (str): Directory to save outputs.
        dilate_kernel_size (int, optional): Kernel size for mask dilation.
    """

    """
    1. Checks if 'checkpoints/' exists and is non-empty. If not, downloads & extracts it.
    2. Generates masks using SAM, performs inpainting for each mask, and saves outputs.
    """
    # Step A: Check if checkpoints folder exists and is non-empty
    ckpt_root = Path(CHECKPOINTS_DIR)
    if not (ckpt_root.exists() and any(ckpt_root.iterdir())):
        print(f"⚠ '{CHECKPOINTS_DIR}' not found or empty. Downloading now...")
        download_all_checkpoints("./checkpoints")
    else:
        print(f"✔ Found existing '{CHECKPOINTS_DIR}', skipping download.")

    # Load the image as an array
    img_array = load_img_to_array(img_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Predict masks using SAM
    masks, _, _ = predict_masks_with_sam(
        img_array,
        latest_coords.unsqueeze(0),  # Shape: (1, 1, 2)
        point_labels.unsqueeze(0),   # Shape: (1, 1)
        device=device,
    )

    # Dilate each mask if required
    if dilate_kernel_size is not None:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    # Prepare output directories
    img_stem = Path(img_path).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save visualizations, masks, and perform inpainting for each mask
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / "with_points.png"
        img_mask_p = out_dir / f"with_mask_{idx}.png"

        # Save the mask as an image
        save_array_to_img(mask, mask_p)

        dpi = plt.rcParams['figure.dpi']
        height, width = img_array.shape[:2]
        plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
        plt.imshow(img_array)
        plt.axis('off')

        # Plot points on the image
        show_points(plt.gca(), latest_coords.numpy(), point_labels.numpy(), size=(width * 0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=2)

        # Overlay the mask and save
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Inpaint and save the result
        inpaint_and_save(img_path, str(mask_p), config_p, ckpt_p, output_dir=str(out_dir), mask_idx=idx)


if __name__ == "__main__":
    img_path = "baseball.jpg"
    latest_coords = torch.tensor([[250, 250]], dtype=torch.float32)
    point_labels = torch.tensor([1], dtype=torch.int64)
    output_dir = "output"

    mask_and_inpaint(
        img_path,
        latest_coords,
        point_labels
    )
