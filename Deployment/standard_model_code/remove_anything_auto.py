import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import zipfile
import gdown

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import (
    load_img_to_array,
    save_array_to_img,
    dilate_mask,
    show_mask,
    show_points,
    get_clicked_point,
)

# ——— Constants ———
DRIVE_FOLDER_URL = "https://drive.google.com/uc?export=download&id=1paU4nSaxxz2yklEzqPIx5v0JPdfJ1JQw"
CHECKPOINTS_DIR   = "checkpoints"


def download_and_extract_drive_folder(folder_url: str, output_dir: str):
    """
    1) Uses gdown.download_folder to grab every file under `folder_url` into `output_dir`.
    2) Recursively scans `output_dir` for any .zip files, extracts each in place, then deletes the .zip.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading Drive folder → {output_dir} …")
    gdown.download_folder(
        url=folder_url,
        output=output_dir,
        quiet=False,
        use_cookies=False
    )
    print("Download complete.\n")

    for root, _, files in os.walk(output_dir):
        for fname in files:
            if fname.lower().endswith(".zip"):
                zip_path = os.path.join(root, fname)
                print(f"Found ZIP: {zip_path}. Extracting…")
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(root)
                    os.remove(zip_path)
                    print(f"Extracted and removed {zip_path}")
                except zipfile.BadZipFile:
                    print(f"  [!] Skipping invalid ZIP: {zip_path}")
                except Exception as e:
                    print(f"  [!] Error extracting {zip_path}: {e}")
    print("All ZIPs processed.\n")


def inpaint_and_save(image_path, mask_path, config_p, ckpt_p, output_dir="output/", mask_idx=0):
    """
    Load an image and mask, perform inpainting using LaMa, and save the output.
    """
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(img.size, resample=Image.NEAREST)

    img_arr = np.array(img)
    mask_arr = np.array(mask)

    os.makedirs(output_dir, exist_ok=True)

    inpainted_img = inpaint_img_with_lama(img_arr, mask_arr, config_p, ckpt_p)
    base_name = os.path.basename(image_path)
    output_filename = f"{os.path.splitext(base_name)[0]}_inpainted_mask_{mask_idx}.jpg"
    output_path = os.path.join(output_dir, output_filename)

    Image.fromarray(inpainted_img).save(output_path)
    print(f"Inpainted image saved at: {output_path}")


def mask_and_inpaint(
    img_path,
    latest_coords,
    point_labels=1,
    config_p=r"lama/configs/prediction/default.yaml",
    ckpt_p=r"checkpoints/big-lama",
    output_dir=r"output",
    dilate_kernel_size=15,
):
    """
    1. Checks if 'checkpoints/' exists and is non-empty. If not, downloads & extracts it.
    2. Generates masks using SAM, performs inpainting for each mask, and saves outputs.
    """
    # Step A: Check if checkpoints folder exists and is non-empty
    ckpt_root = Path(CHECKPOINTS_DIR)
    if not (ckpt_root.exists() and any(ckpt_root.iterdir())):
        print(f"⚠ '{CHECKPOINTS_DIR}' not found or empty. Downloading now...")
        download_and_extract_drive_folder(DRIVE_FOLDER_URL, CHECKPOINTS_DIR)
    else:
        print(f"✔ Found existing '{CHECKPOINTS_DIR}', skipping download.")

    # Step B: Define paths to subfolders
    big_lama_ckpt = os.path.join(CHECKPOINTS_DIR, "big-lama")
    sam_ckpt      = os.path.join(CHECKPOINTS_DIR, "sam")
    lama_cfg      = os.path.join(big_lama_ckpt, "configs", "prediction", "default.yaml")

    # Warn if expected subpaths are missing
    if not os.path.isdir(big_lama_ckpt):
        print(f"⚠ LaMa checkpoint folder not found at '{big_lama_ckpt}'")
    if not os.path.isdir(sam_ckpt):
        print(f"⚠ SAM checkpoint folder not found at '{sam_ckpt}'")
    if not os.path.isfile(lama_cfg):
        print(f"⚠ LaMa config file not found at '{lama_cfg}'")

    # Step C: Load image as array for SAM
    img_array = load_img_to_array(img_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step D: Predict masks with SAM
    masks, _, _ = predict_masks_with_sam(
        img_array,
        latest_coords.unsqueeze(0),  # (1, 1, 2)
        point_labels.unsqueeze(0),   # (1, 1)
        device=device,
        checkpoint_dir=sam_ckpt,
    )

    # Step E: Dilate masks if requested
    if dilate_kernel_size:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    # Step F: Prepare output directory
    img_stem = Path(img_path).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step G: Save visualizations and run inpainting
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        save_array_to_img(mask, mask_p)

        # Plot original with points
        height, width = img_array.shape[:2]
        dpi = plt.rcParams["figure.dpi"]
        plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
        plt.imshow(img_array)
        plt.axis("off")
        show_points(
            plt.gca(), latest_coords.numpy(), point_labels.numpy(), size=(width * 0.04) ** 2
        )
        plt.savefig(out_dir / "with_points.png", bbox_inches="tight", pad_inches=0.1)
        plt.close()

        # Plot with mask overlay
        plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
        plt.imshow(img_array)
        show_mask(plt.gca(), mask, random_color=False)
        plt.axis("off")
        plt.savefig(out_dir / f"with_mask_{idx}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

        # Step H: Inpaint and save result
        inpaint_and_save(
            img_path,
            str(mask_p),
            lama_cfg,
            big_lama_ckpt,
            output_dir=str(out_dir),
            mask_idx=idx,
        )


if __name__ == "__main__":
    img_path = "baseball.jpg"
    latest_coords = torch.tensor([[250, 250]], dtype=torch.float32)
    point_labels = torch.tensor([1], dtype=torch.int64)
    output_dir = "output"

    mask_and_inpaint(
        img_path,
        latest_coords,
        point_labels,
        config_p="",    # ignored; overridden internally
        ckpt_p="",      # ignored; overridden internally
        output_dir=output_dir,
        dilate_kernel_size=15,
    )
