import os
import sys
import numpy as np
import torch
import yaml
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path

# Limit threading for performance optimization
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for TensorFlow

# Add the "lama" directory to Python path for importing its modules
sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo


@torch.no_grad()
def inpaint_img_with_lama(
        img: np.ndarray,
        mask: np.ndarray,
        config_p: str,
        ckpt_p: str,
        mod=8,
        device="cpu"
):
    """
    Perform image inpainting using the LaMa model.

    Args:
        img (np.ndarray): Input image as a NumPy array.
        mask (np.ndarray): Binary mask indicating regions to inpaint.
        config_p (str): Path to the LaMa model configuration file.
        ckpt_p (str): Path to the model checkpoint directory.
        mod (int): Modulo value for padding dimensions.
        device (str): Device to use for computation ("cuda" or "cpu").

    Returns:
        np.ndarray: Inpainted image as a NumPy array.
    """

    assert len(mask.shape) == 2, "Mask must be a 2D array."

    # Convert binary mask values from [0, 1] to [0, 255]
    if np.max(mask) == 1:
        mask = mask * 255

    # Normalize image and convert to PyTorch tensors
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()

    # Load model configuration using OmegaConf
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p

    device = torch.device(device)

    # Load training configuration from `config.yaml`
    train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True  # Enable prediction-only mode
    train_config.visualizer.kind = 'noop'  # Disable visualization

    # Load pre-trained model checkpoint
    checkpoint_path = os.path.join(predict_config.model.path, 'models', predict_config.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()  # Freeze model parameters (no updates during inference)

    if not predict_config.get('refine', False):
        model.to(device)

    # Prepare input batch with image and mask tensors
    batch = {
        'image': img.permute(2, 0, 1).unsqueeze(0),  # Convert HWC to CHW and add batch dimension
        'mask': mask[None, None]  # Add batch and channel dimensions to mask tensor
    }

    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]  # Original size before padding

    # Pad image and mask tensors to dimensions divisible by `mod`
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)

    batch = move_to_device(batch, device)  # Move batch data to specified device (CPU/GPU)
    batch['mask'] = (batch['mask'] > 0) * 1

    # Perform inference using the LaMa model on the input batch
    batch = model(batch)

    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)  # Convert CHW back to HWC format
    cur_res = cur_res.detach().cpu().numpy()  # Move result back to CPU

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]  # Crop result back to original size

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')  # Rescale pixel values back to [0, 255]
    
    return cur_res  # Return inpainted image as a NumPy array
