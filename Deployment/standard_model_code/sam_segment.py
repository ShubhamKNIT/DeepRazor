import os
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
import torch
from PIL import Image
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

'''
    utilities functions from sam-2 model
'''

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def predict_masks_with_sam(
        img: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        device="cpu",
        refinement_mode=None
):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)

    sam2_checkpoint = r"checkpoints\sam2.1_hiera_large.pt"
    model_cfg = r"C:\projects_github\deeprazor_pipeline\checkpoints\sam2.1_hiera_l.yaml"
    # sam2_checkpoint = "/pretrained_model/checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = "/content/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    predictor.set_image(img)

    '''
    Predict with SAM2ImagePredictor.predict. The model returns masks,
    quality predictions for those masks, and low resolution mask logits 
    that can be passed to the next iteration of prediction.
    '''

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    '''
    With multimask_output=True (the default setting), SAM 2 outputs 3 masks,
    where scores gives the model's own estimation of the quality of these masks.
    This setting is intended for ambiguous input prompts, and helps the model
    disambiguate different objects consistent with the prompt
    '''

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    return masks, scores, logits


img_path = "baseball.jpg"
latest_coords = torch.tensor([[250, 250]], dtype=torch.float32)
point_labels = torch.tensor([1], dtype=torch.int64)
config_p = r"lama\configs\prediction\default.yaml"
ckpt_p = r"checkpoints\big-lama"
output_dir = r"output"

# Call the function to perform masking and inpainting
# mask_and_inpaint(img_path, latest_coords, point_labels, config_p, ckpt_p, output_dir)