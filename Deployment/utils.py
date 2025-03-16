import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from torchvision import transforms
import streamlit as st
import zipfile
import os
import uuid
import subprocess
from pathlib import Path

# Ensure a unique session id exists
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

# Define a unique folder using the session id
RESULT_FOLDER = Path("results") / st.session_state['session_id']
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_ZIP = f"results_{st.session_state['session_id'][:10]}.zip"

def download_image(image, label="Download Image", filename="image.jpg", mime="image/jpeg"):
    # image = Image.fromarray(image)
    buf = BytesIO()
    image.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return st.download_button(
        label=label,
        data=byte_im, 
        file_name=filename, 
        mime=mime
    )

def apply_transformation(img):
    transorm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return transorm(img)

def tensor_to_image(tensor):
    np_img = tensor.cpu().numpy()
    np_img = np.clip(np_img * 255, 0, 255).astype(np.uint8)
    return np.transpose(np_img, (1, 2, 0))

def preprocess_image(image_path, target_size=(640, 640), output_path="preprocessed.jpg"):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    image = image / 255.0
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(output_path, image)
    return output_path

def infuse_image_mask(img_path, mask_path, target_path="infused.jpg"):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    img = apply_transformation(img)
    mask = apply_transformation(mask)
    img_in = img * (1 - mask) + mask
    img_in = img_in.detach().numpy().squeeze().transpose(1, 2, 0)
    img_in = (img_in * 255).astype(np.uint8)
    img_in = Image.fromarray(img_in)
    img_in.save(target_path)
    return target_path

def add_selected_masks(mask_paths, img_path, combined_mask_path="combined_mask.jpg", combined_infused_path="combined_infused.jpg"):
    """Combine multiple mask images and update the infused image based on the combined mask."""
    combined_mask = None
    for mask_path in mask_paths:
        current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        current_mask = cv2.resize(current_mask, (640, 640), interpolation=cv2.INTER_AREA)
        if combined_mask is None:
            combined_mask = current_mask
        else:
            combined_mask = np.maximum(combined_mask, current_mask)
    cv2.imwrite(combined_mask_path, combined_mask)
    combined_infused = infuse_image_mask(img_path, combined_mask_path, target_path=combined_infused_path)
    return combined_mask, combined_infused

def mask_to_json(mask: np.ndarray, stroke_width: int = 3, 
                 stroke_color: str = "#ffffff", fill_color: str = "#ffffff") -> dict:
    """
    Convert a binary or grayscale mask (numpy array) to a Fabric.js-style JSON structure.
    Each contour is processed so that its coordinates are adjusted relative to its bounding box.
    The resulting object is positioned absolutely using the bounding box's top-left corner.
    """
    # Ensure mask is uint8
    mask_uint8 = mask.astype(np.uint8)
    # Threshold the mask (if not binary already)
    _, thresh = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for cnt in contours:
        if len(cnt) > 0:
            cnt = cnt.squeeze(1)  # shape (n, 2)
            if cnt.ndim != 2 or cnt.shape[0] < 2:
                continue  # skip too-small contours
            
            # Compute bounding box for the contour
            x_min = int(cnt[:, 0].min())
            y_min = int(cnt[:, 1].min())
            x_max = int(cnt[:, 0].max())
            y_max = int(cnt[:, 1].max())
            
            # Adjust contour coordinates relative to the bounding box
            path_commands = []
            first_point = cnt[0].tolist()
            first_point_adj = [first_point[0] - x_min, first_point[1] - y_min]
            path_commands.append(["M", first_point_adj[0], first_point_adj[1]])
            for pt in cnt[1:]:
                pt_adj = [int(pt[0] - x_min), int(pt[1] - y_min)]
                path_commands.append(["L", pt_adj[0], pt_adj[1]])
            path_commands.append(["Z"])  # Close the path
            
            obj = {
                "type": "path",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": x_min,
                "top": y_min,
                "width": x_max - x_min,
                "height": y_max - y_min,
                "fill": fill_color,
                "stroke": stroke_color,
                "strokeWidth": stroke_width,
                "strokeDashArray": None,
                "strokeLineCap": "round",
                "strokeDashOffset": 0,
                "strokeLineJoin": "round",
                "strokeUniform": False,
                "strokeMiterLimit": 10,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "flipX": False,
                "flipY": False,
                "opacity": 1,
                "shadow": None,
                "visible": True,
                "backgroundColor": "",
                "fillRule": "nonzero",
                "paintFirst": "fill",
                "globalCompositeOperation": "source-over",
                "skewX": 0,
                "skewY": 0,
                "path": path_commands,
            }
            objects.append(obj)
    return {"version": "4.4.0", "objects": objects}

def zip_results(results_folder, target_path="results.zip"):
    """Zip the contents of a folder and save the archive to a target path."""
    with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(results_folder):
            for file in files:
                file_path = os.path.join(root, file)
                # Store the file in the zip archive with a path relative to results_folder.
                arcname = os.path.relpath(file_path, results_folder)
                zipf.write(file_path, arcname)
    return target_path

def list_all_files(directory):
    """List all files in a directory and its subdirectories."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files