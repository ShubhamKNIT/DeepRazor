import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from utils import apply_transformation, download_image, preprocess_image, infuse_image_mask, add_selected_masks

@st.cache_resource
def make_inference(img_path, model_path, classes=None):
    model = YOLO(model_path)
    # If classes is None, YOLO predicts all classes.
    results = model(img_path, classes=classes)
    return results

def predict_mask(img_path, model_path="./Deployment/yolo11m-seg.pt", classes=None):
    preprocessed_img_path = preprocess_image(img_path)
    results = make_inference(preprocessed_img_path, model_path, classes=classes)
    masks = results[0].masks
    masks_paths = []
    for i, mask_obj in enumerate(masks):
        mask_path = f"mask_{i}.jpg"
        mask_img = Image.fromarray(
            (mask_obj.data.squeeze(0).cpu().numpy().astype(np.float32) * 255).astype(np.uint8)
        ).convert("L")
        mask_img.save(mask_path)
        masks_paths.append(mask_path)
    return masks_paths

# ----- Streamlit App -----
st.title("Mask Prediction Tool")
st.write("Upload an image to predict its mask.")

# Initialize session state variables
if "masks_paths" not in st.session_state:
    st.session_state.masks_paths = None
if "img_path" not in st.session_state:
    st.session_state.img_path = None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Class Selection with "Select All" Option ---
class_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
    26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
    71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}
inverse_class_dict = {v: k for k, v in class_dict.items()}

# Checkbox to quickly select all classes
select_all = st.checkbox("Select all classes", value=True)
if select_all:
    _class_selector = list(class_dict.values())
else:
    _class_selector = st.multiselect("Select classes to predict:", list(class_dict.values()))

# Prevent none errors due to classes: if no class is selected, default to all classes.
if not _class_selector:
    st.warning("No classes selected. Predicting for all classes.")
    classes = None
else:
    classes = [inverse_class_dict[cls] for cls in _class_selector]

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    img_path = uploaded_file.name
    image.save(img_path)
    st.session_state.img_path = img_path
    if st.button("Predict Mask"):
        with st.spinner("Running inference..."):
            masks_paths = predict_mask(img_path, classes=classes)
        st.session_state.masks_paths = masks_paths
        st.success("Inference complete!")

# --- Mask Selection & Updating Infused Image ---
if st.session_state.masks_paths is not None:
    # Create a list of mask options (e.g., "Mask 0", "Mask 1", ...)
    options = [f"Mask {i}" for i in range(len(st.session_state.masks_paths))]
    selected_options = st.multiselect("Select one or more mask(s) to view or combine:", options, default=options[0])
    
    if selected_options:
        # Convert selections to indices
        selected_indices = [int(option.split(" ")[-1]) for option in selected_options]
        if len(selected_indices) == 1:
            # Single mask selection: update the infused image for the selected mask.
            selected_mask_path = st.session_state.masks_paths[selected_indices[0]]
            st.image(selected_mask_path, caption=f"Selected Mask {selected_indices[0]}", use_container_width=True)
            infused_path = infuse_image_mask(st.session_state.img_path, selected_mask_path, target_path=f"infused_{selected_indices[0]}.jpg")
            st.image(infused_path, caption=f"Infused Image for Mask {selected_indices[0]}", use_container_width=True)
            mask = Image.open(selected_mask_path)
            download_image(mask, label="Download Selected Mask", filename=f"mask_{selected_indices[0]}.jpg", mime="image/jpeg")
            infused_img = Image.open(infused_path)
            download_image(infused_img, label="Download Infused Image", filename=f"infused_{selected_indices[0]}.jpg", mime="image/jpeg")
        else:
            # Multiple mask selection: combine masks and update the infused image.
            selected_mask_paths = [st.session_state.masks_paths[i] for i in selected_indices]
            combined_mask, combined_infused_path = add_selected_masks(selected_mask_paths, st.session_state.img_path)
            combined_mask_img = Image.fromarray(combined_mask.astype(np.uint8))
            st.image(combined_mask_img, caption="Combined Mask", use_container_width=True)
            st.image(combined_infused_path, caption="Infused Image for Combined Mask", use_container_width=True)
            download_image(combined_mask_img, label="Download Combined Mask", filename="combined_mask.jpg", mime="image/jpeg")
            combined_infused_img = Image.open("combined_infused.jpg")
            download_image(combined_infused_img, label="Download Combined Infused Image", filename="combined_infused.jpg", mime="image/jpeg")
