import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO
import zipfile
from ultralytics import YOLO
from utils import (
    download_image,
    preprocess_image, 
    infuse_image_mask,
    add_selected_masks, 
    zip_files_in_session, 
    file_selector,
    save_file_in_session,
    retrieve_file_from_session,
    YOLO_MODEL_PATH,
)

# --- Global Page Configuration ---
page_name = "mask_prediction"
st.title("Mask Prediction Tool")

# Upload/Browse image file
st.write("Upload an image to predict its mask.")
img_file, img_mode = file_selector("image", ["jpg", "jpeg", "png"], "img", category="all", page=page_name)

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

select_all = st.checkbox("Select all classes", value=True)
if select_all:
    _class_selector = list(class_dict.values())
else:
    _class_selector = st.multiselect("Select classes to predict:", list(class_dict.values()))

if not _class_selector:
    st.warning("No classes selected. Predicting for all classes.")
    classes = None
else:
    classes = [inverse_class_dict[cls] for cls in _class_selector]

# --- Process the Uploaded Image ---
prefix = "prefix"  # default prefix
if img_file is not None:
    if img_mode == "Upload":
        image = Image.open(img_file).convert("RGB")
        prefix = img_file.name.split('.')[0]
        st.session_state.img_bytes = img_file.getvalue()
        save_file_in_session(img_file.name, st.session_state.img_bytes, category="uploaded", page=page_name)
    elif img_mode == "Browse":
        image = Image.open(BytesIO(img_file)).convert("RGB")
        prefix = st.text_input("Enter prefix for output files:", value="output")
        st.session_state.img_bytes = img_file
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Mask"):
        with st.spinner("Predicting mask..."):
            # Preprocess image (returns bytes).
            preprocessed_bytes = preprocess_image(st.session_state.img_bytes, target_size=(640, 640))
            preprocessed_image = Image.open(BytesIO(preprocessed_bytes)).convert("RGB")
            
            # Run inference with YOLO.
            model = YOLO(YOLO_MODEL_PATH)
            results = model(preprocessed_image, classes=classes)
            masks = results[0].masks
            
            mask_keys = []
            for i, mask_obj in enumerate(masks):
                mask_array = (mask_obj.data.squeeze(0).cpu().numpy().astype(np.float32) * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_array).convert("L")
                buf = BytesIO()
                mask_img.save(buf, format="JPEG")
                mask_bytes = buf.getvalue()
                # Save each predicted mask under "generated" for the current page.
                file_name = f"{prefix}_mask_{i}.jpg"
                save_file_in_session(file_name, mask_bytes, category="generated", page=page_name)
                mask_keys.append(file_name)
            st.session_state.masks_keys = mask_keys
        st.success("Inference complete!")

# --- Mask Selection & Updating Infused Image ---
if st.session_state.get("masks_keys"):
    options = [f"Mask {i}" for i in range(len(st.session_state.masks_keys))]
    selected_options = st.multiselect("Select one or more mask(s) to view or combine:", options, default=options[0])
    
    if selected_options:
        selected_indices = [int(opt.split(" ")[-1]) for opt in selected_options]
        if len(selected_indices) == 1:
            mask_bytes = retrieve_file_from_session(st.session_state.masks_keys[selected_indices[0]], category="generated", page=page_name)
            infused_bytes = infuse_image_mask(st.session_state.img_bytes, mask_bytes)
        else:
            selected_mask_bytes = [
                retrieve_file_from_session(st.session_state.masks_keys[i], category="generated", page=page_name)
                for i in selected_indices
            ]
            combined_mask_bytes, combined_infused_bytes = add_selected_masks(selected_mask_bytes, st.session_state.img_bytes)
            mask_bytes = combined_mask_bytes
            infused_bytes = combined_infused_bytes

        # Save outputs under "generated" for the current page.
        pred_mask_name = f"{prefix}_predicted_mask.jpg"
        pred_infused_name = f"{prefix}_predicted_infused.jpg"
        save_file_in_session(pred_mask_name, mask_bytes, category="generated", page=page_name)
        save_file_in_session(pred_infused_name, infused_bytes, category="generated", page=page_name)

        st.image(mask_bytes, caption="Selected/Combined Mask", use_container_width=True)
        st.image(infused_bytes, caption="Infused Image", use_container_width=True)

        mask_img = Image.open(BytesIO(mask_bytes))
        download_image(mask_img, label="Download Predicted Mask", filename=pred_mask_name, mime="image/jpeg")
        infused_img = Image.open(BytesIO(infused_bytes))
        download_image(infused_img, label="Download Predicted Infused Image", filename=pred_infused_name, mime="image/jpeg")
    
    # Create a ZIP archive for all generated files across pages.
    zip_buffer = zip_files_in_session(category="all")
    st.download_button(
        label="Download All Results as Zip",
        data=zip_buffer,
        file_name="results.zip",
        mime="application/zip"
    )
