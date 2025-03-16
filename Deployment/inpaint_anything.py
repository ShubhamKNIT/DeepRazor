import os
import shutil
import streamlit as st
from PIL import Image
from utils import zip_results, list_all_files, RESULT_FOLDER, RESULT_ZIP
from inpaint_anything_predicts import make_inference

# Define global folder for results
result_folder = os.path.join(RESULT_FOLDER, "inpainting_anything")
os.makedirs(result_folder, exist_ok=True)
st.title("Inpainting Prediction Interface")

# Initialize session state variables
if "result_files" not in st.session_state:
    st.session_state.result_files = None
if "save_folder" not in st.session_state:
    st.session_state.save_folder = None

# Upload image and mask files
upload_browse = st.selectbox("Select an option", ["Upload", "Browse"])
if upload_browse == "Upload":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    uploaded_mask = st.file_uploader("Upload a Mask", type=["jpg", "jpeg", "png"])
elif upload_browse == "Browse":
    uploaded_image = st.selectbox("Select an image to inpaint:", list_all_files(RESULT_FOLDER))
    uploaded_mask = st.selectbox("Select a mask to inpaint:", list_all_files(RESULT_FOLDER))

# Clear session state if image is removed
if uploaded_image is None:
    st.session_state.result_files = None
    st.session_state.save_folder = None

# Run prediction if both image and mask are uploaded
if st.button("Predict"):
    if uploaded_image is None or uploaded_mask is None:
        st.error("Please upload both an image and a mask.")
    else:
        # Create a temporary folder for uploads
        temp_folder = "temp_uploads"
        os.makedirs(temp_folder, exist_ok=True)
        
        # Save uploaded files to disk
        if upload_browse == "Upload":
            image_path = os.path.join(temp_folder, uploaded_image.name)
            mask_path = os.path.join(temp_folder, uploaded_mask.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getvalue())
            with open(mask_path, "wb") as f:
                f.write(uploaded_mask.getvalue())
        elif upload_browse == "Browse":
            image_path = uploaded_image
            mask_path = uploaded_mask

        st.image(Image.open(image_path), caption="Original Image", use_container_width=True)
        st.image(Image.open(mask_path), caption="Mask", use_container_width=True)
        
        # Set ONNX model path and create a unique results folder for this run
        onnx_model_path = "onnx_gen_models/ia_gen_55.onnx"
        # Save results in a subfolder of RESULT_FOLDER
        save_folder = result_folder 
        os.makedirs(save_folder, exist_ok=True)
        
        st.info("Running prediction... Please wait.")
        make_inference(image_path, mask_path, onnx_model_path, save_folder)
        st.success("Prediction complete!")
        
        # Clean up temporary uploads
        shutil.rmtree(temp_folder)
        
        # Save result file names in session state
        result_files = sorted([f for f in os.listdir(save_folder) if f.lower().endswith((".jpg", ".png"))])
        st.session_state.result_files = result_files
        st.session_state.save_folder = save_folder

if st.session_state.result_files is not None and st.session_state.save_folder is not None:
    st.subheader("Results")
    num_results = len(st.session_state.result_files)
    idx = st.slider("Select result", 0, num_results - 1, 0)
    
    # Construct the path to the selected result image
    result_image_path = os.path.join(st.session_state.save_folder, st.session_state.result_files[idx])
    result_img = Image.open(result_image_path)
    st.image(result_img, caption=st.session_state.result_files[idx], use_container_width=True)
    
    # Provide download button for individual result image
    with open(result_image_path, "rb") as file:
        st.download_button(
            label="Download this image",
            data=file,
            file_name=st.session_state.result_files[idx],
            mime="image/jpeg" if st.session_state.result_files[idx].lower().endswith("jpg") else "image/png"
        )
    
    # Provide a ZIP download button for all results
    zip_path = zip_results(RESULT_FOLDER, target_path=RESULT_ZIP)
    with open(zip_path, "rb") as zip_file:
        st.download_button(
            label="Download All Results as Zip",
            data=zip_file,
            file_name=os.path.basename(zip_path),
            mime="application/zip"
        )
