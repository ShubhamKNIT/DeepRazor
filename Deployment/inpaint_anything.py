import streamlit as st
from PIL import Image
from io import BytesIO
from utils import (
    file_selector, 
    download_image, 
    save_file_in_session, 
    zip_files_in_session,
    ONNX_MODEL_PATH,
)
from inpaint_anything_predicts import make_inference 

# Set the page name for hierarchical file storage.
page_name = "inpainting_prediction"
st.title("Inpainting Prediction Interface")

# Initialize session state for image, mask, and result files.
if "img_bytes" not in st.session_state:
    st.session_state["img_bytes"] = None
if "mask_bytes" not in st.session_state:
    st.session_state["mask_bytes"] = None
if "result_files" not in st.session_state:
    st.session_state["result_files"] = {}  # dictionary: filename -> image bytes

# Upload/Browse image and mask files.
uploaded_image, mode_img = file_selector("image", ["jpg", "jpeg", "png"], "img", category="all", page=page_name)
uploaded_mask, mode_mask = file_selector("mask", ["png"], "mask", category="all", page=page_name)

# When an image is available, display it and store its bytes.
if uploaded_image is not None:
    if mode_img == "Upload":
        img_bytes = uploaded_image.getvalue()
    else:
        img_bytes = uploaded_image
    st.session_state["img_bytes"] = img_bytes
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

# When a mask is available, display it and store its bytes.
if uploaded_mask is not None:
    if mode_mask == "Upload":
        mask_bytes = uploaded_mask.getvalue()
    else:
        mask_bytes = uploaded_mask
    st.session_state["mask_bytes"] = mask_bytes
    mask = Image.open(BytesIO(mask_bytes)).convert("L")
    st.image(mask, caption="Mask", use_container_width=True)

# Run prediction when the user clicks the button.
if st.button("Predict"):
    if st.session_state["img_bytes"] is None or st.session_state["mask_bytes"] is None:
        st.error("Please upload both an image and a mask.")
    else:
        with st.spinner("Running prediction..."):
            result_dict = make_inference(
                st.session_state["img_bytes"],
                st.session_state["mask_bytes"],
                onnx_model_path=ONNX_MODEL_PATH
            )
            # Save each result under the current page in the "generated" category.
            for fname, fbytes in result_dict.items():
                save_file_in_session(fname, fbytes, category="generated", page=page_name)
            st.session_state["result_files"] = result_dict
        st.success("Prediction complete!")

# If results exist, display them and provide download options.
if st.session_state.get("result_files"):
    st.subheader("Results")
    result_keys = list(st.session_state["result_files"].keys())
    idx = st.slider("Select result", 0, len(result_keys) - 1, 0)
    selected_key = result_keys[idx]
    result_bytes = st.session_state["result_files"][selected_key]
    result_img = Image.open(BytesIO(result_bytes))
    st.image(result_img, caption=selected_key, use_container_width=True)

    # Provide a download button for the selected result image.
    download_image(
        result_img,
        label="Download this image",
        filename=selected_key,
        mime="image/jpeg" if selected_key.lower().endswith("jpg") else "image/png"
    )

    # Create an in-memory ZIP archive for all result files across all pages.
    zip_buffer = zip_files_in_session(category="all")
    st.download_button(
        label="Download Results as Zip",
        data=zip_buffer,
        file_name="results.zip",
        mime="application/zip"
    )