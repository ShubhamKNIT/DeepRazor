import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
from utils import download_image, mask_to_json, zip_results, list_all_files, RESULT_FOLDER, RESULT_ZIP

result_folder = os.path.join(RESULT_FOLDER, "draw_mask")
os.makedirs(result_folder, exist_ok=True)
st.title("Image Mask Drawing Tool")

# Upload image and mask files
upload_browse = st.selectbox("Select an option", ["Upload", "Browse"])
if upload_browse == "Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="img") 
    mask_file = st.file_uploader("Choose a mask...", type=["png"], key="mask")
elif upload_browse == "Browse":
    uploaded_file = st.selectbox("Select an image to draw on:", list_all_files(result_folder), key="img")
    mask_file = st.selectbox("Select a mask to update:", list_all_files(result_folder), key="mask")

if uploaded_file is not None:
    # Open and resize the background image
    image = Image.open(uploaded_file).resize((640, 640)).convert("RGB")
    image.save(os.path.join(result_folder, "uploaded_image.jpg"))
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Draw a mask on the image:")
    
    # Canvas parameters
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.color_picker("Stroke color: ", "#ffffff")
    bg_color = st.color_picker("Background color: ", "#000000")
    drawing_mode = st.selectbox("Drawing mode: ", ("freedraw", "line", "rect", "circle", "transform"))
    
    # Prepare the initial drawing from mask file
    if mask_file is not None:
        mask_img = Image.open(mask_file).resize((640, 640)).convert("L")
        mask_img.save(os.path.join(result_folder, "uploaded_mask.png"))
        mask_array = np.array(mask_img)
        initial_drawing = mask_to_json(mask_array, stroke_color=stroke_color)
    else:
        initial_drawing = None

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=image,
        initial_drawing=initial_drawing,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode=drawing_mode,
        key="canvas",
    )
    
    # Extract the modified mask from the canvas result
    if canvas_result.image_data is not None:
        # Here we assume the alpha channel holds the drawn mask
        mask_data = canvas_result.image_data[:, :, 3]
        st.write("Extracted Mask:")
        st.image(mask_data, caption="Extracted Mask", use_container_width=True)
        
        # Convert the mask array into a PIL image
        mask_img_output = Image.fromarray(mask_data)
        
        # Save the mask into the results folder
        os.makedirs(result_folder, exist_ok=True)
        result_mask_path = os.path.join(result_folder, "mask.png")
        mask_img_output.save(result_mask_path)
        # st.success(f"Mask saved in results folder: {result_mask_path}")
        
        # Provide download option for the mask
        download_image(mask_img_output, label="Download Mask", filename="mask.png", mime="image/png")
        zip_path = zip_results(result_folder, target_path=RESULT_ZIP)
        with open(zip_path, "rb") as zip_file:
            st.download_button(
                label="Download All Results as Zip",
                data=zip_file,
                file_name=os.path.basename(zip_path),
                mime="application/zip"
            )