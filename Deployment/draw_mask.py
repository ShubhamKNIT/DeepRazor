import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
from utils import download_image, mask_to_json

st.title("Image Mask Drawing Tool")

# Upload image and mask files
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="img")
mask_file = st.file_uploader("Choose a mask...", type=["png"], key="mask")

if uploaded_file is not None:
    # Open and resize the background image
    image = Image.open(uploaded_file).resize((640, 640))
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Draw a mask on the image:")
    
    # Canvas parameters
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.color_picker("Stroke color: ", "#ffffff")
    bg_color = st.color_picker("Background color: ", "#000000")
    drawing_mode = st.selectbox("Drawing mode: ", ("freedraw", "line", "rect", "circle", "transform"))
    
    # Prepare the initial drawing from the uploaded mask if available.
    if mask_file is not None:
        mask_img = Image.open(mask_file).resize((640, 640)).convert("L")
        # Convert the mask image to a numpy array
        mask_array = np.array(mask_img)
        # Convert the mask to JSON data using the helper function
        initial_drawing = mask_to_json(mask_array, stroke_color=stroke_color)
    else:
        initial_drawing = None

    # Create a canvas component, passing in the initial drawing if available.
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
        
        # Download the mask as an image file
        mask_img_output = Image.fromarray(mask_data)
        download_image(mask_img_output, label="Download Mask", filename="mask.png", mime="image/png")
