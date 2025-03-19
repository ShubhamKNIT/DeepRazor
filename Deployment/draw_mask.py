import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from io import BytesIO
import zipfile
from ultralytics import YOLO
from utils import (
    download_image,
    mask_to_json,
    file_selector,
    save_file_in_session,
    zip_files_in_session,
)

# --- Global Page Configuration ---
page_name = "mask_drawing"

st.title("Image Mask Drawing Tool")
st.write("Upload an image to predict its mask.")

# Upload/Browse image and (optionally) mask files.
img_file, img_mode = file_selector("image", ["jpg", "jpeg", "png"], "img", category="all", page=page_name)
mask_file, mask_mode = file_selector("mask", ["png"], "mask", category="all", page=page_name)

if img_file is not None:
    # Process the uploaded image.
    if img_mode == "Upload":
        image = Image.open(img_file).resize((640, 640)).convert("RGB")
        st.session_state["img_bytes"] = img_file.getvalue()
        save_file_in_session(img_file.name, st.session_state["img_bytes"], category="uploaded", page=page_name)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    elif img_mode == "Browse":
        image = Image.open(BytesIO(img_file)).resize((640, 640)).convert("RGB")
        st.session_state["img_bytes"] = img_file
        st.image(image, caption="Selected Image", use_container_width=True)
    
    st.write("Draw a mask on the image:")
    background_image = image

    # Canvas parameters.
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.color_picker("Stroke color: ", "#ffffff")
    bg_color = st.color_picker("Background color: ", "#000000")
    drawing_mode = st.selectbox("Drawing mode: ", ("freedraw", "line", "rect", "circle", "transform"))

    # Prepare initial drawing if a mask file is provided.
    if mask_file is not None:
        if mask_mode == "Upload":
            mask_img = Image.open(mask_file).resize((640, 640)).convert("L")
        else:
            mask_img = Image.open(BytesIO(mask_file)).resize((640, 640)).convert("L")
        mask_array = np.array(mask_img)
        initial_drawing = mask_to_json(mask_array, stroke_color=stroke_color)
    else:
        initial_drawing = None

    # Create the drawable canvas.
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=background_image,
        initial_drawing=initial_drawing,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # When the canvas returns new image data, extract the mask.
    if canvas_result.image_data is not None:
        mask_data = canvas_result.image_data[:, :, 3].astype(np.uint8)
        st.write("Extracted Mask:")
        st.image(mask_data, caption="Extracted Mask", use_container_width=True)
        mask_img_output = Image.fromarray(mask_data)

        output_mask_name = st.text_input("Enter output mask file name:", value="output_mask.png")

        # Convert the mask image to bytes.
        buf = BytesIO()
        mask_img_output.save(buf, format="PNG")
        mask_bytes = buf.getvalue()

        # Save the generated mask
        if st.button("Save Mask"):
            save_file_in_session(output_mask_name, mask_bytes, category="generated", page=page_name)

        # donwload
        download_image(mask_img_output, label="Download Mask", filename=output_mask_name, mime="image/png")
        zip_buffer = zip_files_in_session(category="all")
        st.download_button(
            label="Download All Results as Zip",
            data=zip_buffer,
            file_name="results.zip",
            mime="application/zip"
        )
