import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import io
import json
import cv2

# Custom CSS to override dark mode styles for download buttons
st.markdown(
    """
    <style>
    div.stDownloadButton > button {
      background-color: #007BFF !important;
      color: white !important;
      border: none !important;
      border-radius: 4px !important;
      padding: 0.5em 1em !important;
      box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls
drawing_mode = "freedraw"
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create the canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=512,
    width=512,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == "point" else 0,
    display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    key="full_app",
)


# Download Mask Image (binary mask from canvas output)
if canvas_result.image_data is not None:
    # Convert canvas image to grayscale and apply a threshold to create a binary mask
    mask_array = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
    _, binary_mask = cv2.threshold(mask_array, 254, 255, cv2.THRESH_BINARY)
    mask_img = Image.fromarray(binary_mask)
    buf_mask = io.BytesIO()
    mask_img.save(buf_mask, format="PNG")
    mask_bytes = buf_mask.getvalue()
    
    st.download_button(
         label="Download Mask Image",
         data=mask_bytes,
         file_name="mask_image.png",
         mime="image/png"
    )

# If JSON data is available, display it with a header and add a download button.
if canvas_result.json_data is not None:
    st.markdown("### JSON Output")
    objects = pd.json_normalize(canvas_result.json_data["objects"])
    for col in objects.select_dtypes(include=["object"]).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)
    
    json_str = json.dumps(canvas_result.json_data, indent=2)
    st.download_button(
         label="Download JSON Data",
         data=json_str,
         file_name="canvas_data.json",
         mime="application/json"
    )
