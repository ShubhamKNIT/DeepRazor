import os
import sys
import torch
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from streamlit_image_coordinates import streamlit_image_coordinates
from utils import (
    save_file_in_session,
    retrieve_file_from_session,
    list_files_in_session,
    zip_files_in_session,
    file_selector,
)

# ——— Main Settings ———
PAGE_NAME = "point_click_inpainting"

if "processed_results" not in st.session_state:
    st.session_state["processed_results"] = {}
if "current_coords" not in st.session_state:
    st.session_state["current_coords"] = None

# ——— Mock Inpainting Function ———
def mask_and_inpaint(img_path, coordinates, point_labels, mask_radius, blur_strength):
    """
    Creates a circular mask around (x,y) with radius=mask_radius,
    then blurs that region by blur_strength and saves both mask + result.
    """
    try:
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)
        x, y = int(coordinates[0][0]), int(coordinates[0][1])

        # Build mask
        mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        yv, xv = np.ogrid[:img_array.shape[0], :img_array.shape[1]]
        condition = (xv - x) ** 2 + (yv - y) ** 2 <= mask_radius ** 2
        mask[condition] = 255

        # Blur masked area
        from scipy import ndimage
        blurred = ndimage.gaussian_filter(img_array, sigma=blur_strength)
        mask_3d = np.stack([mask] * 3, axis=-1) / 255.0
        inpainted = img_array * (1 - mask_3d) + blurred * mask_3d

        # Save under output/<uploaded_name>/
        uploaded_file_name = os.path.splitext(os.path.basename(img_path))[0]
        output_dir = os.path.join("output", uploaded_file_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save mask PNG
        mask_img = Image.fromarray(mask)
        mask_name = f"with_mask_{x}_{y}.png"
        mask_img.save(os.path.join(output_dir, mask_name))

        # Save inpainted JPG
        inpainted_img = Image.fromarray(inpainted.astype(np.uint8))
        inpaint_name = f"{uploaded_file_name}_inpainted_{x}_{y}.jpg"
        inpainted_img.save(os.path.join(output_dir, inpaint_name))

        # Store bytes in session
        buf = BytesIO()
        mask_img.save(buf, format="PNG")
        save_file_in_session(mask_name, buf.getvalue(), category="generated", page=PAGE_NAME)

        buf = BytesIO()
        inpainted_img.save(buf, format="JPEG")
        save_file_in_session(inpaint_name, buf.getvalue(), category="generated", page=PAGE_NAME)

        return True
    except Exception as e:
        st.error(f"Error in mask_and_inpaint: {e}")
        return False

# ——— App Header ———
st.title("🔨 Point-And-Click Inpainting Demo")
st.markdown("Remove unwanted objects by clicking on them below.")
st.sidebar.header("⚙️ Settings")

# ——— Sidebar Controls ———
show_coords = st.sidebar.checkbox("Show click coordinates overlay", value=True)
auto_process = st.sidebar.checkbox("Auto-process on click", value=False)

mask_radius = st.sidebar.slider("Mask radius", 10, 200, 50)
blur_strength = st.sidebar.slider("Blur strength", 1, 20, 5)

if st.sidebar.button("Clear All Results"):
    st.session_state["processed_results"] = {}
    st.experimental_rerun()

# ——— File Upload / Selection ———
uploaded_file, upload_mode = file_selector(
    "image", ["png", "jpg", "jpeg", "webp"], "img", category="all", page=PAGE_NAME
)
if not uploaded_file:
    st.info("Please upload or select an image to get started.")
    st.stop()

# Save into temp/ so we can resize and pass to streamlit_image_coordinates
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

if upload_mode == "Upload":
    orig_path = os.path.join(temp_dir, uploaded_file.name)
    with open(orig_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    uploaded_file_name = os.path.splitext(uploaded_file.name)[0]
else:
    orig_path = os.path.join(temp_dir, "selected_image.jpg")
    with open(orig_path, "wb") as f:
        f.write(uploaded_file)
    uploaded_file_name = "selected_image"

st.success(f"✅ Loaded: {uploaded_file_name}")

# ——— Resize Logic ———
DISPLAY_WIDTH = 600
img = Image.open(orig_path)
w, h = img.size

if w > DISPLAY_WIDTH:
    new_h = int((DISPLAY_WIDTH / w) * h)
    resized = img.resize((DISPLAY_WIDTH, new_h), Image.LANCZOS)
else:
    resized = img.copy()

resized_path = os.path.join(temp_dir, f"resized_{uploaded_file_name}.png")
resized.save(resized_path)

# ——— Single‐Image Placeholder ———
st.subheader("🖱️ Click to Mask")
placeholder = st.empty()

# Render the clickable image inside this same placeholder
with placeholder.container():
    coords = streamlit_image_coordinates(resized_path, key="img_click")

# If user clicked, draw marker + replace the image in the same placeholder
if coords:
    if isinstance(coords, dict):
        x_disp, y_disp = coords["x"], coords["y"]
        ts = coords.get("timestamp") or coords.get("time")
    else:
        x_disp, y_disp, ts = coords

    st.session_state["current_coords"] = (x_disp, y_disp)
    overlay = resized.copy()
    draw = ImageDraw.Draw(overlay)

    # Draw a small red circle
    r = 5
    draw.ellipse(
        [(x_disp - r, y_disp - r), (x_disp + r, y_disp + r)],
        fill="red",
    )

    # Draw coordinates text if requested
    if show_coords:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        text = f"({int(x_disp)}, {int(y_disp)})"
        draw.text((x_disp + 8, y_disp - 8), text, fill="red", font=font)

    # Replace the placeholder with the overlaid image
    placeholder.image(overlay, use_column_width=False, width=DISPLAY_WIDTH)
    st.markdown(f"*Clicked at ({int(x_disp)}, {int(y_disp)})*")

    # Convert back to original‐scale coordinates
    scale_x = w / resized.width
    scale_y = h / resized.height
    orig_x = int(x_disp * scale_x)
    orig_y = int(y_disp * scale_y)

    # “Remove Object” button appears below the same image
    should_process = auto_process or st.button("🚀 Remove Object", use_container_width=True)
    if should_process:
        with st.spinner("Processing..."):
            latest_coords = torch.tensor([[orig_x, orig_y]], dtype=torch.float32)
            point_labels = torch.tensor([1], dtype=torch.int64)

            success = mask_and_inpaint(
                orig_path,
                latest_coords,
                point_labels,
                mask_radius,
                blur_strength,
            )
            if success:
                st.success("✅ Done!")
                st.session_state["processed_results"][f"{orig_x}_{orig_y}"] = {
                    "x": orig_x, "y": orig_y, "timestamp": ts
                }
            else:
                st.error("❌ Inpainting failed")
else:
    # If no click yet, nothing replaces the placeholder; it remains showing the clickable image
    pass

# ——— Results Tabs ———
if st.session_state["processed_results"]:
    st.subheader("🎨 Results")
    generated = list_files_in_session(category="generated", page=PAGE_NAME)
    mask_files = [f for f in generated if f.startswith("with_mask_")]
    inpainted_files = [
        f for f in generated if f.startswith(f"{uploaded_file_name}_inpainted")
    ]

    tab_orig, tab_masks, tab_inp = st.tabs(["🖼️ Original", "🎭 Masks", "✨ Inpainted"])
    with tab_orig:
        st.image(orig_path, caption="Original", use_column_width=True)
        with open(orig_path, "rb") as f:
            st.download_button(
                "📥 Download Original",
                data=f.read(),
                file_name=f"original_{uploaded_file_name}.jpg",
                mime="image/jpeg",
            )

    with tab_masks:
        if mask_files:
            cols = st.columns(min(len(mask_files), 3))
            for idx, mask_name in enumerate(mask_files):
                buf = retrieve_file_from_session(
                    mask_name, category="generated", page=PAGE_NAME
                )
                if buf:
                    with cols[idx % 3]:
                        st.image(buf, caption=f"Mask {idx+1}", use_column_width=True)
                        st.download_button(
                            f"Download Mask {idx+1}",
                            data=buf,
                            file_name=mask_name,
                            mime="image/png",
                            key=f"dl_mask_{idx}",
                        )
        else:
            st.warning("No masks found.")

    with tab_inp:
        if inpainted_files:
            cols = st.columns(min(len(inpainted_files), 3))
            for idx, inp_name in enumerate(inpainted_files):
                buf = retrieve_file_from_session(
                    inp_name, category="generated", page=PAGE_NAME
                )
                if buf:
                    with cols[idx % 3]:
                        st.image(buf, caption=f"Result {idx+1}", use_column_width=True)
                        st.download_button(
                            f"Download Result {idx+1}",
                            data=buf,
                            file_name=inp_name,
                            mime="image/jpeg",
                            key=f"dl_inp_{idx}",
                        )
        else:
            st.warning("No inpainted results found.")

# ——— Download Options ———
st.subheader("📦 Download All")
col1, col2, col3 = st.columns(3)
with col1:
    page_zip = zip_files_in_session(category="generated", page=PAGE_NAME)
    st.download_button(
        "Download Page Results",
        data=page_zip,
        file_name=f"{PAGE_NAME}_results.zip",
        mime="application/zip",
    )
with col2:
    all_zip = zip_files_in_session(category="generated")
    st.download_button(
        "Download All Generated",
        data=all_zip,
        file_name="all_generated_results.zip",
        mime="application/zip",
    )
with col3:
    all_data = zip_files_in_session(category="all")
    st.download_button(
        "Download Everything",
        data=all_data,
        file_name="complete_results.zip",
        mime="application/zip",
    )

# ——— History & Footer ———
if st.session_state["processed_results"]:
    with st.expander("📊 Processing History"):
        for key, data in st.session_state["processed_results"].items():
            x, y, ts = data["x"], data["y"], data.get("timestamp", "")
            st.write(f"• Click ({x}, {y}) at {ts}")

st.markdown("---")
st.markdown(
    "💡 Tip: Click near the center of the unwanted object for best results. "
    "Adjust the mask radius and blur strength from the sidebar."
)