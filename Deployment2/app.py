import os
import glob
import torch
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
from streamlit_image_coordinates import streamlit_image_coordinates
from remove_anything_auto import mask_and_inpaint
from checkpoint_downloader import download_all_checkpoints

# Configure page
st.set_page_config(
    page_title="Point-And-Click Inpainting",
    page_icon="🔨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define checkpoint directory
CHECKPOINTS_DIR = './checkpoints'

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .results-section {
        margin-top: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .coordinate-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .fixed-image {
        max-width: 512px;
        width: 100%;
        height: auto;
    }
</style>
""", unsafe_allow_html=True)

# ——— Checkpoint Check and Download Function ———
@st.cache_resource
def check_and_download_checkpoints():
    """Check if checkpoints exist, download if not"""
    ckpt_root = Path(CHECKPOINTS_DIR)
    
    if not (ckpt_root.exists() and any(ckpt_root.iterdir())):
        st.info(f"⚠️ '{CHECKPOINTS_DIR}' not found or empty. Downloading now...")
        
        # Show progress during download
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with progress_placeholder.container():
            progress_bar = st.progress(0)
            
        with status_placeholder.container():
            status_text = st.empty()
            
        try:
            status_text.text("🔄 Downloading SAM2.1 model weights...")
            progress_bar.progress(25)
            
            status_text.text("🔄 Downloading SAM2.1 configuration...")
            progress_bar.progress(50)
            
            status_text.text("🔄 Downloading Big-Lama model...")
            progress_bar.progress(75)
            
            # Download checkpoints
            result = download_all_checkpoints(CHECKPOINTS_DIR)
            
            if result["status"] == "success":
                progress_bar.progress(100)
                status_text.text("✅ All checkpoints downloaded successfully!")
                st.success(f"✔️ Checkpoints downloaded to '{CHECKPOINTS_DIR}'")
                
                # Clear progress indicators
                progress_placeholder.empty()
                status_placeholder.empty()
                
                return True
            else:
                st.error("❌ Failed to download checkpoints. Please check your internet connection.")
                return False
                
        except Exception as e:
            st.error(f"❌ Error downloading checkpoints: {str(e)}")
            return False
    else:
        st.success(f"✔️ Found existing '{CHECKPOINTS_DIR}', skipping download.")
        return True

def get_image_dimensions(image_path):
    """Get original image dimensions"""
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)

def transform_coordinates(click_x, click_y, original_width, original_height, display_width):
    """Transform coordinates from display image to original image"""
    # Calculate the scaling factor
    scale_factor = original_width / display_width
    
    # Calculate display height maintaining aspect ratio
    display_height = int(original_height / scale_factor)
    
    # Transform coordinates
    original_x = int(click_x * scale_factor)
    original_y = int(click_y * scale_factor)
    
    # Ensure coordinates are within bounds
    original_x = max(0, min(original_x, original_width - 1))
    original_y = max(0, min(original_y, original_height - 1))
    
    return original_x, original_y, scale_factor

def add_click_dot_to_image(image_path, x, y, dot_color="red", dot_size=8):
    """Add a colored dot to the image at the clicked coordinates (original size only)."""
    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)

        adjusted_dot_size = max(4, min(dot_size, img.width // 100))

        left = x - adjusted_dot_size
        top = y - adjusted_dot_size
        right = x + adjusted_dot_size
        bottom = y + adjusted_dot_size

        draw.ellipse([left - 2, top - 2, right + 2, bottom + 2], fill="white", outline="white")
        draw.ellipse([left, top, right, bottom], fill=dot_color, outline=dot_color)

        return img
    except Exception as e:
        st.error(f"Error adding dot to image: {str(e)}")
        return Image.open(image_path)


# ——— Main UI ———
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🔨 Point‑And‑Click Inpainting")
st.markdown("**Remove unwanted objects from your images with a simple click!**")
st.markdown('</div>', unsafe_allow_html=True)

# ——— Checkpoint Verification ———
st.subheader("🔧 Model Setup")
checkpoints_ready = check_and_download_checkpoints()

if not checkpoints_ready:
    st.error("❌ Cannot proceed without model checkpoints. Please check your internet connection and refresh the page.")
    st.stop()

# ——— Sidebar for settings ———
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Checkpoint status
    st.subheader("📦 Model Status")
    ckpt_root = Path(CHECKPOINTS_DIR)
    if ckpt_root.exists():
        checkpoint_files = list(ckpt_root.glob("*"))
        st.success(f"✅ {len(checkpoint_files)} files loaded")
        
        with st.expander("View Checkpoint Files"):
            for file in checkpoint_files:
                file_size = file.stat().st_size / (1024*1024)  # MB
                st.text(f"📄 {file.name} ({file_size:.1f} MB)")
    
    # Image quality settings
    st.subheader("Output Quality")
    image_quality = st.slider("JPEG Quality", 70, 100, 95)
    
    # Display settings
    st.subheader("Display Options")
    fixed_width = st.slider("Image Display Width", 256, 1024, 512)
    dot_color = st.selectbox("Click Dot Color", ["red", "blue", "green", "yellow", "purple"], index=0)
    auto_download = st.checkbox("Auto-download results", value=False)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        temp_cleanup = st.checkbox("Auto-cleanup temp files", value=True)
        max_image_size = st.number_input("Max image size (pixels)", 512, 2048, 1024)

# ——— File Upload Section ———
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.subheader("📁 Upload Your Image")

uploaded = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg", "webp"],
    help="Supported formats: PNG, JPG, JPEG, WebP"
)

if uploaded:
    # Display file info
    file_size = len(uploaded.getvalue()) / (1024 * 1024)  # MB
    st.success(f"✅ Uploaded: {uploaded.name} ({file_size:.2f} MB)")
else:
    st.info("👆 Please upload an image to get started")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# ——— Image Processing ———
# Create temp directory and save file
temp_dir = "temp"
output_dir = "output"
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

img_path = os.path.join(temp_dir, uploaded.name)
uploaded_file_name = os.path.splitext(uploaded.name)[0]

# Save uploaded file
with open(img_path, "wb") as f:
    f.write(uploaded.getbuffer())

# Get original image dimensions
original_width, original_height = get_image_dimensions(img_path)

# ——— Image Display and Interaction ———
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🖱️ Click to Select Object")
    st.markdown("**Click on the object you want to remove from the image:**")
    
    # Display image with click coordinates
    coords = streamlit_image_coordinates(
        img_path, 
        key="img_click",
        width=fixed_width
    )

with col2:
    st.subheader("🎯 Actions")
    
    if coords:
        # Handle coordinate data
        if isinstance(coords, dict):
            display_x, display_y = coords["x"], coords["y"]
            timestamp = coords.get("timestamp") or coords.get("time")
        else:
            display_x, display_y, timestamp = coords
        
        # Transform coordinates to original image space
        original_x, original_y, scale_factor = transform_coordinates(
            display_x, display_y, original_width, original_height, fixed_width
        )
        
        st.success(f"📍 Click detected")
        st.info(f"Display: ({display_x}, {display_y})")
        st.info(f"Original: ({original_x}, {original_y})")
        st.info(f"Scale: {scale_factor:.2f}x")
        
        # Process button
        if st.button("🚀 Remove Object", type="primary", use_container_width=True):
            # Show processing progress
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 Initializing models...")
                    progress_bar.progress(20)
                    
                    # Use ORIGINAL coordinates for processing
                    latest_coords = torch.tensor([[original_x, original_y]], dtype=torch.float32)
                    point_labels = torch.tensor([1], dtype=torch.int64)
                    
                    status_text.text("🎭 Generating mask...")
                    progress_bar.progress(50)
                    
                    # Call the inpainting function with original coordinates
                    mask_and_inpaint(img_path, latest_coords, point_labels)
                    
                    status_text.text("🎨 Inpainting image...")
                    progress_bar.progress(80)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    st.success("🎉 Object removed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                finally:
                    # Clear progress indicators after a moment
                    import time
                    time.sleep(1)
                    progress_container.empty()
    else:
        st.info("👆 Click on the image to select a point")

# ——— Results Display ———
if coords:
    subfolder_path = os.path.join(output_dir, uploaded_file_name)
    
    if os.path.exists(subfolder_path):
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.subheader("🎨 Results")
        
        # Find result files
        mask_files = sorted(glob.glob(os.path.join(subfolder_path, "with_mask_*.png")))
        inpaint_files = sorted(glob.glob(os.path.join(subfolder_path, f"{uploaded_file_name}_inpainted_mask_*.jpg")))
        
        # Create tabs for different result types
        tab1, tab2, tab3 = st.tabs(["🖼️ Original", "🎭 Masks", "✨ Inpainted"])
        
        with tab1:
            # Show original image with click dot at ORIGINAL coordinates
            if isinstance(coords, dict):
                display_x, display_y = coords["x"], coords["y"]
            else:
                display_x, display_y = coords[0], coords[1]
            
            # Transform coordinates
            original_x, original_y, _ = transform_coordinates(
                display_x, display_y, original_width, original_height, fixed_width
            )
            
            # Create image with dot at original coordinates
            img_with_dot = add_click_dot_to_image(
                img_path, original_x, original_y
            )
            st.image(img_with_dot, caption="Original Image (with click point)", width=fixed_width)
        
        with tab2:
            if mask_files:
                if len(mask_files) == 1:
                    st.image(mask_files[0], caption="Generated Mask", use_container_width=True)
                else:
                    cols = st.columns(min(len(mask_files), 3))
                    for i, mask_file in enumerate(mask_files):
                        with cols[i % 3]:
                            st.image(mask_file, caption=f"Mask {i+1}", use_container_width=True)
            else:
                st.warning("⚠️ No mask images found")
        
        with tab3:
            if inpaint_files:
                if len(inpaint_files) == 1:
                    st.image(inpaint_files[0], caption="Inpainted Result", use_container_width=True)
                    
                    # Download button
                    if auto_download or st.button("📥 Download Result"):
                        with open(inpaint_files[0], "rb") as file:
                            st.download_button(
                                label="💾 Download Inpainted Image",
                                data=file.read(),
                                file_name=f"inpainted_{uploaded.name}",
                                mime="image/jpeg"
                            )
                else:
                    cols = st.columns(min(len(inpaint_files), 3))
                    for i, inpaint_file in enumerate(inpaint_files):
                        with cols[i % 3]:
                            st.image(inpaint_file, caption=f"Result {i+1}", use_container_width=True)
            else:
                st.warning("⚠️ No inpainted images found")
        
#         st.markdown('</div>', unsafe_allow_html=True)

# ——— Cleanup ———
if temp_cleanup and coords:
    # Clean up temporary files (optional)
    try:
        if os.path.exists(img_path):
            os.remove(img_path)
    except:
        pass  # Ignore cleanup errors

# ——— Footer ———
st.markdown("---")
st.markdown(
    "💡 **Tip:** For best results, click on the center of the object you want to remove. "
    "The AI will automatically detect and mask the entire object. The colored dot shows your click location."
)
