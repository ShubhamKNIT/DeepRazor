import streamlit as st
import zipfile
from io import BytesIO
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import os

ONNX_MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "onnx_gen_models/ia_gen_55.onnx")
YOLO_MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "yolo11m-seg.pt")

# -----------------------------
# Global: Set the current page name.
# -----------------------------
page_name = "mask_prediction"  # Change as needed for different pages.

# Initialize the hierarchical file storage in session state.
if "files" not in st.session_state:
    st.session_state["files"] = {}
if page_name not in st.session_state["files"]:
    st.session_state["files"][page_name] = {"uploaded": {}, "generated": {}}

#########################################
# Session State File Management Helpers #
#########################################

def save_file_in_session(file_name, file_content, category="generated", page=None):
    """
    Save a file's content in session state under a specific page and category.
    
    :param file_name: Name to use as key.
    :param file_content: File content (typically bytes).
    :param category: Either "uploaded" or "generated".
    :param page: Page name; if None, use global page_name.
    """
    if page is None:
        page = page_name
    if "files" not in st.session_state:
        st.session_state["files"] = {}
    if page not in st.session_state["files"]:
        st.session_state["files"][page] = {"uploaded": {}, "generated": {}}
    if category not in st.session_state["files"][page]:
        st.session_state["files"][page][category] = {}
    st.session_state["files"][page][category][file_name] = file_content

def retrieve_file_from_session(file_name, category="generated", page=None):
    """
    Retrieve a file's content from session state.
    
    :param file_name: Name/key of the file. For category "all", expected format is "page/category/filename".
    :param category: Either "uploaded", "generated", or "all".
    :param page: Page name; if None, use global page_name.
    :return: The file content if it exists, else None.
    """
    if file_name is None:
        return None

    files = st.session_state.get("files", {})
    if category == "all":
        if "/" in file_name:
            parts = file_name.split("/")
            if len(parts) >= 3:
                pg = parts[0]
                cat = parts[1]
                fname = "/".join(parts[2:])
                return files.get(pg, {}).get(cat, {}).get(fname, None)
        # Fallback search:
        for pg in files:
            for cat in files[pg]:
                if file_name in files[pg][cat]:
                    return files[pg][cat][file_name]
        return None
    elif category in ["uploaded", "generated"]:
        if page is None:
            page = page_name
        return files.get(page, {}).get(category, {}).get(file_name, None)
    else:
        st.warning(f"Invalid category: {category}. Use 'uploaded', 'generated', or 'all'.")
        return None

def list_files_in_session(category="generated", page=None):
    """
    List all file names stored in session state for a given category.
    
    :param category: Either "uploaded", "generated", or "all".
    :param page: Page name; if None, use global page_name.
    :return: List of file names. For "all", include subdirectory paths.
    """
    files = st.session_state.get("files", {})
    if category == "all":
        file_list = []
        for pg in files:
            for cat in files[pg]:
                for fname in files[pg][cat].keys():
                    file_list.append(f"{pg}/{cat}/{fname}")
        return file_list
    else:
        if page is None:
            page = page_name
        return list(st.session_state.get("files", {}).get(page, {}).get(category, {}).keys())

def zip_files_in_session(category="generated", page=None):
    """
    Zip files stored in session state.
    
    :param category: "uploaded", "generated", or "all".
    :param page: If provided (and category is not "all"), zip only files under that page.
    :return: A BytesIO object containing the zip archive.
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        files = st.session_state.get("files", {})
        if category == "all":
            # Zip files across all pages, using subdirectories.
            for pg in files:
                for cat in files[pg]:
                    for fname, fcontent in files[pg][cat].items():
                        zipf.writestr(f"{pg}/{cat}/{fname}", fcontent)
        else:
            if page is None:
                page = page_name
            for fname, fcontent in st.session_state.get("files", {}).get(page, {}).get(category, {}).items():
                zipf.writestr(f"{category}/{fname}", fcontent)
    zip_buffer.seek(0)
    return zip_buffer

def file_selector(label, file_types, key, category="uploaded", page=None):
    """
    Allow users to either upload a file or browse existing ones stored in session state.
    When uploading, the user is prompted to enter a custom file name.
    
    Returns a tuple of (file content bytes or file uploader object, mode used).
    If category is "all", browsing will list files from all pages, while uploading will default to "uploaded" under current page.
    """
    if page is None:
        page = page_name
    option = st.selectbox(f"Select an option for {label}", ["Upload", "Browse"], key=f"{key}_option")
    if category == "all":
        available_files = list_files_in_session("all")
    else:
        available_files = list_files_in_session(category, page=page)
    
    if option == "Upload":
        uploaded_file = st.file_uploader(f"Choose a {label}...", type=file_types, key=key)
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            custom_name = st.text_input(
                f"Enter file name to save for this {label}",
                value=uploaded_file.name,
                key=f"{key}_filename"
            )
            # If category is "all", default to saving under "uploaded" for the current page.
            save_category = "uploaded" if category == "all" else category
            save_file_in_session(custom_name, file_bytes, category=save_category, page=page)
        return uploaded_file, "Upload"
    elif option == "Browse":
        selected_file = st.selectbox(f"Select a {label}:", available_files, key=key)
        file_bytes = None
        if category == "all":
            file_bytes = retrieve_file_from_session(selected_file, category="all")
        else:
            file_bytes = retrieve_file_from_session(selected_file, category=category, page=page)
        return file_bytes, "Browse"

def download_image(image, label="Download Image", filename="image.jpg", mime="image/jpeg"):
    buf = BytesIO()
    image.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return st.download_button(
        label=label,
        data=byte_im, 
        file_name=filename, 
        mime=mime
    )

#####################################
# Other Utility Functions (unchanged)
#####################################

def apply_transformation(img):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return transform(img)

def tensor_to_image(tensor):
    np_img = tensor.cpu().numpy()
    np_img = np.clip(np_img * 255, 0, 255).astype(np.uint8)
    return np.transpose(np_img, (1, 2, 0))

def preprocess_image(image_bytes, target_size=(640, 640)):
    """
    Preprocess an image provided as bytes.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    image = image / 255.0
    image = (image * 255).astype(np.uint8)
    success, encoded_image = cv2.imencode('.jpg', image)
    return encoded_image.tobytes() if success else None

def infuse_image_mask(img_bytes, mask_bytes):
    """
    Infuse an image with a mask. Both are provided as bytes.
    Returns the infused image as bytes.
    """
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    mask = Image.open(BytesIO(mask_bytes)).convert("L")
    img_trans = apply_transformation(img)
    mask_trans = apply_transformation(mask)
    img_in = img_trans * (1 - mask_trans) + mask_trans
    img_in = img_in.detach().numpy().squeeze().transpose(1, 2, 0)
    img_in = (img_in * 255).astype(np.uint8)
    infused_img = Image.fromarray(img_in)
    buf = BytesIO()
    infused_img.save(buf, format="JPEG")
    return buf.getvalue()

def add_selected_masks(mask_bytes_list, img_bytes):
    """
    Combine multiple mask images (each provided as bytes) and update the infused image
    based on the combined mask. Returns combined mask bytes and infused image bytes.
    """
    combined_mask = None
    for mask_bytes in mask_bytes_list:
        nparr = np.frombuffer(mask_bytes, np.uint8)
        current_mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        current_mask = cv2.resize(current_mask, (640, 640), interpolation=cv2.INTER_AREA)
        if combined_mask is None:
            combined_mask = current_mask
        else:
            combined_mask = np.maximum(combined_mask, current_mask)
    success, combined_mask_encoded = cv2.imencode('.jpg', combined_mask)
    combined_mask_bytes = combined_mask_encoded.tobytes() if success else None
    combined_infused_bytes = infuse_image_mask(img_bytes, combined_mask_bytes)
    return combined_mask_bytes, combined_infused_bytes

def mask_to_json(mask: np.ndarray, stroke_width: int = 3, 
                 stroke_color: str = "#ffffff", fill_color: str = "#ffffff") -> dict:
    """
    Convert a binary or grayscale mask (numpy array) to a Fabric.js-style JSON structure.
    """
    mask_uint8 = mask.astype(np.uint8)
    _, thresh = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for cnt in contours:
        if len(cnt) > 0:
            cnt = cnt.squeeze(1)
            if cnt.ndim != 2 or cnt.shape[0] < 2:
                continue
            x_min = int(cnt[:, 0].min())
            y_min = int(cnt[:, 1].min())
            x_max = int(cnt[:, 0].max())
            y_max = int(cnt[:, 1].max())
            path_commands = []
            first_point = cnt[0].tolist()
            first_point_adj = [first_point[0] - x_min, first_point[1] - y_min]
            path_commands.append(["M", first_point_adj[0], first_point_adj[1]])
            for pt in cnt[1:]:
                pt_adj = [int(pt[0] - x_min), int(pt[1] - y_min)]
                path_commands.append(["L", pt_adj[0], pt_adj[1]])
            path_commands.append(["Z"])
            obj = {
                "type": "path",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": x_min,
                "top": y_min,
                "width": x_max - x_min,
                "height": y_max - y_min,
                "fill": fill_color,
                "stroke": stroke_color,
                "strokeWidth": stroke_width,
                "strokeDashArray": None,
                "strokeLineCap": "round",
                "strokeDashOffset": 0,
                "strokeLineJoin": "round",
                "strokeUniform": False,
                "strokeMiterLimit": 10,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "flipX": False,
                "flipY": False,
                "opacity": 1,
                "shadow": None,
                "visible": True,
                "backgroundColor": "",
                "fillRule": "nonzero",
                "paintFirst": "fill",
                "globalCompositeOperation": "source-over",
                "skewX": 0,
                "skewY": 0,
                "path": path_commands,
            }
            objects.append(obj)
    return {"version": "4.4.0", "objects": objects}