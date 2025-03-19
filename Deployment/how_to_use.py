import streamlit as st

page_name = "how_to_use"
st.title("How to Use This App")

st.subheader("Watch Demo Video")
st.video("https://www.youtube.com/watch?v=IRND7na0cWA")

st.subheader("Instructions")
st.write(
    "1. **Draw Mask**: Upload an image and draw a mask on it. The mask will be used to segment the image using YOLO.\n",
    "2. **YOLO Predicts Mask**: Upload an image and predict the mask using YOLO. Select the classes to predict.\n",
    "3. **Inpaint Anything**: Upload an image and a mask to inpaint the masked region using a deep learning model.",
)