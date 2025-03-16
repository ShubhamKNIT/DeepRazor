import streamlit as st

draw_mask_page = st.Page("./draw_mask.py", title="Draw Mask")
yolo_page = st.Page("./segment_anything.py", title="YOLO Perdicts Mask")
inpaint_page = st.Page("./inpaint_anything.py", title="Inpaint Anything")

pg = st.navigation({
    "Select a tool": [draw_mask_page, yolo_page, inpaint_page],
})
pg.run()