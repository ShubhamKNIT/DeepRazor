import streamlit as st

how_to_use_page = st.Page("./how_to_use.py", title="How to Use")
draw_mask_page = st.Page("./draw_mask.py", title="Draw Mask")
yolo_page = st.Page("./segment_anything.py", title="YOLO Perdicts Mask")
inpaint_page = st.Page("./inpaint_anything.py", title="Inpaint Anything")

pg = st.navigation({
    "Select a tool": [how_to_use_page, draw_mask_page, yolo_page, inpaint_page],
})
pg.run()