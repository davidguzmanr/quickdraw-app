import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# Specify canvas parameters in application
stroke_width = st.sidebar.slider(label='Stroke width: ', min_value=1, max_value=25, value=3)
drawing_mode = st.sidebar.selectbox(
    label='Drawing tool:', options=('freedraw', 'line', 'rect', 'circle', 'transform')
)
realtime_update = st.sidebar.checkbox('Update in realtime', True)

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color='black',
    update_streamlit=realtime_update,
    height=400,
    width=400,
    drawing_mode=drawing_mode,
    key='canvas',
)

if canvas_result.image_data is not None:
    image = canvas_result.image_data

    st.image(image)
