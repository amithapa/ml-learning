import requests
import base64
import cv2
import os
import numpy as np

import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates
from streamlit_dimensions import st_dimensions

api_endpoint = "https://amitthapa.ap-south-1.modelbit.com/v1/remove_background/latest"

# set layout
st.set_page_config(layout="wide")

col_01, col_02 = st.columns(2)

# file uploader
file = col_02.file_uploader("", type=["jpeg", "jpg", "png"])

# read the image
if file is not None:
    image = Image.open(file).convert("RGB")

    # screen_dim = st_dimensions(key="main")
    # print(screen_dim)

    image = image.resize((600, int(image.height * 600 / image.width)))


    # create buttons
    col_1, col_2 = col_02.columns(2)

    placeholder_0 = col_02.empty()
    with placeholder_0:
        value = im_coordinates(image)
        if value is not None:
            print(value)

    if col_1.button("Original", use_container_width=True):
        placeholder_0.empty()
        placeholder_1 = col_02.empty()
        with placeholder_1:
            col_02.image(image, use_column_width=True)
    
    if col_1.button("Remove background", type="primary", use_container_width=True):
        placeholder_0.empty()
        placeholder_2 = col_02.empty()

        file_name = f"{file.name}_{value['x']}_{value['y']}.png"

        if os.path.exists(file_name):
            result_image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        else:
            _, image_bytes = cv2.imencode('.png', np.asarray(image))

            image_bytes = image_bytes.tobytes()

            image_bytes_encoded_base64 = base64.b64encode(image_bytes).decode('utf-8')

            api_data = {
                "data": [image_bytes_encoded_base64, value["x"], value["y"]]
            }

            response = requests.post(api_endpoint, json=api_data)

            result_image = response.json()["data"]

            result_image_bytes = base64.b64decode(result_image)

            result_image = cv2.imdecode(np.frombuffer(result_image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            cv2.imwrite(file_name, result_image)

        with placeholder_2:
            col_02.image(result_image, use_column_width=True)
    

    # visualize image


    # click on image, get coordinates

    # call api