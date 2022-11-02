import streamlit as st
import requests
from PIL import Image
import json
import numpy as np
import tensorflow as tf


st.title('Mask and Non Mask User Prediction')


def img_predict(img):
    predimg = np.array(img)[:, :, :3]
    predimg = tf.image.resize(predimg, size=(220, 220))
    predimg = predimg / 255.0

    image_tensor = tf.expand_dims(predimg, 0)
    image_tensor = image_tensor.numpy().tolist()
    

    input_data_json = json.dumps({
    'signature_name':'serving_default',
    'instances':image_tensor})

    URL = "http://backend-mask-detect.herokuapp.com/v1/models/mask_detection2:predict"
    r=requests.post(URL,data= input_data_json)
    res=r.json()

    prediction=np.array(res['predictions'][0]).argmax()


    if prediction==0:
        text_pred = '## This person is using mask'
        st.markdown(text_pred, unsafe_allow_html=True)
    else:
        text_pred = '## This person is not using mask'
        st.markdown(text_pred, unsafe_allow_html=True)
    st.image(img, use_column_width=True)

# Image Upload Option
file = st.file_uploader("Upload an image...", type=["jpg", "png", 'Tiff'])
if file is not None:
    img = Image.open(file)

col1, col2, col3 = st.columns([3, 1, 3])
with col2:
    submitted = st.button('Predict')

    
if submitted:
    img_predict(img)
