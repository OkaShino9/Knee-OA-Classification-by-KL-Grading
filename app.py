import streamlit as st
from fastbook import *
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

def preprocess_image(image):
    if len(image.shape) != 2 or image.dtype != np.uint8:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.uint8(image)
    denoised_image = cv2.fastNlMeansDenoising(image)
    
    auto_contrast = cv2.normalize(denoised_image, None, 0, 255, cv2.NORM_MINMAX)
    
    equ = cv2.equalizeHist(auto_contrast)
    smoothed_image = cv2.GaussianBlur(equ, (5, 5), 0)
    inverted_image = 255 - smoothed_image
    return inverted_image

class PreprocessTransform(Transform):
    def encodes(self, img: PILImage):
        img_np = np.array(img)
        preprocessed_img = preprocess_image(img_np)
        return PILImage.create(preprocessed_img)

learn_inf = load_learner("model.pkl")

def classify_img(data):
    pred, pred_idx, probs = learn_inf.predict(data)
    return pred, probs[pred_idx]

st.title("KNEE OSTEOARTHRITIS CLASSIFICATION BY KELLGREN AND LAWRENCE GRADING SYSTEM🦴🦵")

st.sidebar.image("https://raw.githubusercontent.com/OkaShino9/Knee-OA-Classification-by-KL-Grading/main/logo.png", use_column_width=True)
st.sidebar.write('# UPLOAD A X-RAY KNEE IMAGE TO CLASSIFY! 🧐')

option = st.sidebar.radio('',['Use a test image', 'Use your own image'])

bytes_data = None

st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.markdown('This web application was developed by Chananchai Chanmol and is a part of AI Builders 2024 program, organized by VISTEC, Central Digital and Mahidol University.')
st.sidebar.write("AI Builders page [link](https://www.facebook.com/aibuildersx)")
st.sidebar.write("Medium [link](https://medium.com/@kungkao123456789/knee-osteoarthritis-classification-by-kellgren-and-lawrence-grading-system-with-fast-ai-2738287b0c2e)")
st.sidebar.write("GitHub [link](https://github.com/OkaShino9/Knee-OA-Classification-by-KL-Grading)")
st.sidebar.image("https://raw.githubusercontent.com/OkaShino9/Knee-OA-Classification-by-KL-Grading/main/AIB_Logo.png", use_column_width=True)

if option == 'Use a test image':
    base_url = "https://raw.githubusercontent.com/OkaShino9/Knee-OA-Classification-by-KL-Grading/main/images/"
    image_files = {
        "0": ["9003175L.png", "9003175R.png", "9006407L.png", "9232078R.png"],
        "1": ["9004184L.png", "9035317R.png", "9049999R.png", "9079361R.png"],
        "2": ["9007827L.png", "9007827R.png", "9008884L.png", "9008884R.png"],
        "3": ["9011053L.png", "9012867L.png", "9018389L.png", "9027422L.png"],
        "4": ["9070207R.png", "9115049L.png", "9127180L.png", "9230504R.png"]
    }

    class_folders = list(image_files.keys())
    selected_folder = st.selectbox("Choose a folder (class):", class_folders)
    if selected_folder:
        selected_image = st.selectbox("Choose an image:", image_files[selected_folder])
        if selected_image:
            image_url = f"{base_url}{selected_folder}/{selected_image}"
            response = requests.get(image_url)
            if response.status_code == 200:
                bytes_data = response.content
                st.image(bytes_data, caption="Test image")
            else:
                st.write("Error fetching image from GitHub")

elif option == 'Use your own image':
    uploaded_image = st.file_uploader("Choose your image:")
    if uploaded_image:
        bytes_data = uploaded_image.getvalue()
        st.image(bytes_data, caption="Uploaded image")

if bytes_data:
    classify = st.button("CLASSIFY!")
    if classify:
        image = Image.open(BytesIO(bytes_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        label, confidence = classify_img(image)
        confidence_percentage = confidence * 100
        st.write(f"This is grade {label}! with probability of {confidence_percentage:.2f}%")
