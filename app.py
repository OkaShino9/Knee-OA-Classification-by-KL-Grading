import streamlit as st
from fastbook import *
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

def preprocess_image(image):
    # Ensure the input is a single-channel 8-bit image
    if len(image.shape) != 2 or image.dtype != np.uint8:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.uint8(image)
    # Denoise the image
    denoised_image = cv2.fastNlMeansDenoising(image)
    
    # Normalize the image to improve contrast
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

## LOAD MODEL
learn_inf = load_learner("model.pkl")

## CLASSIFIER
def classify_img(data):
    pred, pred_idx, probs = learn_inf.predict(data)
    return pred, probs[pred_idx]

st.title("Knee Osteoarthritis Classification by KL Grading 🦴🦵")

# Sidebar for selecting image source
st.sidebar.write('# Upload a x-ray knee image to classify!')

# Radio button to choose the image source
option = st.sidebar.radio('', ['Use a test image', 'Use your own image'])

bytes_data = None

if option == 'Use a test image':
    base_url = "https://raw.githubusercontent.com/OkaShino9/Knee-OA-Classification-by-KL-Grading/main/images/"
    class_folders = ["0", "1", "2", "3", "4"]

    selected_folder = st.selectbox("Choose a folder (class):", class_folders)
    if selected_folder:
        folder_url = f"{base_url}{selected_folder}/"
        response = requests.get(f"https://api.github.com/repos/OkaShino9/Knee-OA-Classification-by-KL-Grading/contents/images/{selected_folder}")
        if response.status_code == 200:
            files = response.json()
            image_files = [file["name"] for file in files if file["name"].lower().endswith((".png", ".jpg", ".jpeg"))]
            selected_image = st.selectbox("Choose an image:", image_files)
            if selected_image:
                image_url = f"{folder_url}{selected_image}"
                response = requests.get(image_url)
                bytes_data = response.content
                st.image(bytes_data, caption="Test image")
        else:
            st.write("Error fetching image list from GitHub")

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
        st.write(f"This is grade {label}! ({confidence:.04f})")
