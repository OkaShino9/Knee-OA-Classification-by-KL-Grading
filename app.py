import streamlit as st
from fastbook import *
import cv2
import numpy as np

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

## LOAD MODEl
learn_inf = load_learner("model.pkl")
## CLASSIFIER
def classify_img(data):
    pred, pred_idx, probs = learn_inf.predict(data)
    return pred, probs[pred_idx]

st.title("Knee Osteoarthritis Classification by KL Grading 🦴🦵")

bytes_data = None

if option == 'Use a test image':
    # List of test image URLs from your GitHub repository
    test_image_urls = [
        "https://github.com/OkaShino9/Knee-OA-Classification-by-KL-Grading/tree/main/images/0",
        "https://github.com/OkaShino9/Knee-OA-Classification-by-KL-Grading/tree/main/images/1",
        "https://github.com/OkaShino9/Knee-OA-Classification-by-KL-Grading/tree/main/images/2",
        "https://github.com/OkaShino9/Knee-OA-Classification-by-KL-Grading/tree/main/images/3",
        "https://github.com/OkaShino9/Knee-OA-Classification-by-KL-Grading/tree/main/images/4",
    ]
    
    test_image = st.selectbox("Choose a test image:", test_image_urls)
    response = requests.get(test_image)
    bytes_data = response.content
    st.image(bytes_data, caption="Test image")

elif option == 'Use your own image':
    uploaded_image = st.file_uploader("Choose your image:")
    if uploaded_image:
        bytes_data = uploaded_image.getvalue()
        st.image(bytes_data, caption="Uploaded image")

if bytes_data:
    classify = st.button("CLASSIFY!")
    if classify:
        image = Image.open(BytesIO(bytes_data))
        label, confidence = classify_img(image)
        st.write(f"This is grade {label}! ({confidence:.04f})")


