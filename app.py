import streamlit as st
from fastbook import *
import glob
from random import shuffle
import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    # Ensure the input is a single-channel 8-bit image
    if len(image.shape) != 2 or image.dtype != np.uint8:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.uint8(image)
    # Denoise the image
    denoised_image = cv2.fastNlMeansDenoising(image)
    # Normalize the image to improve contrast
    auto_contrast = cv2.normalize(denoised_image, None, 0, 255, cv2.NORM_MINMAX)
    # Apply CLAHE
    equ = cv2.equalizeHist(auto_contrast)
    smoothed_image = cv2.GaussianBlur(equ, (5, 5), 0)
    return smoothed_image

class PreprocessTransform(Transform):
    def encodes(self, img: PILImage):
        img_np = np.array(img)
        preprocessed_img = preprocess_image(img_np)
        return PILImage.create(preprocessed_img)

learn = load_learner('model.pkl')
model = learn.model

st.title("Knee Osteoarthritis Classification by KL Grading")

def predict(img, learn):
    # Resize image
    pimg = img.resize([224,224])
    # Predict using the model
    pred, pred_idx, pred_prob = learn.predict(pimg)
    pred = pred.split('_')[1:]
    pred = ' '.join(pred)
    # Display prediction
    st.success(f'This is "Grade {pred}" with the probability of {pred_prob[pred_idx]*100:.02f}%')
    # Show the predicted image
    st.image(img, use_column_width=True)
    st.balloons()

st.sidebar.write('# Upload a x-ray knee image to classify!')

# Radio button to choose the image source
option = st.sidebar.radio('', ['Use a validation image', 'Use your own image', 'Take a photo'])
# Load validation images and shuffle
valid_images = glob.glob('OkaShino9/Knee-OA-Classification-by-KL-Grading/images/*')
valid_images.sort()
for i in range(len(valid_images)):
    valid_images[i] = valid_images[i].replace('OkaShino9/Knee-OA-Classification-by-KL-Grading/images/', '')

if option == 'Use a validation image':
    st.sidebar.write('### Select a validation image')
    fname = st.sidebar.selectbox('', valid_images)
    img_path = f'OkaShino9/Knee-OA-Classification-by-KL-Grading/images/{fname}'
    img = Image.open(img_path)
    st.sidebar.image(img, 'Is this the image you want to predict?', use_column_width=True)
    if st.sidebar.button("Predict Now!"):
        predict(img, model)

else option == 'Use your own image':
    st.sidebar.write('### Select an image to upload')
    fname = st.sidebar.file_uploader('', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if fname is None:
        st.sidebar.write("Please select an image...")
    else:
        img = Image.open(fname)
        img = img.convert('RGB')
        st.sidebar.image(img, 'Is this the image you want to predict?', use_column_width=True)
        if st.sidebar.button("Predict Now!"):
            predict(img, model)

