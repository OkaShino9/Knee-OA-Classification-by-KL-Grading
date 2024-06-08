import streamlit as st
from fastbook import *
import glob
from random import shuffle
import urllib.request
from PIL import Image
import os
from git import Repo
import shutil

class PreprocessTransform(Transform):
    def encodes(self, img: PILImage):
        img_np = np.array(img)
        preprocessed_img = preprocess_image(img_np)
        return PILImage.create(preprocessed_img)

learn = load_learner('pre_resnet34.pkl')
model = learn.model

st.title("Knee Osteoarthritis Classification by KL Grading")

def predict(img, learn):
    # ย่อขนาดรูป
    pimg = img.resize([224,224])
    
    # ทำนายจากโมเดลที่ให้
    pred, pred_idx, pred_prob = learn.predict(pimg)
        
    pred = pred.split('_')[1:]
    
    if pred[-1] == 'Dog':
        pred = ' '.join(pred[:len(pred)-1])
    else:
        pred = ' '.join(pred)

    # โชว์ผลการทำนาย
    st.success(f'This is "Grade {pred}" with the probability of {pred_prob[pred_idx]*100:.02f}%')
    
    # โชว์รูปที่ถูกทำนาย
    st.image(img, use_column_width=True)
    
    st.balloons()

st.sidebar.write('# Upload a x-ray knee image to classify!')

# radio button สำหรับเลือกว่าจะทำนายรูปจาก set set หรือ upload รูปเอง
option = st.sidebar.radio('', ['Use a validation image', 'Use your own image', 'Take a photo'])
# โหลดรูปจาก set set แล้ว shuffle
valid_images = glob.glob('images')
valid_images.sort()
for i in range(len(valid_images)):
    k = str(valid_images[i])
    k =k.replace('images','')
    valid_images[i] = k

if option == 'Use a validation image':
    st.sidebar.write('### Select a validation image')
    fname = st.sidebar.selectbox('', valid_images)
    
    # เปิดรูป
    img = Image.open(f'{fname}')

    st.sidebar.image(img, f'Is this the image you want to predict?', use_column_width=True)

    if st.sidebar.button("Predict Now!"):
        # เรียก function ทำนาย
        predict(img, model)
        
elif option == 'Use your own image':
    st.sidebar.write('### Select an image to upload')
    fname = st.sidebar.file_uploader('',
                                     type=['jpg', 'jpeg', 'png'],
                                     accept_multiple_files=False)
    if fname is None:
        st.sidebar.write("Please select an image...")
    else:
        # เปิดรูป
        img = Image.open(fname)
        # เปลี่ยน format ภาพ
        img = img.convert('RGB')
        img.save('fname.jpg')
        
        img = Image.open('fname.jpg')
        
        st.sidebar.image(img, f'Is this the image you want to predict?', use_column_width=True)

        if st.sidebar.button("Predict Now!"):
            # เรียก function ทำนาย
            predict(img, model)
else:
        fname = st.sidebar.camera_input('Take a photo of a knee x-ray image')
        if fname is None:
            st.sidebar.write("Please take a photo...")
        else:
            # เปิดรูป
            img = Image.open(fname)
            # เปลี่ยน format ภาพ
            img = img.convert('RGB')
            img.save('fname.jpg')

            img = Image.open('fname.jpg')

            st.sidebar.image(img, 'Is this the image you want to predict?', use_column_width=True)

            if st.sidebar.button("Predict Now!"):
                # เรียก function ทำนาย
                predict(img, model)
