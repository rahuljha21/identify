import streamlit as st
import keras as keras
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
def prediction1(img_file):
    model=load_model('cotton.h5')
    image=keras.preprocessing.image.load_img(img_file,target_size=(150,150))
    image=img_to_array(image)
    image=image.reshape(1,150,150,3)
    label=model.predict(image)
    label=decode_prediction(label)
    label=label[0][0]
    return label
uploaded=st.file_uploader('Choose the image')
if uploaded is not None:
    image=Image.open(uploaded)
    st.image(image,caption='Image uploaded',use_column_width=True)
    st.write('Classifying')
    label=prediction1(uploaded)
    st.write(label)

