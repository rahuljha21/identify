import streamlit as st
import keras as keras
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions


st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
model=VGG16()
uploaded=st.file_uploader('Choose the image')
if uploaded is not None:
    image=Image.open(uploaded)
    st.image(image,caption='Image uploaded',use_column_width=True)
    st.write('Classifying')
    image=load_img(uploaded,target_size=(224,224))
    image=img_to_array(image)
    image=image.reshape((1,224,224,3))
    image=preprocess_input(image)
    label=model.predict(image)
    label=decode_prediction(label)
    label=label[0][0]
    st.write(label)
