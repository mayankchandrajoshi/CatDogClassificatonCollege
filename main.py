import tensorflow as tf
import numpy as np
import streamlit as st
import cv2
from PIL import Image,ImageOps


# loading the saved model
loaded_model = tf.saved_model.load('./trained_model')

# creating a function for Classification

def classifier(input_image):

    input_image_resize =  ImageOps.fit(input_image, (244,244), Image.ANTIALIAS);

    input_image_resize = np.asarray(input_image_resize)

    input_image_scaled = input_image_resize/255

    input_image_tensor = tf.convert_to_tensor(input_image_scaled, dtype=tf.float32)
    input_image_tensor = tf.expand_dims(input_image_tensor, axis=0)
    input_image_tensor = tf.image.resize(input_image_tensor, [224, 224])

    input_prediction = loaded_model(input_image_tensor, True, None)

    input_pred_label = np.argmax(input_prediction)

    if input_pred_label == 0:
        return "It is a Cat"
    else:
        return "It is a Dog"


def main():
    # giving a title
    st.title('Cat and Dog classifier')

    # getting the input data from the user
    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
    
    image = '';

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)

    # code for Prediction
    prediction = ''
    
    # creating a button for Prediction
    if image and st.button('Classify') :
        prediction = classifier(image)
        
    st.success(prediction)

if __name__ == '__main__':
    main()
