import cv2
import numpy as np
import streamlit as st
from PIL import Image


# Use the new caching mechanism in Streamlit
@st.cache_resource
def get_predictor_model():
    from model import Model
    model = Model()
    return model


# Set up the header section
header = st.container()
model = get_predictor_model()

with header:
    st.title('Violence Detection App')
    st.text('Using this app, you can classify whether there is a fight on the street, a fire, a car crash, or if everything is okay.')

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Load and process the image
    image = Image.open(uploaded_file).convert('RGB')
    image = np.array(image)

    # Make prediction using the model
    prediction = model.predict(image=image)
    label_text = prediction['label'].title()
    
    # Display the prediction results
    st.write(f'Predicted label is: **{label_text}**')
    st.write('Original Image')
    
    # Display the image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    st.image(image, channels='RGB')  # Display image in RGB format
