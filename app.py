# Streamlit app for brain tumor detection

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model('brainCNN.h5')

# Class labels (update this based on your actual dataset classes)
class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']

# Streamlit UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan and get the prediction result.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image to match model input (224x224)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show prediction
    st.subheader("Prediction:")
    st.success(f"{predicted_class} ({confidence*100:.2f}% confidence)")
