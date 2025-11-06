import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# help of chatgpt for design structure 
# App Configuration

st.set_page_config(
    page_title="AI-Powered Image Classification",
    page_icon="❤️",
    layout="wide"
)


# use css
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
    }
    h1 {
        color: #2C3E50;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        font-size: 16px;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #45a049;
        color: #fff;
    }
    .result {
        text-align: center;
        font-size: 22px;
        color: #2E86C1;
        font-weight: 600;
        margin-top: 15px;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# Load Model
try:
    model = load_model("../model/best_model.h5")
    class_names = ['birds', 'cats', 'dogs', 'fruits', 'tiger_lion']
except Exception as e:
    st.error(f" Model could not be loaded. Error: {e}")
    st.stop()

st.title("🤖 AI-Powered Image Classification")
st.markdown("""
    <div style="text-align:center; font-size:18px;">
        Upload an image to classify it into one of five categories:<br>
        <b>Birds, Cats, Dogs, Fruits, or Tiger/Lion</b>
    </div>
""", unsafe_allow_html=True)

st.write("")

# File Upload

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])

    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption=" Uploaded Image", use_container_width=True)

    with col2:
        # Preprocess image
        img_resized = img.resize((150, 150))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        prediction = model.predict(img_array)
        result = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display result
        st.markdown(f"<div class='result'> Predicted Class: <b>{result.upper()}</b></div>", unsafe_allow_html=True)
        st.progress(int(confidence))
        st.info(f"**Confidence:** {confidence:.2f}%")

        if confidence < 70:
            st.warning("Confidence is a bit low — consider using a clearer image.")


st.markdown("""
    <hr>
    <div style='text-align:center; font-size:14px; color:gray;'>
        Developed by <b>AI Developer - Flikt Technology</b> | Streamlit App
    </div>
""", unsafe_allow_html=True)
