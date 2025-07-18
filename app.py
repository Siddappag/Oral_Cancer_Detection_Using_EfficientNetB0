import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="Oral Cancer Detector", page_icon="🩺", layout="centered")

# Custom Background Image & Font Styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://plus.unsplash.com/premium_photo-1699387204159-184c8e0ac55e?q=80&w=1378&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        opacity: 10px;
        
    }
    h1, h2, h3 {
        font-family: 'Trebuchet MS', sans-serif;
        color: #2E8B57;
    }
    .css-1aumxhk {
        font-family: 'Verdana';
    }
    </style>
""", unsafe_allow_html=True)




# Header
st.markdown("""
    <h1 style='text-align: center;'>🩺 Oral Cancer Detection</h1>
    <p style='text-align: center; font-size:18px;'>Upload an oral cavity image to detect whether it's <b>cancerous or non-cancerous</b>.</p>
    <hr>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_eff_model():
    model = load_model('oral_cancer_efficientnet_model.h5', compile=False)
    return model

model = load_eff_model()

# Sidebar: Example Images
with st.sidebar:
    st.markdown("### 📚 Example Images")
    st.image("https://as2.ftcdn.net/v2/jpg/05/52/92/39/1000_F_552923900_mfShHvIrJWygRDLGfP6pYp4gO3nM7C31.jpg", caption="Cancer Case", use_container_width=True)
    st.image("https://as2.ftcdn.net/v2/jpg/11/36/15/77/1000_F_1136157716_VN3vxCHfS4BSMAcUaiNBNF4fZSp5hwzi.jpg", caption="Non-Cancer Case", use_container_width=True)

# File uploader
uploaded_file = st.file_uploader("📷 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# Inference
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Show preview
    st.markdown("### 🖼️ Preview of Uploaded Image")
    st.image(image, use_container_width=True, caption="Uploaded Image")

    

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_preprocessed = preprocess_input(img_array)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)

    # Predict
    prediction = model.predict(img_expanded)[0][0]
    prob = float(prediction)

    # Interpret (assumes 0 = Cancer, 1 = Non-Cancer)
    is_non_cancer = prob > 0.5
    pred_label = "🟢 Non-Cancer" if is_non_cancer else "🔴 Cancer"
    pred_prob = prob if is_non_cancer else 1 - prob

    # Display result
    st.markdown("### 🧠 Prediction Result")
    st.markdown(f"""
        <div style='padding: 20px; background-color: #1B2631; border-radius: 10px; text-align: center;'>
            <h2>{pred_label}</h2>
            <p style='font-size: 18px;'>Confidence: <b>{pred_prob:.2%}</b></p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Confidence Level")
    st.progress(pred_prob)

    st.info("📌 Note: This is an AI-based prediction. Always consult a medical professional for a confirmed diagnosis.")

# About & Disclaimer sections
with st.expander("📌 About the Model"):
    st.markdown("""
    - Model: EfficientNetB0
    - Accuracy: ~98% on test set
    - Trained on oral cancer image dataset (~1,081 samples)
    - Includes fine-tuning, focal loss, and image augmentation
    """)

with st.expander("⚠️ Disclaimer"):
    st.markdown("""
    This tool is for **educational and demonstration purposes only**.
    It is **not a substitute for medical diagnosis**.
    Always consult a certified medical professional.
    """)

