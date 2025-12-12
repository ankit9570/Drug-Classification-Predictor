import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cloudpickle
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Drug Prediction App",
    page_icon="💊",
    layout="centered"
)

# ---------------------------------------------------
# ADVANCED BACKGROUND + CUSTOM UI CSS
# ---------------------------------------------------

background_image_url = "https://images.unsplash.com/photo-1580281658731-036f3ca3a8a9?auto=format&fit=crop&w=1400&q=60"

st.markdown(f"""
    <style>
        /* Background Image */
        .stApp {{
            background: url('{background_image_url}');
            background-size: cover;
            background-position: center;
        }}

        /* Glassmorphism card */
        .glass-card {{
            background: rgba(255, 255, 255, 0.22);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 30px;
            border-radius: 18px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px rgba(0,0,0,0.25);
            margin-top: 25px;
        }}

        /* Predict Button */
        .stButton>button {{
            background: linear-gradient(90deg, #0072ff 0%, #00d4ff 100%);
            color: white;
            border-radius: 10px;
            padding: 10px 25px;
            font-size: 18px;
            border: none;
        }}
        .stButton>button:hover {{
            background: linear-gradient(90deg, #005fcc 0%, #00aacc 100%);
            transform: scale(1.03);
        }}

        /* Prediction Result Box */
        .result-box {{
            background: rgba(0, 0, 0, 0.65);
            color: #fff;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 25px;
            font-size: 24px;
            font-weight: bold;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 40px;
            font-size: 12px;
            color: #ffffff;
            text-shadow: 1px 1px 2px black;
        }}

        .title-text {{
            color: white;
            font-size: 38px;
            font-weight: 900;
            text-shadow: 2px 2px 6px black;
        }}

    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.markdown("<h1 class='title-text'>💊 Drug Classification Predictor</h1>", unsafe_allow_html=True)

# ---------------------------------------------------
# MODEL LOADING FUNCTION
# ---------------------------------------------------
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except:
        with open(model_path, "rb") as f:
            return cloudpickle.load(f)

# Auto-detect any .pkl file
model_file = None
for f in os.listdir("."):
    if f.endswith(".pkl") or f.endswith(".joblib"):
        model_file = f
        break

loaded_model = load_model(model_file) if model_file else None

# Sidebar info
if loaded_model:
    st.sidebar.success(f"Model Loaded: {model_file}")
else:
    st.sidebar.error("❌ No model found. Place your .pkl file in this folder.")

# ---------------------------------------------------
# GLASS CARD UI
# ---------------------------------------------------
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    st.subheader("🔎 Enter Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=35)
        sex = st.selectbox("Sex", ["M", "F"])

    with col2:
        bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
        chol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
        na_to_k = st.number_input("Na to K Ratio", value=15.3, step=0.1)

    predict_btn = st.button("🔮 Predict Drug")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# PREDICTION RESULT
# ---------------------------------------------------
if predict_btn:

    if loaded_model is None:
        st.error("Model not loaded! Please place .pkl model in this folder.")
    else:
        input_df = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "BP": bp,
            "Cholesterol": chol,
            "Na_to_K": na_to_k
        }])

        prediction = loaded_model.predict(input_df)[0]

        st.markdown(f"""
            <div class='result-box'>
                🚀 Predicted Drug: <span style='color:#00e6ff'>{prediction}</span>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("""
    <div class='footer'>
        © 2025 • Made with ❤️ using Streamlit • Advanced UI Edition
    </div>
""", unsafe_allow_html=True)
