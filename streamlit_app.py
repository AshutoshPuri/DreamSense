import streamlit as st
import pandas as pd
import joblib
import base64

# ===============================
# 🌠 Background Setup
# ===============================
def set_background():
    file_path = "app/assets/Dream-2.png"
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: top center;
            background-attachment: fixed;
            margin: 0;
            padding: 0;
            font-family: 'Playfair Display', serif;
        }}

        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,500&display=swap');

        #MainMenu, header, footer {{
            visibility: hidden;
        }}

        .center-box {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90%;
            max-width: 600px;
            text-align: center;
        }}

        .dream-prompt {{
            font-size: 1.8em;
            color: #f3f3f3;
            font-style: italic;
            margin-bottom: 1em;
            text-shadow: 1px 1px 3px #000000;
        }}

        .stTextArea textarea {{
            background-color: rgba(255, 255, 255, 0.7) !important;
            color: #111111 !important;
            font-size: 1.1em;
            border-radius: 10px !important;
            padding: 10px 14px;
            border: 1px solid rgba(0, 0, 0, 0.3) !important;
            resize: vertical !important;
            min-height: 100px !important;
            max-height: 400px !important;
        }}

        .stButton>button {{
            background-color: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            font-weight: 500;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 8px;
            padding: 0.6em 1.2em;
            margin-top: 1em;
        }}

        .stButton>button:hover {{
            background-color: rgba(255,255,255,0.2);
            color: #fff;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 🖼️ Set the background image
set_background()

# ===============================
# 🧠 Load ML model and vectorizer
# ===============================
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("models/dream_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ===============================
# ✨ Dream Input Box (Centered)
# ===============================
st.markdown('<div class="center-box">', unsafe_allow_html=True)
st.markdown('<div class="dream-prompt">Enter the dream, as if whispering to the moon...</div>', unsafe_allow_html=True)

dream_input = st.text_area("", height=80, key="dream_input")

if st.button("Interpret Emotion ✨"):
    if dream_input.strip():
        vec = vectorizer.transform([dream_input])
        proba = model.predict_proba(vec)[0]
        pred = 1 if proba[1] > 0.75 else 0  # Only positive if >65% sure

        label_map = {0: "negative", 1: "positive"}
        pred_label = label_map.get(pred, "unknown")

        emoji_map = {
            "positive": "🌈 A calm and comforting emotion flows through this dream.",
            "negative": "🌩️ A storm of emotion brews within this dream."
        }

        st.markdown(
            f"""
            <div style="
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                font-size: 1.5em;
                font-weight: 400;
                color: #ffe8c4;
                text-shadow: 1px 1px 2px #000;
                margin-top: 20px;
                border: 1px solid rgba(255,255,255,0.3);
                box-shadow: 0 0 12px rgba(255, 255, 255, 0.2);
                font-style: italic;
            ">
                {emoji_map.get(pred_label, "💭 A mystery unfolds in this dream...")}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("The dream cannot be empty. Tell us what you saw.")
        
st.markdown('</div>', unsafe_allow_html=True)
