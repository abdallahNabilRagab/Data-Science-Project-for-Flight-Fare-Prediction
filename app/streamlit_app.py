import sys
import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from PIL import Image
import base64

# =========================================================
# Project Path Setup
# =========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# =========================================================
# Imports
# =========================================================
from src.inference import predict   # ✅ USE INFERENCE LAYER

# =========================================================
# Paths
# =========================================================
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Data_Train.xlsx")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
COVER_IMAGE_PATH = os.path.join(ASSETS_DIR, "end to end data science project.png")

# =========================================================
# Streamlit Configuration
# =========================================================
st.set_page_config(
    page_title="✈️ Flight Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Sidebar – Dark/Light Theme Toggle
# =========================================================
st.sidebar.header("🌙 Theme & Hero Styling")

theme_choice = st.sidebar.radio("Choose Theme", ["Light", "Dark"])

if theme_choice == "Dark":
    bg_color = "#111111"
    text_color = "#FFFFFF"
else:
    bg_color = "#FFFFFF"
    text_color = "#111111"

# =========================================================
# Sidebar – Hero Image Controls
# =========================================================
overlay_opacity = st.sidebar.slider(
    "Overlay Darkness",
    min_value=0.0,
    max_value=0.9,
    value=0.45,
    step=0.05
)

border_thickness = st.sidebar.slider(
    "Border Thickness (px)",
    min_value=0,
    max_value=20,
    value=6
)

border_color = st.sidebar.color_picker(
    "Border Color",
    value="#1f77b4"
)

# =========================================================
# Apply Theme via CSS
# =========================================================
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
    transition: all 0.5s ease;
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Main Title (Sticky Header + Above Image)
# =========================================================
st.markdown(
    f"""
    <style>
    .sticky-header {{
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: {bg_color};
        color: {text_color};
        padding: 15px 0;
        font-weight: 800;
        font-size: 2.8rem;
        text-align: center;
        border-bottom: 2px solid {border_color};
        transition: all 0.5s ease;
    }}
    .sticky-header.scrolled {{
        background-color: {border_color};
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}
    </style>

    <div class="sticky-header" id="main-header">
        ✈️ Flight Price Prediction System
    </div>

    <script>
    window.addEventListener('scroll', function() {{
        const header = document.getElementById('main-header');
        if(window.scrollY > 50){{
            header.classList.add('scrolled');
        }} else {{
            header.classList.remove('scrolled');
        }}
    }});
    </script>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color: #6c757d; margin-bottom:30px;'>End-to-End Data Science Project for Flight Fare Prediction</p>",
    unsafe_allow_html=True
)

# =========================================================
# Helper – Image to Base64
# =========================================================
def image_to_base64(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

# =========================================================
# Hero Section (Image + Overlay ONLY)
# =========================================================
if os.path.exists(COVER_IMAGE_PATH):
    img_base64 = image_to_base64(COVER_IMAGE_PATH)

    st.markdown(
        f"""
        <style>
        .hero-container {{
            position: relative;
            width: 100%;
            border-radius: 18px;
            overflow: hidden;
            border: {border_thickness}px solid {border_color};
            margin-bottom: 40px;
            transition: all 0.5s ease;
        }}

        .hero-image {{
            width: 100%;
            display: block;
            transition: transform 1s ease;
        }}

        .hero-container:hover .hero-image {{
            transform: scale(1.02);
        }}

        .hero-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, {overlay_opacity});
            transition: background-color 0.5s ease;
        }}

        .hero-container:hover .hero-overlay {{
            background-color: rgba(0,0,0,{min(overlay_opacity + 0.1, 0.9)});
        }}
        </style>

        <div class="hero-container">
            <img src="data:image/png;base64,{img_base64}" class="hero-image"/>
            <div class="hero-overlay"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Cover image not found in assets folder.")

st.divider()

# =========================================================
# Load Artifacts
# =========================================================
@st.cache_resource
def load_support_artifacts():
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return preprocessor, feature_columns

preprocessor, feature_columns = load_support_artifacts()

# =========================================================
# Load Raw Data
# =========================================================
@st.cache_data
def load_raw_data(path):
    df = pd.read_excel(path)
    for col in ["Route", "Duration", "Total_Stops", "Additional_Info"]:
        if col not in df.columns:
            df[col] = ""
    return df

df_raw = load_raw_data(RAW_DATA_PATH)

# =========================================================
# Sidebar – User Inputs
# =========================================================
st.sidebar.header("✈️ Flight Scenario Input")

airline = st.sidebar.selectbox("Airline", sorted(df_raw["Airline"].unique()))
source = st.sidebar.selectbox("Source Airport", sorted(df_raw["Source"].unique()))
destination = st.sidebar.selectbox("Destination Airport", sorted(df_raw["Destination"].unique()))

if source == destination:
    st.sidebar.error("Source and Destination cannot be the same.")
    st.stop()

total_stops = st.sidebar.selectbox(
    "Total Stops",
    ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]
)

date_of_journey = st.sidebar.date_input("Date of Journey")
dep_time_obj = st.sidebar.time_input("Departure Time", datetime.strptime("10:00", "%H:%M").time())
arrival_time_obj = st.sidebar.time_input("Arrival Time", datetime.strptime("12:00", "%H:%M").time())

dep_time = dep_time_obj.strftime("%H:%M")
arrival_time = arrival_time_obj.strftime("%H:%M")

additional_info = st.sidebar.text_input("Additional Info", "")

# =========================================================
# Duration Calculation
# =========================================================
def calculate_duration(dep, arr):
    dep_h, dep_m = map(int, dep.split(":"))
    arr_h, arr_m = map(int, arr.split(":"))
    mins = (arr_h * 60 + arr_m) - (dep_h * 60 + dep_m)
    if mins <= 0:
        mins += 24 * 60
    return f"{mins // 60}h {mins % 60}m"

duration = calculate_duration(dep_time, arrival_time)

# =========================================================
# Input DataFrame
# =========================================================
input_flight = pd.DataFrame([{
    "Airline": airline,
    "Source": source,
    "Destination": destination,
    "Total_Stops": total_stops,
    "Date_of_Journey": date_of_journey.strftime("%d/%m/%Y"),
    "Dep_Time": dep_time,
    "Arrival_Time": arrival_time,
    "Duration": duration,
    "Additional_Info": additional_info,
    "Route": f"{source} → {destination}"
}])

# =========================================================
# Compatibility Score
# =========================================================
X_input = preprocessor.transform(input_flight).reindex(columns=feature_columns, fill_value=0)
X_train = preprocessor.transform(df_raw).reindex(columns=feature_columns, fill_value=0)

z_scores = ((X_input - X_train.mean()).abs()) / X_train.std().replace(0, 1)
compatibility_score = (z_scores <= 3).mean(axis=1).iloc[0]

# =========================================================
# Prediction
# =========================================================
if st.button("🔮 Predict Flight Price"):
    predicted_price = predict(input_flight, model_dir=MODEL_DIR)[0]

    col1, col2 = st.columns(2)
    col1.metric("Predicted Price", f"{predicted_price:.2f}")
    col2.metric("Data Compatibility Score", f"{compatibility_score:.2%}")

    if compatibility_score >= 0.85:
        st.success("✅ High confidence prediction.")
    elif compatibility_score >= 0.65:
        st.warning("⚠️ Moderate confidence prediction.")
    else:
        st.error("❌ Low confidence prediction.")

    st.subheader("🧾 Input Flight Details")
    st.table(input_flight.T)

# =========================================================
# Transparency
# =========================================================
with st.expander("🔍 Model Transparency"):
    st.write("Training samples:", df_raw.shape[0])
    st.write("Features used:", len(feature_columns))
    st.write("Inference Layer:", "src.inference.predict()")
