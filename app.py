# --- app.py ---

import os
import streamlit as st
import requests
import numpy as np
import joblib

st.set_page_config(page_title="Zero Hunger App", page_icon="🌾")

st.title("🌾 Zero Hunger: Crop Yield Predictor")
st.markdown("Predict crop yield using live weather data 🌦️")

# --- Load Model and Scaler ---
model = joblib.load("model/crop_yield_model.pkl")
scaler = joblib.load("model/scaler.pkl")

city = st.text_input("Enter City Name", "Kakamega")

if st.button("Predict Crop Yield"):
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    URL = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(URL)
    data = response.json()

    if "main" not in data:
        st.error("❌ Could not fetch weather data. Check city name or API key.")
    else:
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rain = data.get("rain", {}).get("1h", 0)

        st.write(f"🌤️ Temperature: {temp}°C")
        st.write(f"💧 Humidity: {humidity}%")
        st.write(f"🌧️ Rainfall (1h): {rain} mm")

        # Prepare input (with dummy values for CO2, Soil)
        X_live = np.array([[temp, rain, 0, 0]])
        X_scaled = scaler.transform(X_live)

        prediction = model.predict(X_scaled)[0]
        st.success(f"🌾 Predicted Crop Yield: **{prediction:.2f} units**")
