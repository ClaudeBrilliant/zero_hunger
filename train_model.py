# --- train_model.py ---

import os
import pandas as pd
import numpy as np
import requests
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
from dash import Dash, dcc, html

# --- CONFIG ---
API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Nairobi"
DATA_PATH = "data/climate_change_agriculture_dataset.csv"
MODEL_PATH = "model/crop_yield_model.pkl"

# --- FETCH LIVE WEATHER DATA ---
URL = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
response = requests.get(URL)
data = response.json()

if "main" in data:
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    rain = data.get("rain", {}).get("1h", 0)
else:
    print("⚠️ Weather fetch failed, using default values.")
    temperature, humidity, rain = 25, 60, 0

print(f"🌦️ Live Weather Data for {CITY} → Temp: {temperature}°C | Humidity: {humidity}% | Rain: {rain}mm")

# --- LOAD DATASET ---
df = pd.read_csv(DATA_PATH).dropna()

# --- FEATURE SELECTION ---
X = df[['Temperature', 'Precipitation', 'CO2 Levels', 'Soil Health']]
y = df['Crop Yield']

# --- NORMALIZE & SPLIT ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- TRAIN MODEL ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- SAVE MODEL + SCALER ---
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, "model/scaler.pkl")
print("✅ Model and Scaler saved successfully!")

# --- EVALUATION ---
y_pred = model.predict(X_test)
mae, mse, r2 = mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | R²: {r2:.2f}")

# --- DASHBOARD VISUALIZATION ---
app = Dash(__name__)

fig = px.scatter(
    x=y_test, y=y_pred,
    title="Actual vs Predicted Crop Yield",
    labels={'x': 'Actual Crop Yield', 'y': 'Predicted Crop Yield'},
    trendline="ols"
)

app.layout = html.Div([
    html.H1("🌾 Zero Hunger ML Dashboard", style={'textAlign': 'center', 'color': 'green'}),
    html.Div([
        html.P(f"MAE: {mae:.2f}", style={'color': 'blue'}),
        html.P(f"MSE: {mse:.2f}", style={'color': 'orange'}),
        html.P(f"R²: {r2:.2f}", style={'color': 'red'})
    ], style={'textAlign': 'center'}),
    dcc.Graph(figure=fig)
])

if __name__ == "__main__":
    app.run(debug=True)
