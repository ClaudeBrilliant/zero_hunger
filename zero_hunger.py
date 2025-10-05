# --- Import libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load dataset ---
file_path = "/home/clyde/Downloads/climate_change_agriculture_dataset.csv"
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print(df.head())

# --- Data Cleaning ---
df = df.dropna()
print("\nMissing values after cleaning:\n", df.isnull().sum())

# --- Feature Selection ---
X = df[['Temperature', 'Precipitation', 'CO2 Levels', 'Soil Health']]
y = df['Crop Yield']

# --- Normalize Data (Optional but simple) ---
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test)

print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# --- Visualization: Actual vs Predicted ---
plt.figure(figsize=(6,5))
sns.scatterplot(x=y_test, y=y_pred, color="teal")
plt.xlabel("Actual Crop Yield")
plt.ylabel("Predicted Crop Yield")
plt.title("Actual vs Predicted Crop Yield")
plt.show()

# --- Feature Correlation Heatmap ---
df_encoded = pd.get_dummies(df, drop_first=True)
sns.heatmap(df_encoded.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap (Encoded Data)")
plt.show()

