import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

app = FastAPI()

# Enable CORS (Allows frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and preprocess dataset
df = pd.read_csv("crop_yield_data.csv", header=1)
df.rename(columns={df.columns[0]: "Crops"}, inplace=True)

df_melted = df.melt(id_vars=["Crops"], var_name="Year", value_name="Production")
df_melted["Year"] = df_melted["Year"].astype(int)

df_melted["Production"] = (
    df_melted["Production"].astype(str).str.replace(",", "", regex=True)
    .replace({"..": None, "": None})
    .astype(float)
)
df_melted = df_melted.dropna()

# 🔴 Exclude 2023 and 2024 from training
train_data = df_melted[df_melted["Year"] <= 2022]

@app.get("/predict")
def predict_yield():
    predictions = {}
    errors = {}

    unique_crops = train_data["Crops"].unique()

    for crop in unique_crops:
        crop_df = train_data[train_data["Crops"] == crop][["Year", "Production"]]
        crop_df = crop_df.rename(columns={"Year": "ds", "Production": "y"})
        crop_df["ds"] = pd.to_datetime(crop_df["ds"], format="%Y")
        crop_df["cap"] = crop_df["y"].max() * 1.2

        model = Prophet(growth="logistic", yearly_seasonality=True, seasonality_mode="multiplicative")
        model.fit(crop_df)

        future = model.make_future_dataframe(periods=5, freq="Y")
        future["cap"] = crop_df["cap"].max()
        forecast = model.predict(future)

        future_predictions = forecast[forecast["ds"].dt.year >= 2023][["ds", "yhat"]]
        predictions[crop] = {int(year.year): float(pred) for year, pred in zip(future_predictions["ds"], future_predictions["yhat"])}

        actuals = crop_df["y"]
        predicted = model.predict(crop_df)[["yhat"]]
        mae = mean_absolute_error(actuals, predicted)
        errors[crop] = round(mae, 2)

    return {"predictions": predictions, "model_errors": errors}

@app.get("/predict/onion")
def predict_onion_production():
    onion_data = df_melted[df_melted["Crops"] == "Onion"]
    years = onion_data["Year"].values.reshape(-1, 1)
    production = onion_data["Production"].values
    
    model = LinearRegression()
    model.fit(years[:-1], production[:-1])

    future_years = np.array([2025, 2026, 2027, 2028, 2029]).reshape(-1, 1)
    future_predictions = model.predict(future_years)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(years[:-1], production[:-1], color='blue', label="Actual Data")
    plt.plot(future_years, future_predictions, 'r--', label="Predicted Data")
    plt.xlabel("Year")
    plt.ylabel("Onion Production")
    plt.title("Onion Production Forecast (2025-2029)")
    plt.legend()
    plt.grid()
    plt.show()
    
    return {"onion_production_predictions": {year: round(pred, 2) for year, pred in zip(future_years.flatten(), future_predictions)}}