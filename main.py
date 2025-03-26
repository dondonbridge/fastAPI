from fastapi import FastAPI, HTTPException
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dataset mapping
csv_mapping = {
    "aurora": "datasets/aurora-production.csv",
    "bataan": "datasets/bataan-production.csv",
    "bulacan": "datasets/bulacan-production.csv",
    "nueva-ecija": "datasets/nueva-ecija-production.csv",
    "pampanga": "datasets/pampanga-production.csv",
    "tarlac": "datasets/tarlac-production.csv",
    "zambales": "datasets/zambales-production.csv",
}

# Function to load and preprocess CSV (for wide-format dataset)
def load_and_preprocess(csv_file):
    try:
        df = pd.read_csv(csv_file)

        # Remove commas from all numerical values
        df.replace(",", "", regex=True, inplace=True)

        # Convert wide format to long format
        df_melted = df.melt(id_vars=["Crops"], var_name="Year", value_name="Production")

        # Ensure "Year" is an integer
        df_melted["Year"] = pd.to_numeric(df_melted["Year"], errors="coerce").astype("Int64")

        # Ensure "Production" is a float
        df_melted["Production"] = pd.to_numeric(df_melted["Production"], errors="coerce")

        # Drop rows with missing values
        df_melted.dropna(inplace=True)

        return df_melted
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading {csv_file}: {str(e)}")

def predict_crop_yield(csv_file):
    train_data = load_and_preprocess(csv_file)
    predictions = {}
    errors = {}
    unique_crops = train_data["Crops"].unique()
    
    for crop in unique_crops:
        crop_df = train_data[train_data["Crops"] == crop][["Year", "Production"]]
        
        if len(crop_df) < 2:
            continue  # Skip if there's not enough data to train a model

        X = crop_df[["Year"]].values.reshape(-1, 1)
        y = crop_df["Production"].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_years = np.array(range(2023, 2028)).reshape(-1, 1)  # Adjust future years
        future_predictions = model.predict(future_years)
        
        predictions[crop] = {int(year): float(pred) for year, pred in zip(future_years.flatten(), future_predictions)}
        
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        errors[crop] = float(mae)  # Convert to native Python float
    
    return {"predictions": predictions, "model_errors": errors}

@app.get("/predict")
def root_predict():
    raise HTTPException(status_code=400, detail="Specify a dataset name, e.g., /predict/aurora")

@app.get("/predict/{dataset_name}")
def predict_yield(dataset_name: str):
    if dataset_name not in csv_mapping:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    
    file_path = csv_mapping[dataset_name]
    
    return predict_crop_yield(file_path)
