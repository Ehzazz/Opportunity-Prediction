from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load your trained models
model_total = joblib.load('model_total.pkl')
model_count = joblib.load('model_count.pkl')
model_total_extended = joblib.load('model_total_extended.pkl')
model_count_extended = joblib.load('model_count_extended.pkl')

app = FastAPI()

# Define input data models
class InputDataTotal(BaseModel):
    Year: int

class InputDataCount(BaseModel):
    Year: int
    Average_Cost_Won: float  # Assuming this is a float value

@app.post("/predict/total/")
async def predict_total(input_data: InputDataTotal):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame({
        'Year': [input_data.Year]
    })
    
    # Prepare input data for model prediction
    input_features = input_df.values
    
    # Make predictions
    prediction_total = model_total.predict(input_features)
    prediction_total_extended = model_total_extended.predict(input_features)

    # Return predictions
    return {
        "prediction_total_Amount": prediction_total.tolist(),
        "prediction_total_Amount_extended": prediction_total_extended.tolist(),
    }

@app.post("/predict/count/")
async def predict_count(input_data: InputDataCount):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame({
        'Year': [input_data.Year],
        'Average Cost Won': [input_data.Average_Cost_Won]  # Use correct column name
    })
    
    # Prepare input data for model prediction
    input_features = input_df.values
    
    # Make predictions
    prediction_count = model_count.predict(input_features)
    prediction_count_extended = model_count_extended.predict(input_features)

    # Return predictions
    return {
        "Number of opportunities Won": prediction_count.tolist(),
        "Number of Opportuniites won 2025": prediction_count_extended.tolist(),
    }

# If running this script directly, use the command: uvicorn your_script_name:app --reload
