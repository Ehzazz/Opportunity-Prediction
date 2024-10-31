from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.responses import StreamingResponse
import io

# Load your trained models
model_total = joblib.load('model_total.pkl')

app = FastAPI()

# Define input data model
class InputDataTotal(BaseModel):
    Year: int

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

    # Format predictions to two decimal points
    formatted_predictions = [round(float(pred), 2) for pred in prediction_total]

    # Read the graph image and convert it to a byte stream
    with open("predicted_graph2.png", "rb") as f:
        image_bytes = f.read()

    return {
        "prediction_total": formatted_predictions,
        "graph": image_bytes  # This will return the image as bytes
    }

# If you want to also serve the graph separately
@app.get("/get_graph/")
async def get_graph():
    # Serve the saved graph image
    return FileResponse("predicted_graph2.png")
