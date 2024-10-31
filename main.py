from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd

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

    # Specify the path to the saved graph image
    graph_path = "predicted_graph2.png"

    # Return predictions and the graph image
    return {
        "prediction_total": formatted_predictions,
        "graph_url": f"/get_graph/"
    }

@app.get("/get_graph/")
async def get_graph():
    # Serve the saved graph image
    return FileResponse("predicted_graph2.png")
