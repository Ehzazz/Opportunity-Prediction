# api.py

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from Averagecostprediction import plot_comparison_graph  # Import your function from ML model

# Load your trained model
model_total = joblib.load('model_total.pkl')

app = FastAPI()

# Define input data model
class InputDataTotal(BaseModel):
    Year: int

@app.post("/predict/total/")
async def predict_total(input_data: InputDataTotal):
    user_input_year = input_data.Year

    # Ensure the year is within the expected range
    if user_input_year < 2020 or user_input_year > 2030:
        return JSONResponse(content={"error": "Year must be between 2020 and 2030."}, status_code=400)

    # Prepare input data for model prediction
    year_array = np.array([[user_input_year]])
    
    # Make predictions
    prediction_total = model_total.predict(year_array)
    formatted_prediction = round(float(prediction_total[0]), 2)

    # Call the graph plotting function from your ML model
    graph_buffer = plot_comparison_graph(user_input_year)  # Generate the graph for the given year

    # Convert the graph to base64
    graph_base64 = base64.b64encode(graph_buffer.getvalue()).decode('utf-8')

    return JSONResponse(content={
        "prediction_total": formatted_prediction,
        "graph": f"data:image/png;base64,{graph_base64}",  # Embed the graph as a base64 string
        "message": f"Graph for {user_input_year} has been generated."
    })

# You can run the FastAPI server and call the endpoint with a year to see the predictions and generated graph.
