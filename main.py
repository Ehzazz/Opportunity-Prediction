from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

# Load your trained models
model_total = joblib.load('model_total.pkl')

app = FastAPI()

# Define input data model
class InputDataTotal(BaseModel):
    Year: int

def create_graph():
    # Example data for plotting
    years = ['2022', '2023', '2024']
    total_amounts = [1000, 1500, 2000]  # Replace with your actual data or predictions

    # Create a bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(years, total_amounts, color=['blue', 'blue', 'orange'])

    # Adding title and labels
    plt.title('Total Amount for Won Opportunities of 2022, 2023, and Predicted for 2024', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Total Amount Won ($)', fontsize=14)

    # Adding data labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.2f}', ha='center', va='bottom')

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()  # Close the plot to free memory
    return buf

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

    # Create the graph
    graph_buffer = create_graph()

    # Encode the graph as base64
    graph_base64 = base64.b64encode(graph_buffer.getvalue()).decode('utf-8')

    # Return predictions and graph
    return JSONResponse(content={
        "prediction_total": formatted_predictions,
        "graph": f"data:image/png;base64,{graph_base64}"  # Embed the graph as a base64 string
    })
