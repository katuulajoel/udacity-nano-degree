import os
import json
import joblib
import numpy as np
import pandas as pd
from io import StringIO


# Function to load the model
def model_fn(model_dir):
    model_file = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_file)
    return model

# Function to preprocess the input data
def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        # Convert the CSV string to a StringIO object (which behaves like a file)
        csv_string = StringIO(request_body)
        # Read the CSV data
        data = pd.read_csv(csv_string, header=None)
        return data
    else:
        # Optionally handle other content types
        raise ValueError("This model only supports CSV input")

# Function to run predictions
def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    return predictions

def output_fn(prediction, response_content_type):
    # Convert numpy array to a Python list
    if isinstance(prediction, np.ndarray):
        prediction_list = prediction.tolist()
    else:
        prediction_list = prediction

    # Check the response content type and format accordingly
    if response_content_type == "application/jsonlines":
        # Format each prediction as a line in JSONLines
        output_lines = [json.dumps({"prediction": pred}) for pred in prediction_list]
        output = "\n".join(output_lines)
    elif response_content_type == "application/json":
        # Format the prediction as standard JSON
        output = json.dumps({"predictions": prediction_list})
    else:
        raise ValueError(f"This model only supports JSONLines or JSON output, not {response_content_type}")

    return output.encode('utf-8')
