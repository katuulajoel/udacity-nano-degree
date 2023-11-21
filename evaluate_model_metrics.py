# evaluate_model_metrics.py
import subprocess
import sys
import argparse
import json
import os
import tarfile
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    }

    return metrics

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    # Extract the model from the tar.gz file
    tar_path = os.path.join('/opt/ml/processing/input/model', 'model.tar.gz')
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path='/opt/ml/processing/input/model')
    
    print(os.listdir('/opt/ml/processing/input/model'))

    # Load the model
    model_path = os.path.join('/opt/ml/processing/input/model', 'model.joblib')
    model = joblib.load(model_path)

    # Load dataset
    X_test = pd.read_csv(f'/opt/ml/processing/input/data/validation_features.csv')
    y_test = pd.read_csv(f'/opt/ml/processing/input/data/validation_labels.csv').values.ravel()

    # Evaluate the model
    evaluation_metrics = evaluate_model(model, X_test, y_test)

    os.makedirs('/opt/ml/processing/output/metrics', exist_ok=True)

    # Save the evaluation metrics
    eval_output_path = os.path.join('/opt/ml/processing/output/metrics', 'evaluation.json')
    with open(eval_output_path, 'w') as f:
        f.write(json.dumps(evaluation_metrics))

    print("Evaluation metrics saved to:", eval_output_path)
