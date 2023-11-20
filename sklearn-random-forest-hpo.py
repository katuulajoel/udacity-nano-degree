import argparse
import joblib
import os
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the tree')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Minimum number of samples required to split an internal node')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='Minimum number of samples required to be at a leaf node')
    parser.add_argument('--criterion', type=str, default='gini', help='Function to measure the quality of a split')

    # Parse arguments
    args = parser.parse_args()

    print(os.listdir('/opt/ml/input/data'))
    print("args", args)

    train_data_dir = '/opt/ml/input/data/train'  # SageMaker's local path for training data
    test_data_dir = '/opt/ml/input/data/test'   # SageMaker's local path for test data

    # Load dataset
    X_train = pd.read_csv('/opt/ml/input/data/train/train_features.csv')
    y_train = pd.read_csv('/opt/ml/input/data/train/train_labels.csv').values.ravel()
    X_test = pd.read_csv('/opt/ml/input/data/test/test_features.csv')
    y_test = pd.read_csv('/opt/ml/input/data/test/test_labels.csv').values.ravel()

    # Instantiate a RandomForestClassifier
    rf_classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion,
        random_state=42
    )

    # Fit the RandomForestClassifier to the data
    rf_classifier.fit(X_train, y_train)

    # Save the model to the specified directory
    if not os.path.exists('/opt/ml/model'):
        os.makedirs('/opt/ml/model')
    joblib.dump(rf_classifier, '/opt/ml/model/model.joblib')

    # Predict the labels for the test set
    y_pred = rf_classifier.predict(X_test)

    # Generate and print the classification report
    report = classification_report(y_test, y_pred)
    print(report)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logging.info(f"test:precision: {precision}")
    logging.info(f"test:recall: {recall}")
    logging.info(f"test:f1-score: {f1}")

    # Save the report to a file in the model directory
    with open('/opt/ml/model/classification_report.txt', 'w') as f:
        f.write(report)
