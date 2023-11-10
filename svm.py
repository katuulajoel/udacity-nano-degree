import argparse
import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report

if __name__ == '__main__':
    # Parse argument variables passed via the CreateTrainingJob request
    parser = argparse.ArgumentParser()

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args = parser.parse_args()

    # Load dataset from the location specified by args.train and args.test
    X_train = pd.read_csv(os.path.join(args.train, 'X_train_resampled.csv'))
    y_train = pd.read_csv(os.path.join(args.train, 'y_train_resampled.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(args.test, 'X_test_svm.csv'))
    y_test = pd.read_csv(os.path.join(args.test, 'y_test_svm.csv')).values.ravel()

    # Define the parameter distribution for RandomizedSearchCV
    param_dist = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    }

    # Instantiate a SVC classifier
    svm_classifier = SVC(random_state=42)

    # Set up the cross-validation scheme
    stratified_kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    # Set up the RandomizedSearchCV object
    randomized_search = RandomizedSearchCV(
        estimator=svm_classifier,
        param_distributions=param_dist,
        n_iter=10,
        cv=stratified_kfold,
        scoring='recall',
        n_jobs=-1,
        random_state=42,
    )

    # Fit the RandomizedSearchCV object to the data
    randomized_search.fit(X_train, y_train)

    # Get the best model
    best_model = randomized_search.best_estimator_

    # Save the best model to the location specified by args.model_dir
    joblib.dump(best_model, os.path.join(args.model_dir, "model.joblib"))

    # Predict the labels for the test set
    y_pred = best_model.predict(X_test)

    # Generate the classification report
    report = classification_report(y_test, y_pred)
    print(report)

    # Save the report to a file
    with open(os.path.join(args.model_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)