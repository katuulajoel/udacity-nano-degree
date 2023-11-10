import argparse
import os
import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
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
    X_test = pd.read_csv(os.path.join(args.test, 'X_test_nb.csv'))
    y_test = pd.read_csv(os.path.join(args.test, 'y_test_nb.csv')).values.ravel()

    # Instantiate a GaussianNB classifier
    nb_classifier = GaussianNB()

    # Set up the cross-validation scheme
    stratified_kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    # Perform cross-validation
    scores = cross_val_score(nb_classifier, X_train, y_train, cv=stratified_kfold, scoring='accuracy', n_jobs=-1)
    print(f"Cross-validation scores: {scores.mean()} +/- {scores.std()}")

    # Fit the classifier to the data
    nb_classifier.fit(X_train, y_train)

    # Save the classifier to the location specified by args.model_dir
    joblib.dump(nb_classifier, os.path.join(args.model_dir, "model.joblib"))

    # Predict the labels for the test set
    y_pred = nb_classifier.predict(X_test)

    # Generate the classification report
    report = classification_report(y_test, y_pred)
    print(report)

    # Save the report to a file
    with open(os.path.join(args.model_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
