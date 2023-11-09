import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from scipy.stats import randint as sp_randint

if __name__ =='__main__':
    # Parse argument variables passed via the CreateTrainingJob request
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this script, they won't be used, but this is how you could pass them.
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=None)

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args = parser.parse_args()

    # Load dataset from the location specified by args.train and args.test
    X_train = pd.read_csv(os.path.join(args.train, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(args.train, 'y_train.csv'))
    X_test = pd.read_csv(os.path.join(args.test, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(args.test, 'y_test.csv'))

    # Define the parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': sp_randint(50, 200),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': sp_randint(2, 10),
        'min_samples_leaf': sp_randint(1, 4)
    }

    # Instantiate a RandomForestClassifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Set up the cross-validation scheme
    stratified_kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)  # Reduced repeats for faster execution

    # Set up the RandomizedSearchCV object
    random_search = RandomizedSearchCV(estimator=rf_classifier,
                                       param_distributions=param_dist,
                                       n_iter=10,  # Reduced number of iterations for faster execution
                                       cv=stratified_kfold,
                                       scoring='f1',
                                       n_jobs=-1,
                                       random_state=42)

    # Fit the RandomizedSearchCV object to the data
    random_search.fit(X_train, y_train.values.ravel())

    # Get the best model
    best_model = random_search.best_estimator_

    # Save the best model to the location specified by args.model_dir
    joblib.dump(best_model, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(args.model_dir, "scaler.joblib"))

    # Predict the labels for the test set
    y_pred = best_model.predict(X_test)

    # Generate the classification report
    report = classification_report(y_test, y_pred)
    print(report)

    # Save the report to a file
    with open(os.path.join(args.model_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
