import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "fsspec"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "s3fs"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])

import argparse
import os
import boto3
import psutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from imblearn.combine import SMOTETomek
import time
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)
from sagemaker.session import Session

region = 'us-east-1'
print("Region: {}".format(region))
boto_session = boto3.Session(region_name=region)
featurestore_runtime = boto_session.client(
    service_name="sagemaker-featurestore-runtime", region_name=region
)
s3 = boto_session.client(service_name="s3", region_name=region)
sagemaker_session = Session(
    boto_session=boto_session,
    sagemaker_featurestore_runtime_client=featurestore_runtime,
)
bucket = sagemaker_session.default_bucket()
print("The DEFAULT BUCKET is {}".format(bucket))

role = 'arn:aws:iam::863397112005:role/service-role/AmazonSageMaker-ExecutionRole-20231109T153131'

# Function to log CPU and Memory usage
def log_resource_usage():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory.percent}% (Used: {memory.used / (1024**3):.2f} GB, Total: {memory.total / (1024**3):.2f} GB)")


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Parameters for the SageMaker processing job
    parser.add_argument("--processing-instance-count", type=int, default=1)
    parser.add_argument("--processing-instance-type", type=str, default="ml.c5.large")
    parser.add_argument("--train-split-percentage", type=float, default=0.70)
    parser.add_argument("--validation-split-percentage", type=float, default=0.15)
    parser.add_argument("--test-split-percentage", type=float, default=0.15)
    parser.add_argument("--feature-store-offline-prefix", type=str)
    parser.add_argument("--feature-group-name", type=str)

    return parser.parse_args()

def wait_for_feature_group_creation_complete(feature_group):
    try:
        status = feature_group.describe().get("FeatureGroupStatus")
        print("Feature Group status: {}".format(status))
        while status == "Creating":
            print("Waiting for Feature Group Creation")
            time.sleep(5)
            status = feature_group.describe().get("FeatureGroupStatus")
            print("Feature Group status: {}".format(status))
        if status != "Created":
            print("Feature Group status: {}".format(status))
            raise RuntimeError(f"Failed to create feature group {feature_group.name}")
        print(f"FeatureGroup {feature_group.name} successfully created.")
    except:
        print("No feature group created yet.")
        
        
def create_or_load_feature_group(prefix, feature_group_name):

    # Feature Definitions for our records
    feature_definitions = [
        FeatureDefinition(feature_name="type", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="amount", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="oldbalanceOrg", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="day", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="isFraud", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="record_id", feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name="event_time", feature_type=FeatureTypeEnum.STRING),
    ]

    feature_group = FeatureGroup(
        name=feature_group_name, feature_definitions=feature_definitions, sagemaker_session=sagemaker_session
    )

    try:
        print(
            "Waiting for existing Feature Group to become available if it is being created by another instance in our cluster..."
        )
        wait_for_feature_group_creation_complete(feature_group)
    except Exception as e:
        print("Before CREATE FG wait exeption: {}".format(e))

    try:
        record_identifier_feature_name = "record_id"  # Changed from review_id to record_id
        event_time_feature_name = "event_time"  # Added event_time as a feature

        print("Creating Feature Group with role {}...".format(role))
        feature_group.create(
            s3_uri=f"s3://{bucket}/{prefix}",
            record_identifier_name=record_identifier_feature_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=role,
            enable_online_store=False,
        )
        print("Creating Feature Group. Completed.")

        print("Waiting for new Feature Group to become available...")
        wait_for_feature_group_creation_complete(feature_group)
        print("Feature Group available.")
        feature_group.describe()

    except Exception as e:
        print("Exception: {}".format(e))

    return feature_group

def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == "object":
            data_frame[label] = data_frame[label].astype("str").astype("string")
    return data_frame

def main():
    args = parse_arguments()
    
    # feature_group = create_or_load_feature_group(args.feature_store_offline_prefix, args.feature_group_name)

    print(os.listdir('/opt/ml/processing/input/data'))

    try:
        # pass file name online_fraud_dataset in args
        df = pd.read_csv('/opt/ml/processing/input/data/online_fraud_dataset.csv')
        print("Dataframe Shape:", df.shape)
    except Exception as e:
        print("Error loading data:", e)


    try:
        log_resource_usage()
        # Preprocessing steps
        df['type'] = df['type'].replace(['CASH_IN', 'PAYMENT', 'DEBIT'], 'OTHER')
        df.replace(to_replace=['TRANSFER', 'CASH_OUT', 'OTHER'], value=[1, 2, 3], inplace=True)
        df = df[df['type'] != 3]
        df['step'] = df['step'] / 24
        df['step'] = df['step'].round().astype(int)
        df['day'] = df['step'] % 7
        day_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        df['day'] = df['day'].map(day_mapping)

        custom_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ordinal_encoder = OrdinalEncoder(categories=[custom_order])
        df['day'] = ordinal_encoder.fit_transform(df[['day']])

        cols_to_drop = ['nameOrig', 'nameDest', 'step', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'isFlaggedFraud']
        df.drop(cols_to_drop, axis=1, inplace=True)

        df['isFraud'] = df['isFraud'].astype(int)

        smt = SMOTETomek(random_state=2)
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        X, y = smt.fit_resample(X, y)
        print("Data preprocessed.")
        log_resource_usage()
    except Exception as e:
        print("Error during preprocessing:", e)


    try:
        log_resource_usage()
        # Data splitting steps
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(args.validation_split_percentage + args.test_split_percentage), random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(args.test_split_percentage / (args.validation_split_percentage + args.test_split_percentage)), random_state=42)

        scaler = RobustScaler()
        scaled_columns = scaler.fit_transform(X_train[['amount', 'oldbalanceOrg']])
        X_train_scaled = X_train.copy()
        X_train_scaled[['amount', 'oldbalanceOrg']] = scaled_columns

        scaled_columns_validation = scaler.transform(X_val[['amount', 'oldbalanceOrg']])
        X_val_scaled = X_val.copy()
        X_val_scaled[['amount', 'oldbalanceOrg']] = scaled_columns_validation
        
        scaled_columns_test = scaler.transform(X_test[['amount', 'oldbalanceOrg']])
        X_test_scaled = X_test.copy()
        X_test_scaled[['amount', 'oldbalanceOrg']] = scaled_columns_test

        # Adding 'record_id' and 'event_time' to the dataframes
        df_train_records = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
        df_train_records['record_id'] = df_train_records.index
        df_train_records['event_time'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')

        df_validation_records = pd.concat([X_val_scaled, y_val.reset_index(drop=True)], axis=1)
        df_validation_records['record_id'] = df_validation_records.index
        df_validation_records['event_time'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')

        df_test_records = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
        df_test_records['record_id'] = df_test_records.index
        df_test_records['event_time'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')

        print("Train records: {}".format(df_train_records.shape))
        print("Train columns: {}".format(df_train_records.columns))
        print("Train column types: \n{}".format(df_train_records.dtypes))
        print("Data split into train, validation, and test sets.")
        log_resource_usage()
    except Exception as e:
        print("Error during data splitting:", e)


    try:
        log_resource_usage()
        # Data saving steps
        train_features_output_path = os.path.join("opt/ml/processing/output/fraud/train", "train_features.csv")
        train_labels_output_path = os.path.join("opt/ml/processing/output/fraud/train", "train_labels.csv")
        
        validation_features_output_path = os.path.join("opt/ml/processing/output/fraud/validation", "validation_features.csv")
        validation_labels_output_path = os.path.join("opt/ml/processing/output/fraud/validation", "validation_labels.csv")

        test_features_output_path = os.path.join("opt/ml/processing/output/fraud/test", "test_features.csv")
        test_labels_output_path = os.path.join("opt/ml/processing/output/fraud/test", "test_labels.csv")
        
        print("Saving training features to {}".format(train_features_output_path))
        pd.DataFrame(X_train_scaled).to_csv(train_features_output_path, index=False)
        
        print("Saving validation features to {}".format(validation_features_output_path))
        pd.DataFrame(X_val_scaled).to_csv(validation_features_output_path, index=False)

        print("Saving test features to {}".format(test_features_output_path))
        pd.DataFrame(X_test_scaled).to_csv(test_features_output_path, index=False)

        print("Saving training labels to {}".format(train_labels_output_path))
        y_train.to_csv(train_labels_output_path, index=False)
        
        print("Saving validation labels to {}".format(validation_labels_output_path))
        y_val.to_csv(validation_labels_output_path, index=False)

        print("Saving test labels to {}".format(test_labels_output_path))
        y_test.to_csv(test_labels_output_path, index=False)
        print("Data saved to CSV files.")
        log_resource_usage()
    except Exception as e:
        print("Error during data saving:", e)
    
    #Save the processed data to the specified feature store
    #(Add your feature store saving logic here)
    
    """# Add record to feature store
    df_fs_train_records = cast_object_to_string(df_train_records)
    df_fs_validation_records = cast_object_to_string(df_validation_records)
    df_fs_test_records = cast_object_to_string(df_test_records)

     print("Ingesting features...")
    feature_group.ingest(data_frame=df_fs_train_records, max_workers=3, wait=True)
    feature_group.ingest(data_frame=df_fs_validation_records, max_workers=3, wait=True)
    feature_group.ingest(data_frame=df_fs_test_records, max_workers=3, wait=True)

    print('...features ingested!') """
    
    # Check if feature group is created
    """ feature_group_status = None
    try:
        response = feature_group.describe()
        feature_group_status = response.get('FeatureGroupStatus')
        print(f"Feature group status: {feature_group_status}")
    except Exception as e:
        print(f"Error while checking feature group status: {e}")
    
    if feature_group_status == 'Created':
        print('Feature group is successfully created.')
    else:
        print('Feature group creation is not complete or there was an error.')
        return  # or handle this situation as needed """
    
    
    """ offline_store_contents = None
    while offline_store_contents is None:
        objects_in_bucket = s3.list_objects(Bucket=bucket, Prefix=args.feature_store_offline_prefix)
        if "Contents" in objects_in_bucket and len(objects_in_bucket["Contents"]) > 1:
            offline_store_contents = objects_in_bucket["Contents"]
        else:
            print("Waiting for data in offline store...\n")
            sleep(60)

    print("Data available.") """
    

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
