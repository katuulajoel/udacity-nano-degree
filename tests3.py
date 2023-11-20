import s3fs
import pandas as pd

def main():
    # Manually set the S3 path for local testing
    input_data = 's3://sagemaker-us-east-1-863397112005/data/online_fraud_dataset.csv'

    # Initialize s3fs to interact with S3
    fs = s3fs.S3FileSystem(anon=False)

    # Reading the data directly from S3
    with fs.open(input_data, 'rb') as f:
        df = pd.read_csv(f)

    # Print the first few rows to verify
    print(df.head())

if __name__ == "__main__":
    main()