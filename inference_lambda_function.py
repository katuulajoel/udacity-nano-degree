import json
import boto3

def lambda_handler(event, context):
    # Initialize the SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime')

    # Specify your endpoint name (replace 'Your-Endpoint-Name' with your actual endpoint name)
    endpoint_name = event['EndpointName']

    # Original payload
    original_payload = {
        "type": event["type"],  # Example value for "type" after encoding TRANSFER/CASH_OUT/OTHER as 1/2/3
        "amount": event["amount"],  # Example numeric value for "amount"
        "oldbalanceOrg": event["oldbalanceOrg"],  # Example numeric value for "oldbalanceOrg"
        "day": event["day"]  # Example value for "day" after encoding days of the week as integers
    }

    # Convert the original JSON payload to a CSV string
    payload_csv = f"{original_payload['type']},{original_payload['amount']},{original_payload['oldbalanceOrg']},{original_payload['day']}"
    
    # Encode payload to bytes
    payload_bytes = payload_csv.encode('utf-8')

    # Invoke the SageMaker endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',  # Updated content type to 'text/csv'
        Body=payload_bytes
    )

    # Get the inference result and return it
    result = response['Body'].read().decode()
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
