import boto3
import json

def lambda_handler(event, context):
    # The name of the existing SageMaker Model
    existing_model_name = event['ModelName']  # Replace with your actual model name

    # Initialize the SageMaker Boto3 client
    sagemaker_client = boto3.client('sagemaker')

    # Construct the endpoint configuration name
    # Ensure the total length does not exceed 63 characters
    prefix = 'fraud-endpoint-config-'
    max_length_for_model_name = 63 - len(prefix)
    truncated_model_name = existing_model_name[:max_length_for_model_name]

    endpoint_config_name = prefix + truncated_model_name
    endpoint_name = 'fraud-endpoint-' + truncated_model_name  # You may also need to ensure this name is within limits

    # Create an endpoint configuration with the existing SageMaker Model
    create_endpoint_config_response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'ModelName': existing_model_name,
                'VariantName': 'AllTraffic',
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m5.large',
                'InitialVariantWeight': 1
            },
        ]
    )

    # Deploy the endpoint
    create_endpoint_response = sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

    # Print out the response
    print(create_endpoint_response)

    return {
        'statusCode': 200,
        'body': json.dumps('Endpoint created successfully!')
    }
