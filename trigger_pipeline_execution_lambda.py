import json
import boto3

def lambda_handler(event, context):
    # Initialize the SageMaker client
    sagemaker_client = boto3.client('sagemaker')

    # Extract the pipeline name from the event
    pipeline_name = event['pipeline']

    # Extract pipeline parameters from the event
    # Assuming the parameters are provided in the event under the key 'parameters'
    pipeline_parameters = event.get('parameters', [])

    # Start the execution of the pipeline with parameters
    response = sagemaker_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName='PipelineExecutionViaLambda',
        PipelineParameters=pipeline_parameters,
        PipelineExecutionDescription='Triggered by AWS Lambda'
    )

    # Return the response
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
