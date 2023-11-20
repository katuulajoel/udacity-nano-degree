import boto3
import json

runtime= boto3.client('runtime.sagemaker')

body = {
"inputs": "Hello, world",
"parameters": {"max_new_tokens": 1000, "top_p": 0.9, "temperature": 0.6}
}


response = runtime.invoke_endpoint(EndpointName="jumpstart-dft-meta-textgeneration-llama-2-7b",
                                    ContentType='application/json',
                                    Body=json.dumps(body),
                                    CustomAttributes='accept_eula=true')
print(response)
result = json.loads(response['Body'].read().decode())
print(result)
