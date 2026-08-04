import boto3
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-east-1")

model_id = "amazon.nova-lite-v1:0"

user_message = "Explain what the book 'The Lion, the Witch and the Wardrobe' by 'C.S. Lewis' is about?"

conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    streaming_response = client.converse_stream(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    for chunk in streaming_response["stream"]:
        if "contentBlockDelta" in chunk:
            text = chunk["contentBlockDelta"]["delta"]["text"]
            print(text, end="")

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

