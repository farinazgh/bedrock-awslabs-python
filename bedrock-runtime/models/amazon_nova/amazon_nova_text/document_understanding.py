"""
This example shows how to perform single-document question answering/summarization without using:
    Knowledge Bases
    RAG
    Embeddings
    Vector databases
    OpenSearch
The PDF is sent directly in the inference request,
making it suitable for one-off document analysis where persistent indexing or retrieval is unnecessary.
the lost in the middle is not relevant here as the length of the pdf is reaonable enough for nova-lite
"""
import boto3
from botocore.exceptions import ClientError


client = boto3.client("bedrock-runtime", region_name="us-east-1")

model_id = "amazon.nova-lite-v1:0"



with open("amazon-nova-service-cards.pdf", "rb") as file:
    document_bytes = file.read()

conversation = [
    {
        "role": "user",
        "content": [
            {"text": "Briefly compare the models described in this document"},
            {
                "document": {
                    # Available formats: html, md, pdf, doc/docx, xls/xlsx, csv, and txt
                    "format": "pdf",
                    "name": "Amazon Nova Service Cards",
                    "source": {"bytes": document_bytes},
                }
            },
        ],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 500, "temperature": 0.3},
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

