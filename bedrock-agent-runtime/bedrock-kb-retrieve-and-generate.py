import json

import boto3

REGION = "us-east-1"
MODEL_ID = "us.amazon.nova-2-lite-v1:0"
KNOWLEDGE_BASE_ID = "EH5ZJO5D0E"

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


def query_knowledge_base(question):

    response = bedrock_agent_runtime.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": MODEL_ID,
            },
        },
    )

    answer = response["output"]["text"]
    citations = extract_citations(response)

    return answer, citations


def extract_citations(response):

    citations = []

    for citation in response.get("citations", []):
        for ref in citation.get("retrievedReferences", []):

            source = ref.get("location", {}).get("webLocation", {}).get(
                "url"
            ) or ref.get("location", {}).get("s3Location", {}).get("uri")

            citations.append(
                {
                    "text_snippet": ref.get("content", {}).get("text", "")[:200],
                    "source": source,
                }
            )

    return citations


def main():
    if not KNOWLEDGE_BASE_ID:
        raise ValueError("KNOWLEDGE_BASE_ID environment variable is not configured.")

    question = "What food or drink did he ask the White Witch for when they first met, and who prepared it for him and how?"

    answer, citations = query_knowledge_base(question)

    result = {
        "question": question,
        "answer": answer,
        "citations": citations,
    }

    print("\n--- Grounded result ---\n")
    print(result)
    print("\n--- Grounded Answer ---\n")
    print(answer)
    print("\n--- Citations ---\n")
    print(json.dumps(citations, indent=2))


if __name__ == "__main__":
    main()
