import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
KNOWLEDGE_BASE_ID = "ZBKDTKOSAO"
QUERY = "BP-STAFF-003"
NUMBER_OF_RESULTS = 20

client = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
)


def retrieve_from_kb(search_type):
    response = client.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={
            "text": QUERY
        },
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": NUMBER_OF_RESULTS,
                "overrideSearchType": search_type,
            }
        },
    )

    return response.get("retrievalResults", [])


def get_source(result):
    location = result.get("location", {})

    if "s3Location" in location:
        return location["s3Location"].get("uri", "Unknown")

    if "webLocation" in location:
        return location["webLocation"].get("url", "Unknown")

    return str(location)


def print_results(search_type, results):
    print(f"\n{'=' * 70}")
    print(f"{search_type} SEARCH")
    print(f"Query: {QUERY}")
    print(f"Results: {len(results)}")
    print(f"{'=' * 70}\n")

    exact_matches = []

    for index, result in enumerate(results, start=1):
        text = result.get("content", {}).get("text", "")
        score = result.get("score")
        source = get_source(result)

        contains_query = QUERY.lower() in text.lower()

        if contains_query:
            exact_matches.append(index)

        print(f"Result {index}")
        print(f"Score: {score}")
        print(f"Exact query found: {'YES' if contains_query else 'NO'}")
        print(f"Source: {source}")
        print(f"Text: {text}")
        print("-" * 70)

    print()

    if exact_matches:
        print(
            f"{QUERY} found at result position(s): "
            f"{', '.join(map(str, exact_matches))}"
        )
    else:
        print(f"{QUERY} was not found in the top {len(results)} results.")


def run_search(search_type):
    try:
        results = retrieve_from_kb(search_type)
        print_results(search_type, results)

    except ClientError as e:
        print(f"\n--- {search_type} SEARCH FAILED ---")
        print(f"AWS error: {e}")


def main():
    run_search("SEMANTIC")
    run_search("HYBRID")


if __name__ == "__main__":
    main()