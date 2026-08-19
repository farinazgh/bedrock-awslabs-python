import boto3
import json
import time

from numpy import dot
from numpy.linalg import norm

client = boto3.client("bedrock-runtime", region_name="us-east-1")

EMBED_MODEL = "amazon.titan-embed-text-v2:0"
GEN_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
THRESHOLD = 0.40

cache = []


def embed(text):
    """Get embedding vector from Titan."""
    response = client.invoke_model(
        modelId=EMBED_MODEL, body=json.dumps({"inputText": text})
    )

    return json.loads(response["body"].read())["embedding"]


def cosine(a, b):
    """Compute cosine similarity between two vectors."""
    return dot(a, b) / (norm(a) * norm(b))


def answer(query):
    """Check cache, then call model if needed."""
    query_vector = embed(query)

    best_score = 0
    best_response = None

    # Search cache for similar query
    for cached_vec, cached_query, cached_resp in cache:
        score = cosine(query_vector, cached_vec)
        print("cosine similarity: ", score)
        print("\n")
        if score > best_score:
            best_score = score
            best_response = cached_resp

    # Cache hit
    if best_score >= THRESHOLD:
        return f"[CACHE HIT | similarity {best_score:.2f}] " f"{best_response[:120]}..."

    # Cache miss - call model
    start = time.time()

    r = client.invoke_model(
        modelId=GEN_MODEL,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": query}],
            }
        ),
    )

    response = json.loads(r["body"].read())["content"][0]["text"]

    latency = (time.time() - start) * 1000

    # Add result to cache
    cache.append((query_vector, query, response))

    return (
        f"[CACHE MISS | best similarity {best_score:.2f}] "
        f"{response[:120]}... ({latency:.0f}ms)"
    )


def main():
    """Run semantic caching demo."""

    queries = [
        "What is your return policy?",
        "How do I return an item?",
        "What time does the store open?",
        "Can I get a refund?",
    ]

    print("\n=== Bedrock Semantic Cache Demo ===")

    for q in queries:
        print(f"\nQ: {q}")
        print(answer(q))


if __name__ == "__main__":
    main()
