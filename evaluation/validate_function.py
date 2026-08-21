import json
import math
import os
import boto3
from datetime import datetime, timezone

BUCKET_NAME = os.environ["BUCKET_NAME"]
PROMPTS_KEY = os.environ["PROMPTS_KEY"]
MODEL_ID = os.environ["MODEL_ID"]
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
SIMILARITY_THRESHOLD = float(os.environ["SIMILARITY_THRESHOLD"])
NAMESPACE = "GenAI/DeploymentValidation"

s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")
cloudwatch = boto3.client("cloudwatch")


def invoke_model(prompt):
    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 256, "temperature": 0},
    )
    return response["output"]["message"]["content"][0]["text"]


def get_embedding(text):
    body = json.dumps({"inputText": text, "dimensions": 256, "normalize": True})

    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    return json.loads(response["body"].read())["embedding"]


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def check_hallucination(response, expected):
    return expected.lower() not in response.lower()


def check_semantic_drift(response, expected):
    emb_response = get_embedding(response)
    emb_expected = get_embedding(expected)

    return cosine_similarity(emb_response, emb_expected)


def publish_metrics(results):
    total = len(results)

    hallucinations = sum(1 for r in results if r["hallucinated"])

    avg_similarity = sum(r["similarity"] for r in results) / total if total else 0

    drift_detected = sum(1 for r in results if r["similarity"] < SIMILARITY_THRESHOLD)

    timestamp = datetime.now(timezone.utc)

    dimensions = [{"Name": "ModelId", "Value": MODEL_ID}]

    cloudwatch.put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[
            {
                "MetricName": "HallucinationRate",
                "Value": hallucinations / total if total else 0,
                "Unit": "None",
                "Timestamp": timestamp,
                "Dimensions": dimensions,
            },
            {
                "MetricName": "AverageCosineSimilarity",
                "Value": avg_similarity,
                "Unit": "None",
                "Timestamp": timestamp,
                "Dimensions": dimensions,
            },
            {
                "MetricName": "SemanticDriftCount",
                "Value": drift_detected,
                "Unit": "Count",
                "Timestamp": timestamp,
                "Dimensions": dimensions,
            },
            {
                "MetricName": "TotalPromptsTested",
                "Value": total,
                "Unit": "Count",
                "Timestamp": timestamp,
                "Dimensions": dimensions,
            },
        ],
    )


def handler(event, context):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=PROMPTS_KEY)

    benchmarks = json.loads(obj["Body"].read().decode("utf-8"))

    results = []

    for bench in benchmarks:
        prompt = bench["prompt"]
        expected = bench["expectedOutput"]
        category = bench["category"]

        response = invoke_model(prompt)

        hallucinated = check_hallucination(response, expected)

        similarity = check_semantic_drift(response, expected)

        results.append(
            {
                "prompt": prompt,
                "expected": expected,
                "response": response,
                "category": category,
                "hallucinated": hallucinated,
                "similarity": round(similarity, 3),
            }
        )

    publish_metrics(results)

    total = len(results)

    hallucinations = sum(1 for r in results if r["hallucinated"])

    drifts = sum(1 for r in results if r["similarity"] < SIMILARITY_THRESHOLD)

    avg_sim = sum(r["similarity"] for r in results) / total if total else 0

    passed = hallucinations == 0 and drifts == 0

    print(f"Total prompts: {total}")
    print(f"Hallucinations: {hallucinations}")
    print(f"Semantic drift: {drifts} " f"(threshold: {SIMILARITY_THRESHOLD})")
    print(f"Avg cosine similarity: {avg_sim:.3f}")
    print(f'Result: {"PASSED" if passed else "FAILED"}')

    return {
        "passed": passed,
        "totalPrompts": total,
        "hallucinations": hallucinations,
        "semanticDrifts": drifts,
        "averageSimilarity": round(avg_sim, 3),
        "threshold": SIMILARITY_THRESHOLD,
    }
