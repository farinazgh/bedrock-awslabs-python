"""
Bedrock model quality validator.

Runs benchmark prompts against Amazon Bedrock and evaluates:

1. Simple key-fact / hallucination heuristic
2. Semantic similarity using Titan Text Embeddings V2
3. Semantic drift using a similarity threshold
4. Aggregate CloudWatch metrics

Requirements:
    pip install boto3

AWS permissions:
    bedrock:InvokeModel
    cloudwatch:PutMetricData
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

# ============================================================
# Configuration
# ============================================================

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "amazon.nova-lite-v1:0",
)

EMBEDDING_MODEL_ID = os.getenv(
    "BEDROCK_EMBEDDING_MODEL_ID",
    "amazon.titan-embed-text-v2:0",
)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

CLOUDWATCH_NAMESPACE = os.getenv(
    "CLOUDWATCH_NAMESPACE",
    "BedrockModelValidation",
)

# benchmark_prompts.json should be in the same directory
# as this Python file.
BENCHMARK_FILE = Path(__file__).with_name("benchmark_prompts.json")


# ============================================================
# AWS clients
# ============================================================

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
)

cloudwatch = boto3.client(
    "cloudwatch",
    region_name=AWS_REGION,
)


# ============================================================
# Fallback benchmark data
# ============================================================

DEFAULT_BENCHMARKS = [
    {
        "prompt": "What is the capital of France?",
        "expectedOutput": "Paris",
        "category": "factual",
    },
    {
        "prompt": ("Summarize the concept of photosynthesis " "in one sentence."),
        "expectedOutput": (
            "Photosynthesis is the process by which plants "
            "convert sunlight, water, and carbon dioxide "
            "into glucose and oxygen."
        ),
        "category": "summarization",
    },
    {
        "prompt": (
            "What programming language is primarily used " "for iOS app development?"
        ),
        "expectedOutput": "Swift",
        "category": "factual",
    },
    {
        "prompt": ("Explain the difference between TCP and UDP " "in one sentence."),
        "expectedOutput": (
            "TCP is a connection-oriented protocol that "
            "guarantees reliable, ordered delivery of data, "
            "while UDP is connectionless and prioritizes "
            "speed over reliability."
        ),
        "category": "technical",
    },
    {
        "prompt": ("What year did the first human land on the Moon?"),
        "expectedOutput": "1969",
        "category": "factual",
    },
]


# ============================================================
# Benchmark loading
# ============================================================


def load_benchmarks(
    path: Path,
) -> list[dict[str, str]]:
    """
    Load benchmark prompts from JSON.

    The JSON format is expected to contain:

    {
        "prompt": "...",
        "expectedOutput": "...",
        "category": "..."
    }
    """

    if path.exists():
        print(f"Loading benchmark file:")
        print(path)

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    else:
        print(
            f"\nBenchmark file not found:\n{path}\n"
            "Using built-in benchmark prompts instead."
        )

        data = DEFAULT_BENCHMARKS

    if not isinstance(data, list):
        raise ValueError("Benchmark JSON must contain a list of objects.")

    benchmarks = []

    for index, item in enumerate(data, start=1):

        if not isinstance(item, dict):
            raise ValueError(f"Benchmark item #{index} must be an object.")

        prompt = item.get("prompt")
        expected = item.get("expectedOutput")
        category = item.get("category", "unknown")

        if not isinstance(prompt, str):
            raise ValueError(f"Benchmark item #{index} is missing " "'prompt'.")

        if not isinstance(expected, str):
            raise ValueError(f"Benchmark item #{index} is missing " "'expectedOutput'.")

        benchmarks.append(
            {
                "prompt": prompt,
                "expected": expected,
                "category": category,
            }
        )

    if not benchmarks:
        raise ValueError("Benchmark contains no test cases.")

    return benchmarks


# ============================================================
# Generator model
# ============================================================


def invoke_model(prompt: str) -> str:
    """
    Send the benchmark prompt to the Bedrock model.
    """

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt,
                    }
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": 256,
            "temperature": 0,
        },
    )

    content = response["output"]["message"]["content"]

    text_parts = [item["text"] for item in content if "text" in item]

    return "".join(text_parts).strip()


# ============================================================
# Titan embeddings
# ============================================================


def get_embedding(text: str) -> list[float]:
    """
    Generate a normalized embedding using
    Titan Text Embeddings V2.
    """

    request_body = json.dumps(
        {
            "inputText": text,
            "dimensions": 256,
            "normalize": True,
        }
    )

    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=request_body,
        contentType="application/json",
        accept="application/json",
    )

    payload = json.loads(response["body"].read())

    return payload["embedding"]


# ============================================================
# Cosine similarity
# ============================================================


def cosine_similarity(
    vec_a: list[float],
    vec_b: list[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    if len(vec_a) != len(vec_b):
        raise ValueError("Embedding vectors have different dimensions.")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    magnitude_a = math.sqrt(sum(a * a for a in vec_a))

    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# Hallucination heuristic
# ============================================================


def check_hallucination(
    response: str,
    expected: str,
) -> bool:
    """
    Simple benchmark heuristic.

    Returns True when the expected answer is not literally
    present in the generated response.

    NOTE:
    This is NOT a sophisticated hallucination detector.
    It is intentionally kept close to the original lab.
    """

    return expected.lower() not in response.lower()


# ============================================================
# Semantic similarity / semantic drift
# ============================================================


def check_semantic_drift(
    response: str,
    expected: str,
) -> float:
    """
    Calculate semantic similarity between the generated
    response and the benchmark's expected answer.
    """

    response_embedding = get_embedding(response)

    expected_embedding = get_embedding(expected)

    return cosine_similarity(
        response_embedding,
        expected_embedding,
    )


# ============================================================
# Evaluate one benchmark
# ============================================================


def evaluate_benchmark(
    benchmark: dict[str, str],
) -> dict[str, Any]:

    prompt = benchmark["prompt"]
    expected = benchmark["expected"]
    category = benchmark["category"]

    response = invoke_model(prompt)

    hallucinated = check_hallucination(
        response,
        expected,
    )

    similarity = check_semantic_drift(
        response,
        expected,
    )

    semantic_drift = similarity < SIMILARITY_THRESHOLD

    return {
        "category": category,
        "prompt": prompt,
        "expected": expected,
        "response": response,
        "hallucinated": hallucinated,
        "similarity": similarity,
        "drifted": semantic_drift,
    }


# ============================================================
# CloudWatch metrics
# ============================================================


def publish_metrics(
    results: list[dict[str, Any]],
) -> None:

    total = len(results)

    if total == 0:
        return

    hallucinations = sum(1 for result in results if result["hallucinated"])

    drift_count = sum(1 for result in results if result["drifted"])

    average_similarity = sum(result["similarity"] for result in results) / total

    hallucination_rate = hallucinations / total

    timestamp = datetime.now(timezone.utc)

    dimensions = [
        {
            "Name": "ModelId",
            "Value": MODEL_ID,
        }
    ]

    cloudwatch.put_metric_data(
        Namespace=CLOUDWATCH_NAMESPACE,
        MetricData=[
            {
                "MetricName": "HallucinationRate",
                "Value": hallucination_rate,
                "Unit": "None",
                "Timestamp": timestamp,
                "Dimensions": dimensions,
            },
            {
                "MetricName": "AverageCosineSimilarity",
                "Value": average_similarity,
                "Unit": "None",
                "Timestamp": timestamp,
                "Dimensions": dimensions,
            },
            {
                "MetricName": "SemanticDriftCount",
                "Value": drift_count,
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


# ============================================================
# Console report
# ============================================================


def print_results(
    results: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 70)
    print("BEDROCK MODEL QUALITY VALIDATION")
    print("=" * 70)

    print(f"Region:               {AWS_REGION}")
    print(f"Generator model:      {MODEL_ID}")
    print(f"Embedding model:      {EMBEDDING_MODEL_ID}")
    print(f"Similarity threshold: " f"{SIMILARITY_THRESHOLD:.2f}")

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print("-" * 70)
        print(f"TEST {index}")
        print("-" * 70)

        print(f"Category:      " f"{result['category']}")

        print(f"Prompt:        " f"{result['prompt']}")

        print(f"Expected:      " f"{result['expected']}")

        print(f"Response:      " f"{result['response']}")

        print(f"Hallucinated:  " f"{result['hallucinated']}")

        print(f"Similarity:    " f"{result['similarity']:.4f}")

        print(f"Semantic drift:" f" {result['drifted']}")

    total = len(results)

    hallucinations = sum(1 for result in results if result["hallucinated"])

    drift_count = sum(1 for result in results if result["drifted"])

    average_similarity = (
        sum(result["similarity"] for result in results) / total if total else 0.0
    )

    hallucination_rate = hallucinations / total if total else 0.0

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Prompts tested:       " f"{total}")

    print(f"Hallucination rate:   " f"{hallucination_rate:.2%}")

    print(f"Average similarity:   " f"{average_similarity:.4f}")

    print(f"Semantic drift count: " f"{drift_count}")


# ============================================================
# Main
# ============================================================


def main() -> None:

    benchmarks = load_benchmarks(BENCHMARK_FILE)

    print(f"\nLoaded {len(benchmarks)} " f"benchmark prompts.")

    results = []

    for index, benchmark in enumerate(
        benchmarks,
        start=1,
    ):

        print(f"\nRunning benchmark " f"{index}/{len(benchmarks)}...")

        result = evaluate_benchmark(benchmark)

        results.append(result)

    print_results(results)

    publish_metrics(results)

    print()
    print("Metrics published to CloudWatch namespace: " f"{CLOUDWATCH_NAMESPACE}")


if __name__ == "__main__":
    main()
