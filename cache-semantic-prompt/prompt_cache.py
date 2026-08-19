import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")

MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


# Simulate a large system prompt (6000+ tokens)
SYSTEM_PROMPT = """You are a customer support assistant. Here are our policies:

RETURNS POLICY:
Returns are accepted within 30 days of delivery for unused items in original packaging.
Items must be in resellable condition.
Custom or personalized items are non-returnable.
Sale items are final sale unless defective.

SHIPPING POLICY:
Standard shipping: 5-7 business days
Express shipping: 2-3 business days
International shipping: 10-14 business days
Free shipping on orders over $50
We ship to: USA, Canada, UK, France, Germany, Italy, Spain, Australia, Japan

REFUND POLICY:
Refunds processed within 5-7 business days of receiving returned item.
Original shipping costs are non-refundable.
Gift card purchases are non-refundable but never expire.

ACCOUNT SECURITY:
Two-factor authentication required for orders over $500.
Password resets valid for 30 minutes.
Account lockout after 5 failed login attempts.

PRODUCT CATALOG (sample - imagine 5000+ more tokens):
SKU-001: Widget Basic - $19.99
SKU-002: Widget Pro - $39.99
SKU-003: Widget Enterprise - $99.99
...
[Imagine 6000+ tokens of product catalog, policies, FAQ content here]
...

Always respond in a friendly, concise tone."""


def call(query):
    """Make a Bedrock call with prompt caching enabled."""

    response = client.invoke_model(
        modelId=MODEL,
        body=json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "system": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                        # type = "persistent"
                        # type = "durable"
                        # type = "forever"
                        # "cache_control": {
                        #     "type": "ephemeral",
                        #     "ttl": "5m"
                        # }
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
            }
        ),
    )

    body = json.loads(response["body"].read())
    usage = body["usage"]

    print(f"Query: {query}")
    print(f"  Cache creation: " f"{usage.get('cache_creation_input_tokens', 0)}")
    print(f"  Cache read: " f"{usage.get('cache_read_input_tokens', 0)}")
    print(f"  Regular input: {usage['input_tokens']}")
    print()


def main():
    """Run the Bedrock prompt caching demo."""

    print("\n=== Bedrock Prompt Caching Demo ===\n")

    # First call - creates/writes the prompt cache
    call("What is your return policy?")

    # Second call - reads from cache if still within the cache TTL
    call("How do I cancel my order?")


if __name__ == "__main__":
    main()
