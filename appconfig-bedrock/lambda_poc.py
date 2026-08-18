from typing import Any

import boto3
import json


def lambda_handler(event, context):

    config_data = get_app_config_data()

    model_id = config_data["modelId"]
    max_tokens = config_data.get("maxTokens", 300)
    temperature = config_data.get("temperature", 0.7)

    print(f"Model selected from AppConfig: {model_id}")
    print(f"max_tokensl selected from AppConfig: {max_tokens}")
    print(f"temperature selected from AppConfig: {temperature}")

    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

    prompt = event.get(
        "prompt", "Explain what a proof of concept is in 2 bullet points."
    )

    bedrock_response = bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )

    response_text = bedrock_response["output"]["message"]["content"][0]["text"]
    usage = bedrock_response["usage"]

    return {
        "statusCode": 200,
        "model_used": model_id,
        "source": "AppConfig",
        "prompt": prompt,
        "response": response_text,
        "token_usage": {
            "input_tokens": usage["inputTokens"],
            "output_tokens": usage["outputTokens"],
            "total_tokens": usage["totalTokens"],
        },
    }


def get_app_config_data() -> Any:
    appconfig = boto3.client("appconfigdata", region_name="us-east-1")

    session = appconfig.start_configuration_session(
        ApplicationIdentifier="narnia",
        EnvironmentIdentifier="dev",
        ConfigurationProfileIdentifier="model-config",
    )

    response = appconfig.get_latest_configuration(
        ConfigurationToken=session["InitialConfigurationToken"]
    )

    config_data = json.loads(response["Configuration"].read())
    return config_data
