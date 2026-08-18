import json
import boto3

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
appconfig = boto3.client("appconfigdata", region_name="us-east-1")

APPCONFIG_APP = "BedrockModelConfig"
APPCONFIG_ENV = "production"
APPCONFIG_PROFILE = "ModelSettings"

config_token = None
cached_config = None


def get_model_config():
    global config_token, cached_config

    try:
        if not config_token:
            session_response = appconfig.start_configuration_session(
                ApplicationIdentifier=APPCONFIG_APP,
                EnvironmentIdentifier=APPCONFIG_ENV,
                ConfigurationProfileIdentifier=APPCONFIG_PROFILE,
            )
            config_token = session_response["InitialConfigurationToken"]

        config_response = appconfig.get_latest_configuration(
            ConfigurationToken=config_token
        )
        config_token = config_response["NextPollConfigurationToken"]

        config_content = config_response["Configuration"].read()
        if config_content:
            cached_config = json.loads(config_content)
            print("Retrieved new configuration from AppConfig")
        else:
            print("No new configuration, using cached version")

        return cached_config if cached_config else get_default_config()

    except Exception as e:
        print(f"Error fetching AppConfig: {e}")
        return get_default_config()


def get_default_config():
    return {
        "currentModel": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "modelParameters": {"maxTokens": 2048, "temperature": 0.7},
    }


def lambda_handler(event, context):
    try:
        user_prompt = event.get("prompt", "Hello, how are you?")

        config = get_model_config()
        model_id = config["currentModel"]
        params = config["modelParameters"]

        print(f"Using model: {model_id}")
        print(f"Parameters: {params}")

        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={
                "maxTokens": params.get("maxTokens", 2048),
                "temperature": params.get("temperature", 0.7),
            },
        )

        generated_text = response["output"]["message"]["content"][0]["text"]

        return {"model": model_id, "response": generated_text, "parameters": params}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}
