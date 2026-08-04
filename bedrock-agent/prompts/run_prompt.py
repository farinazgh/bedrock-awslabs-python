import logging

from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def invoke_prompt(client, prompt_arn, variables):
    print()
    try:
        logger.info("Generating response with prompt: %s", prompt_arn)

        prompt_variables = {
            key: {"text": str(value)} for key, value in variables.items()
        }

        response = client.converse(modelId=prompt_arn, promptVariables=prompt_variables)

        message = response["output"]["message"]
        result = ""
        for content in message["content"]:
            result += content["text"]

        logger.info("Finished generating response with prompt: %s", prompt_arn)

        return result

    except ClientError as e:
        logger.exception("Client error invoking prompt version: %s", str(e))
        raise
    except Exception as e:
        logger.error("Error invoking prompt: %s", str(e))
        raise


