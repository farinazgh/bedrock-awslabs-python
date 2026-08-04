import logging
import re
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prompt(
    client, prompt_name, prompt_description, prompt_template, model_id=None
):
    print()
    try:
        logger.info("Creating prompt: %s.", prompt_name)

        variant = {
            "name": "default",
            "templateType": "TEXT",
            "templateConfiguration": {
                "text": {"text": prompt_template, "inputVariables": []}
            },
        }

        variables = re.findall(r"{{(.*?)}}", prompt_template)
        for var in variables:
            variant["templateConfiguration"]["text"]["inputVariables"].append(
                {"name": var.strip()}
            )

        if model_id:
            variant["modelId"] = model_id

        create_params = {
            "name": prompt_name,
            "description": prompt_description,
            "variants": [variant],
        }

        response = client.create_prompt(**create_params)

        logger.info(
            "Successfully created prompt: %s. ID: %s", prompt_name, response["id"]
        )

        return response

    except ClientError as e:
        logger.exception("Client error creating prompt: %s", str(e))
        raise

    except Exception as e:
        logger.exception("Unexpected error creating prompt: %s", str(e))
        raise


def create_prompt_version(client, prompt_id, description=None):

    print()
    try:
        logger.info("Creating version for prompt ID: %s.", prompt_id)

        create_params = {"promptIdentifier": prompt_id}

        if description:
            create_params["description"] = description

        response = client.create_prompt_version(**create_params)

        logger.info("Successfully created prompt version: %s", response["version"])
        logger.info("Prompt version ARN: %s", response["arn"])

        return response

    except ClientError as e:
        logger.exception("Client error creating prompt version: %s", str(e))
        raise

    except Exception as e:
        logger.exception("Unexpected error creating prompt version: %s", str(e))
        raise


def get_prompt(client, prompt_id):

    try:
        logger.info("Getting prompt ID: %s.", prompt_id)

        response = client.get_prompt(promptIdentifier=prompt_id)

        logger.info("Retrieved prompt ID: %s. Name: %s", prompt_id, response["name"])

        return response

    except ClientError as e:
        logger.exception("Client error getting prompt: %s", str(e))
        raise

    except Exception as e:
        logger.exception("Unexpected error getting prompt: %s", str(e))
        raise


def delete_prompt(client, prompt_id):

    try:
        logger.info("Deleting prompt ID: %s.", prompt_id)

        response = client.delete_prompt(promptIdentifier=prompt_id)

        logger.info("Finished deleting prompt ID: %s", prompt_id)

        return response

    except ClientError as e:
        logger.exception("Client error deleting prompt: %s", str(e))
        raise

    except Exception as e:
        logger.exception("Unexpected error deleting prompt: %s", str(e))
        raise


def list_prompts(client, max_results=10):

    try:
        logger.info("Listing prompts:")

        paginator = client.get_paginator("list_prompts")

        pagination_config = {"maxResults": max_results}

        all_prompts = []

        for page in paginator.paginate(**pagination_config):
            all_prompts.extend(page.get("promptSummaries", []))

        logger.info("Successfully listed %s prompts.", len(all_prompts))
        return all_prompts

    except ClientError as e:
        logger.exception("Client error listing prompts: %s", str(e))
        raise
    except Exception as e:
        logger.exception("Unexpected error listing prompts: %s", str(e))
        raise
