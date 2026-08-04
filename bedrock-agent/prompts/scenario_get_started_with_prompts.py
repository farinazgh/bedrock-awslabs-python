"""
Shows how to use the AWS SDK for Python (Boto3) with Amazon Bedrock
to create and use Amazon Bedrock managed prompts.

This scenario demonstrates the following:
1. Create a managed prompt
2. Invoke the prompt
3. Update the prompt
4. Invoke the updated prompt
5. Clean up resources (optional)
"""

import argparse
import boto3
import logging
import time

from prompt import create_prompt, create_prompt_version, delete_prompt
from run_prompt import invoke_prompt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_scenario(bedrock_client, bedrock_runtime_client, model_id, cleanup=True):
    prompt_id = None

    try:
        print("\n=== Step 1: Creating a prompt ===")
        prompt_name = f"PlaylistGenerator-{int(time.time())}"
        prompt_description = "Playlist generator"
        prompt_template = """
          Make me a {{genre}} playlist consisting of the following number of songs: {{number}}."""

        create_response = create_prompt(
            bedrock_client, prompt_name, prompt_description, prompt_template, model_id
        )

        prompt_id = create_response["id"]
        print(f"Created prompt: {prompt_name} with ID: {prompt_id}")

        # Create a version of the prompt
        print("\n=== Creating a version of the prompt ===")
        version_response = create_prompt_version(
            bedrock_client,
            prompt_id,
            description="Initial version of the product description generator",
        )

        prompt_version_arn = version_response["arn"]
        prompt_version = version_response["version"]

        print(f"Created prompt version: {prompt_version}")
        print(f"Prompt version ARN: {prompt_version_arn}")

        print("\n=== Step 2: Invoking the prompt ===")
        input_variables = {
            "genre": "pop",
            "number": "2",
        }

        result = invoke_prompt(
            bedrock_runtime_client, prompt_version_arn, input_variables
        )
        print(f"\n{result}")

        if cleanup:
            print("\n=== Step 3: Cleaning up resources ===")

            print(f"Deleting prompt {prompt_id}...")
            delete_prompt(bedrock_client, prompt_id)

            print("Cleanup complete")
        else:
            print("\n=== Resources were not cleaned up ===")
            print(f"Prompt ID: {prompt_id}")

    except Exception as e:
        logger.exception("Error in scenario: %s", str(e))

        if cleanup and prompt_id:
            try:
                print("\nCleaning up resources after error...")

                try:
                    delete_prompt(bedrock_client, prompt_id)
                    print("Cleanup after error complete")
                except Exception as cleanup_error:
                    logger.error("Error during cleanup: %s", str(cleanup_error))
            except Exception as final_error:
                logger.error("Final error during cleanup: %s", str(final_error))

        raise


def main():
    print()
    parser = argparse.ArgumentParser(
        description="Run the Amazon Bedrock managed prompt scenario."
    )
    parser.add_argument("--region", default="us-east-1", help="The AWS Region to use.")
    parser.add_argument(
        "--model-id",
        default="amazon.nova-lite-v1:0",
        help="The model ID to use for the prompt.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Clean up resources at the end of the scenario.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_false",
        dest="cleanup",
        help="Don't clean up resources at the end of the scenario.",
    )
    args = parser.parse_args()

    bedrock_client = boto3.client("bedrock-agent", region_name=args.region)
    bedrock_runtime_client = boto3.client("bedrock-runtime", region_name=args.region)

    print("=== Amazon Bedrock Managed Prompt Scenario ===")
    print(f"Region: {args.region}")
    print(f"Model ID: {args.model_id}")
    print(f"Cleanup resources: {args.cleanup}")

    try:
        run_scenario(
            bedrock_client, bedrock_runtime_client, args.model_id, args.cleanup
        )

    except Exception as e:
        logger.exception("Error running scenario: %s", str(e))


if __name__ == "__main__":
    main()
