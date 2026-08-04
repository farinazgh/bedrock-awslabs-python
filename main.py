def main():
    print("Hello from bedrock-awslabs-python!")
    import boto3

    sts = boto3.client("sts")
    print(sts.get_caller_identity())


if __name__ == "__main__":
    main()
