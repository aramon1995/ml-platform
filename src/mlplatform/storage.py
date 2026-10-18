import sys
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from mlplatform.config import get_settings

BUCKETS = ("data", "mlflow")


def s3_client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.aws_endpoint_url,
        aws_access_key_id=s.aws_access_key_id,
        aws_secret_access_key=s.aws_secret_access_key,
        region_name="us-east-1",
    )


def create_buckets(retries: int = 30) -> None:
    for attempt in range(retries):
        try:
            client = s3_client()
            existing = {b["Name"] for b in client.list_buckets()["Buckets"]}
            for bucket in set(BUCKETS) - existing:
                client.create_bucket(Bucket=bucket)
            return
        except (BotoCoreError, ClientError):
            if attempt == retries - 1:
                raise
            time.sleep(2)


def upload(path: str, uri: str) -> None:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    s3_client().upload_file(path, bucket, key)


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["create-buckets"]:
            create_buckets()
        case ["upload", path]:
            upload(path, get_settings().data_uri)
        case ["upload", path, uri]:
            upload(path, uri)
        case _:
            sys.exit("usage: python -m mlplatform.storage create-buckets | upload <csv> [s3://bucket/key]")
