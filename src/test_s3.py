import boto3

BUCKET_NAME = "internship-market-analyzer-iw"

s3 = boto3.client("s3")

response = s3.list_objects_v2(
    Bucket=BUCKET_NAME,
    Prefix="analysis/"
)

for item in response.get("Contents", []):
    print(item["Key"])