import boto3
import pandas as pd
from io import BytesIO


BUCKET_NAME = "internship-market-analyzer-iw"


s3 = boto3.client("s3")


response = s3.get_object(
    Bucket=BUCKET_NAME,
    Key="analysis/skill_demand.csv"
)


data = response["Body"].read()


df = pd.read_csv(
    BytesIO(data)
)


print(df.head())