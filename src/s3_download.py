import boto3
import pandas as pd


BUCKET_NAME = "internship-market-analyzer-iw"


s3 = boto3.client("s3")


s3.download_file(
    BUCKET_NAME,
    "analysis/skill_demand.csv",
    "data/skill_demand_from_s3.csv"
)


df = pd.read_csv(
    "data/skill_demand_from_s3.csv"
)


print(df.head())