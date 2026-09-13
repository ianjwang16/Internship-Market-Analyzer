import boto3


BUCKET_NAME = "internship-market-analyzer-iw"


s3 = boto3.client("s3")


files_to_upload = {
    "data/skill_demand.csv":
        "analysis/skill_demand.csv",

    "data/category_skill_demand.csv":
        "analysis/category_skill_demand.csv",

    "data/skill_salary_analysis.csv":
        "analysis/skill_salary_analysis.csv",

    "data/model_top_features.csv":
        "analysis/model_top_features.csv"
}


for local_file, s3_key in files_to_upload.items():

    print(
        f"Uploading {local_file} "
        f"to s3://{BUCKET_NAME}/{s3_key}"
    )

    s3.upload_file(
        local_file,
        BUCKET_NAME,
        s3_key
    )


print("Upload complete.")