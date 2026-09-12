import pandas as pd


# Load raw dataset
df = pd.read_csv("data/internships.csv")


print("=== BEFORE CLEANING ===")
print("Rows:", len(df))
print()

print("Missing values:")
print(df.isnull().sum())
print()

print("Duplicate rows:", df.duplicated().sum())
print()


# Remove duplicate rows
df = df.drop_duplicates()


# Fill missing skills
df["skills"] = df["skills"].fillna("Unknown")


# Fill missing salary with median
median_salary = df["salary"].median()

df["salary"] = df["salary"].fillna(median_salary)


# Clean text columns
df["company"] = df["company"].str.strip()
df["title"] = df["title"].str.strip()
df["location"] = df["location"].str.strip()
df["skills"] = df["skills"].str.strip()


# Create normalized company name
df["company_clean"] = (
    df["company"]
    .str.strip()
    .str.lower()
)


# Convert salary back to integer
df["salary"] = df["salary"].round().astype(int)


# Create salary category
def salary_category(salary):

    if salary < 43:
        return "Low"

    elif salary <= 46:
        return "Medium"

    else:
        return "High"


df["salary_category"] = (
    df["salary"]
    .apply(salary_category)
)


print("=== AFTER CLEANING ===")

print("Rows:", len(df))
print()

print("Missing values:")
print(df.isnull().sum())
print()

print("Duplicate rows:", df.duplicated().sum())
print()


print("Salary categories:")
print(df["salary_category"].value_counts())


# Save cleaned dataset
df.to_csv(
    "data/internships_cleaned.csv",
    index=False
)

print()
print("Cleaned dataset saved successfully.")