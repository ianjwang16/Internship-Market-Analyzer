import pandas as pd


# Load internship dataset
df = pd.read_csv("data/internships.csv")


# Basic dataset information
print("=== INTERNSHIP MARKET ANALYZER ===")
print()

print("Number of internships:", len(df))
print()

print("First 5 jobs:")
print(df.head())
print()


# Salary analysis
average_salary = df["salary"].mean()

print(f"Average hourly salary: ${average_salary:.2f}")


highest_paid_job = df.loc[df["salary"].idxmax()]

print(
    "Highest paying internship:",
    highest_paid_job["company"],
    "-",
    f'${highest_paid_job["salary"]}/hour'
)

print()


# Skill analysis
skills = [
    "Python",
    "Java",
    "C++",
    "JavaScript",
    "SQL",
    "AWS",
    "Git"
]

total_jobs = len(df)

print("=== MOST REQUESTED SKILLS ===")

for skill in skills:

    count = df["skills"].str.contains(
        skill,
        case=False,
        regex=False
    ).sum()

    percentage = (count / total_jobs) * 100

    print(
        f"{skill}: {count} jobs "
        f"({percentage:.1f}%)"
    )