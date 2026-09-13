import pandas as pd
import matplotlib.pyplot as plt


# Load real dataset
df = pd.read_csv(
    "data/internships_newgrad_10k.csv"
)


# Keep internships
internships = df[
    df["job_type"] == "Internship"
].copy()


# Main tech categories
categories = [
    "Software Engineering",
    "Data Science & Analytics",
    "Machine Learning & AI"
]


tech_jobs = internships[
    internships["category"]
    .isin(categories)
].copy()


# Keep jobs with salary information
salary_jobs = tech_jobs.dropna(
    subset=[
        "salary_min",
        "salary_max",
        "salary_type"
    ]
).copy()


# Normalize salary type
salary_jobs["salary_type_clean"] = (
    salary_jobs["salary_type"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# Keep hourly salaries
hourly_jobs = salary_jobs[
    salary_jobs["salary_type_clean"]
    .str.contains(
        "hour",
        case=False,
        na=False
    )
].copy()


# Calculate salary midpoint
hourly_jobs["salary_mid"] = (
    hourly_jobs["salary_min"]
    + hourly_jobs["salary_max"]
) / 2


# Remove unrealistic hourly salaries
hourly_jobs = hourly_jobs[
    hourly_jobs["salary_mid"]
    .between(10, 200)
].copy()


print(
    "Hourly salary jobs:",
    len(hourly_jobs)
)


print(
    "Overall median salary:",
    f'${hourly_jobs["salary_mid"].median():.2f}'
)


# Skills to analyze
skill_patterns = {
    "Python": r"\bpython\b",
    "Java": r"\bjava\b",
    "JavaScript": r"\bjavascript\b",
    "C++": r"(?<!\w)c\+\+(?!\w)",
    "SQL": r"\bsql\b",
    "AWS": r"\baws\b",
    "Azure": r"\bazure\b",
    "Git": r"\bgit\b",
    "Docker": r"\bdocker\b",
    "Kubernetes": r"\bkubernetes\b",
    "React": r"\breact\b",
    "Pandas": r"\bpandas\b",
    "PyTorch": r"\bpytorch\b"
}


results = []


for skill, pattern in skill_patterns.items():

    skill_mask = (
        hourly_jobs["description"]
        .str.contains(
            pattern,
            case=False,
            na=False,
            regex=True
        )
    )

    skill_jobs = hourly_jobs[
        skill_mask
    ]

    if len(skill_jobs) == 0:
        continue

    results.append({
        "skill": skill,
        "job_count": len(skill_jobs),

        "median_salary":
            skill_jobs["salary_mid"]
            .median(),

        "mean_salary":
            skill_jobs["salary_mid"]
            .mean()
    })


# Create DataFrame
salary_results = pd.DataFrame(
    results
)


# Require reasonable sample size
salary_results = salary_results[
    salary_results["job_count"] >= 10
]


# Sort
salary_results = (
    salary_results
    .sort_values(
        "median_salary",
        ascending=False
    )
)


print(
    "\n=== SALARY BY SKILL ==="
)

print(
    salary_results.round(2)
)


# Save results
salary_results.to_csv(
    "data/skill_salary_analysis.csv",
    index=False
)


# Prepare chart
chart_data = (
    salary_results
    .sort_values("median_salary")
    .reset_index(drop=True)
)


plt.barh(
    chart_data["skill"],
    chart_data["median_salary"]
)


# Show sample sizes
for index, row in chart_data.iterrows():

    plt.text(
        row["median_salary"] + 0.5,
        index,
        f'n={row["job_count"]}',
        va="center"
    )


plt.title(
    "Median Hourly Internship Salary by Skill"
)

plt.xlabel(
    "Median Hourly Salary ($)"
)

plt.ylabel("Skill")

plt.tight_layout()

plt.show()