import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv(
    "data/internships_newgrad_10k.csv"
)


# Keep internships
internships = df[
    df["job_type"] == "Internship"
].copy()


# Categories we want to compare
categories = [
    "Software Engineering",
    "Data Science & Analytics",
    "Machine Learning & AI"
]


# Keep selected categories
tech_jobs = internships[
    internships["category"].isin(categories)
].copy()


# Remove jobs without descriptions
tech_jobs = tech_jobs.dropna(
    subset=["description"]
)


print("=== JOB COUNTS ===")

print(
    tech_jobs
    .groupby("category")
    .size()
)


# Skills to detect
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
    "NumPy": r"\bnumpy\b",
    "TensorFlow": r"\btensorflow\b",
    "PyTorch": r"\bpytorch\b"
}


# Store analysis results
results = []


# Analyze every category
for category in categories:

    category_jobs = tech_jobs[
        tech_jobs["category"] == category
    ]

    total_jobs = len(category_jobs)

    for skill, pattern in skill_patterns.items():

        count = (
            category_jobs["description"]
            .str.contains(
                pattern,
                case=False,
                na=False,
                regex=True
            )
            .sum()
        )

        percentage = (
            count / total_jobs
        ) * 100

        results.append({
            "category": category,
            "skill": skill,
            "job_count": count,
            "percentage": percentage
        })


# Convert results to DataFrame
results_df = pd.DataFrame(results)


print(
    "\n=== CATEGORY SKILL DEMAND ==="
)

print(results_df)


# Save results
results_df.to_csv(
    "data/category_skill_demand.csv",
    index=False
)


# Compare major programming skills
languages = [
    "Python",
    "Java",
    "C++",
    "JavaScript",
    "SQL"
]


language_data = results_df[
    results_df["skill"].isin(languages)
]


# Convert to comparison table
pivot_table = language_data.pivot(
    index="skill",
    columns="category",
    values="percentage"
)


print(
    "\n=== LANGUAGE COMPARISON ==="
)

print(
    pivot_table.round(1)
)


# Visualization
pivot_table.plot(
    kind="bar"
)

plt.title(
    "Programming Skill Demand by Internship Category"
)

plt.xlabel("Skill")

plt.ylabel(
    "Percentage of Job Postings"
)

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()