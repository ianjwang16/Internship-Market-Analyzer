import pandas as pd
import matplotlib.pyplot as plt


# Load real job dataset
df = pd.read_csv(
    "data/internships_newgrad_10k.csv"
)


print("=== DATASET ===")
print("Total jobs:", len(df))


# Keep only internships
internships = df[
    df["job_type"] == "Internship"
].copy()

print(
    "Internships:",
    len(internships)
)


# Categories relevant to tech
tech_categories = [
    "Software Engineering",
    "Data Science & Analytics",
    "Machine Learning & AI",
    "IT & Networking",
    "Cybersecurity",
    "Cloud & DevOps"
]


# Keep tech internships
tech_internships = internships[
    internships["category"]
    .isin(tech_categories)
].copy()


# Remove rows without descriptions
tech_internships = (
    tech_internships
    .dropna(subset=["description"])
)


print(
    "Tech internships:",
    len(tech_internships)
)


# Skills we want to detect
skill_patterns = {
    "Python": r"\bpython\b",
    "Java": r"\bjava\b",
    "JavaScript": r"\bjavascript\b",
    "C++": r"(?<!\w)c\+\+(?!\w)",
    "C": r"(?<!\w)c(?!\w)",
    "SQL": r"\bsql\b",
    "AWS": r"\baws\b",
    "Azure": r"\bazure\b",
    "Git": r"\bgit\b",
    "Docker": r"\bdocker\b",
    "Kubernetes": r"\bkubernetes\b",
    "React": r"\breact\b",
    "Node.js": r"\bnode\.?js\b",
    "Pandas": r"\bpandas\b",
    "NumPy": r"\bnumpy\b",
    "TensorFlow": r"\btensorflow\b",
    "PyTorch": r"\bpytorch\b"
}


# Count skill mentions
skill_counts = {}

for skill, pattern in skill_patterns.items():

    count = (
        tech_internships["description"]
        .str.contains(
            pattern,
            case=False,
            na=False,
            regex=True
        )
        .sum()
    )

    skill_counts[skill] = count


# Convert dictionary into Pandas Series
skill_counts = pd.Series(
    skill_counts
).sort_values(
    ascending=False
)


# Calculate percentages
skill_percentages = (
    skill_counts
    / len(tech_internships)
) * 100


# Create results DataFrame
skills_df = pd.DataFrame({
    "skill": skill_counts.index,
    "job_count": skill_counts.values,
    "percentage": skill_percentages.values
})


print(
    "\n=== MOST REQUESTED SKILLS ==="
)

print(skills_df)


# Save results
skills_df.to_csv(
    "data/skill_demand.csv",
    index=False
)


# Visualization
top_skills = skills_df.head(12)

plt.barh(
    top_skills["skill"],
    top_skills["job_count"]
)

plt.xlabel(
    "Number of Internship Postings"
)

plt.ylabel("Skill")

plt.title(
    "Most Requested Skills in Tech Internships"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()