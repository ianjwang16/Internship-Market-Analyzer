import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("data/internships.csv")


# Store all individual skills
all_skills = []


# Extract and clean skills
for skill_list in df["skills"]:

    skills = skill_list.split(",")

    for skill in skills:

        cleaned_skill = skill.strip().lower()

        all_skills.append(cleaned_skill)


# Count how often each skill appears
skill_counts = pd.Series(all_skills).value_counts()


# Display results
total_jobs = len(df)

print("=== TOP INTERNSHIP SKILLS ===")

for skill, count in skill_counts.items():

    percentage = (count / total_jobs) * 100

    print(
        f"{skill}: "
        f"{count} jobs "
        f"({percentage:.1f}%)"
    )


# Create visualization
top_skills = skill_counts.head(10).sort_values()

top_skills.plot(kind="barh")

plt.title("Most Requested Internship Skills")
plt.xlabel("Number of Job Postings")
plt.ylabel("Skill")

plt.tight_layout()

plt.show()