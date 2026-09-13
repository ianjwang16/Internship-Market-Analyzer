import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


# Load dataset
df = pd.read_csv(
    "data/internships_newgrad_10k.csv"
)


# Keep internships
internships = df[
    df["job_type"] == "Internship"
].copy()


# Categories we want to predict
categories = [
    "Software Engineering",
    "Data Science & Analytics",
    "Machine Learning & AI"
]


jobs = internships[
    internships["category"].isin(categories)
].copy()


# Remove rows without descriptions
jobs = jobs.dropna(
    subset=["description", "category"]
)


print("=== DATASET ===")

print(
    jobs["category"].value_counts()
)


# Features
X = (
    jobs["title"].fillna("")
    + " "
    + jobs["description"].fillna("")
)


# Labels
y = jobs["category"]


# Split training/testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(
    "\nTraining jobs:",
    len(X_train)
)

print(
    "Testing jobs:",
    len(X_test)
)


# Create ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# Train model
model.fit(
    X_train,
    y_train
)


# Predict test data
predictions = model.predict(
    X_test
)


# Evaluate accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    f"\nAccuracy: {accuracy:.2%}"
)


# Detailed evaluation
print(
    "\n=== CLASSIFICATION REPORT ==="
)

print(
    classification_report(
        y_test,
        predictions
    )
)


# Test custom job description
test_description = """
We are seeking an intern with experience in
Python, pandas, SQL, statistics, data visualization,
and analyzing large datasets.
"""


custom_prediction = model.predict(
    [test_description]
)


print(
    "\nCustom prediction:",
    custom_prediction[0]
)


# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    xticks_rotation=20
)

plt.title(
    "Job Category Classification"
)

plt.tight_layout()

plt.show()