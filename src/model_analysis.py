import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Load data
df = pd.read_csv(
    "data/internships_newgrad_10k.csv"
)


# Keep internships
internships = df[
    df["job_type"] == "Internship"
].copy()


# Categories
categories = [
    "Software Engineering",
    "Data Science & Analytics",
    "Machine Learning & AI"
]


jobs = internships[
    internships["category"].isin(categories)
].copy()


jobs = jobs.dropna(
    subset=["description", "category"]
)


# Combine title and description
X = (
    jobs["title"].fillna("")
    + " "
    + jobs["description"].fillna("")
)

y = jobs["category"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create model pipeline
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


# Evaluate
predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"Accuracy: {accuracy:.2%}"
)


# Access pipeline components
tfidf = model.named_steps["tfidf"]

classifier = model.named_steps[
    "classifier"
]


# Get feature names
feature_names = (
    tfidf.get_feature_names_out()
)

coefficients = classifier.coef_


# Find strongest features
feature_results = []

top_n = 20


for index, category in enumerate(
    classifier.classes_
):

    category_coefficients = (
        coefficients[index]
    )

    top_indices = (
        category_coefficients
        .argsort()[-top_n:]
        [::-1]
    )

    print(
        f"\n=== {category} ==="
    )

    for feature_index in top_indices:

        feature = (
            feature_names[
                feature_index
            ]
        )

        coefficient = (
            category_coefficients[
                feature_index
            ]
        )

        print(
            f"{feature:30}"
            f"{coefficient:.3f}"
        )

        feature_results.append({
            "category": category,
            "feature": feature,
            "coefficient": coefficient
        })


# Save feature analysis
features_df = pd.DataFrame(
    feature_results
)

features_df.to_csv(
    "data/model_top_features.csv",
    index=False
)


# Save trained model
joblib.dump(
    model,
    "models/job_classifier_model.pkl"
)


print(
    "\nModel saved successfully."
)