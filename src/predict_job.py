import joblib


# Load trained model
model = joblib.load(
    "models/job_classifier_model.pkl"
)


# Example description
description = """
Build machine learning models using
Python, PyTorch, computer vision,
neural networks, and deep learning.
"""


# Predict category
prediction = model.predict(
    [description]
)[0]


# Get probabilities
probabilities = model.predict_proba(
    [description]
)[0]


print(
    "Predicted category:",
    prediction
)

print(
    "\nPrediction probabilities:"
)


for category, probability in zip(
    model.classes_,
    probabilities
):

    print(
        f"{category}: "
        f"{probability:.1%}"
    )