import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Internship Market Analyzer",
    layout="wide"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

skill_data = pd.read_csv(
    "data/skill_demand.csv"
)

salary_data = pd.read_csv(
    "data/skill_salary_analysis.csv"
)

category_data = pd.read_csv(
    "data/category_skill_demand.csv"
)


@st.cache_resource
def load_model():

    return joblib.load(
        "models/job_classifier_model.pkl"
    )


model = load_model()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title(
    "Internship Market Analyzer"
)

st.sidebar.write(
    "Built with Python, Pandas, "
    "scikit-learn, and Streamlit."
)

st.sidebar.markdown(
    """
    **Features**

    - Internship skill analysis
    - Career category comparison
    - Salary analysis
    - ML job classification
    """
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title(
    "Internship Market Analyzer"
)

st.write(
    "Explore technical skill demand, "
    "salary trends, and internship "
    "career categories."
)


# --------------------------------------------------
# Overview
# --------------------------------------------------

top_skill = skill_data.iloc[0]


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Top Skill",
        top_skill["skill"]
    )


with col2:

    st.metric(
        "Job Postings",
        int(top_skill["job_count"])
    )


with col3:

    st.metric(
        "Demand",
        f'{top_skill["percentage"]:.1f}%'
    )


# --------------------------------------------------
# Skill demand
# --------------------------------------------------

st.header(
    "Most Requested Skills"
)


top_skills = skill_data.head(10)


st.bar_chart(
    top_skills,
    x="skill",
    y="job_count"
)


st.dataframe(
    top_skills,
    width="stretch"
)


# --------------------------------------------------
# Category comparison
# --------------------------------------------------

st.header(
    "Skill Demand by Career Area"
)


selected_skill = st.selectbox(
    "Select a skill",
    sorted(
        category_data[
            "skill"
        ].unique()
    )
)


selected_data = category_data[
    category_data["skill"]
    == selected_skill
]


st.bar_chart(
    selected_data,
    x="category",
    y="percentage"
)


# --------------------------------------------------
# Salary analysis
# --------------------------------------------------

st.header(
    "Salary Trends by Skill"
)


st.write(
    "Median advertised hourly salary "
    "for internships mentioning each skill."
)


st.bar_chart(
    salary_data,
    x="skill",
    y="median_salary"
)


st.dataframe(
    salary_data[
        [
            "skill",
            "job_count",
            "median_salary"
        ]
    ],
    width="stretch"
)


st.caption(
    "Salary results include only postings "
    "with usable hourly salary information. "
    "These results show associations in the "
    "dataset and do not imply causation."
)


# --------------------------------------------------
# Machine learning prediction
# --------------------------------------------------

st.header(
    "Job Category Predictor"
)


st.write(
    "Paste an internship job description "
    "below. The machine-learning model "
    "will predict the most likely category."
)


job_description = st.text_area(
    "Job Description",
    height=200
)


if st.button(
    "Predict Category"
):

    if not job_description.strip():

        st.warning(
            "Please enter a job description."
        )

    else:

        prediction = model.predict(
            [job_description]
        )[0]


        probabilities = model.predict_proba(
            [job_description]
        )[0]


        st.success(
            f"Predicted Category: {prediction}"
        )


        probability_df = pd.DataFrame({
            "Category":
                model.classes_,

            "Confidence":
                probabilities * 100
        })


        probability_df = (
            probability_df
            .sort_values(
                "Confidence",
                ascending=False
            )
        )


        st.bar_chart(
            probability_df,
            x="Category",
            y="Confidence"
        )


        st.dataframe(
            probability_df,
            width="stretch"
        )