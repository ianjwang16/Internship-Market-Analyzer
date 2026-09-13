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


# --------------------------------------------------
# Load machine-learning model
# --------------------------------------------------

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

st.sidebar.metric(
    "Model Accuracy",
    "75.91%"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "About"
)

st.sidebar.write(
    "This project analyzes internship job "
    "postings using Python and machine learning "
    "to identify technology demand, salary trends, "
    "and career categories."
)

st.sidebar.subheader(
    "Technology"
)

st.sidebar.markdown(
    """
    - Python
    - Pandas
    - scikit-learn
    - TF-IDF
    - Logistic Regression
    - Streamlit
    - Matplotlib
    """
)


# --------------------------------------------------
# Main title
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
# Dashboard metrics
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
# Dashboard tabs
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Skill Demand",
    "Career Comparison",
    "Salary Analysis",
    "Job Predictor"
])


# --------------------------------------------------
# Tab 1 - Skill Demand
# --------------------------------------------------

with tab1:

    st.header(
        "Most Requested Skills"
    )

    st.write(
        "The most frequently mentioned "
        "technical skills across tech "
        "internship postings."
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
# Tab 2 - Career Comparison
# --------------------------------------------------

with tab2:

    st.header(
        "Skill Demand by Career Area"
    )

    st.write(
        "Compare how frequently a technical "
        "skill appears across Software Engineering, "
        "Data Science, and Machine Learning roles."
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

    st.dataframe(
        selected_data[
            [
                "category",
                "job_count",
                "percentage"
            ]
        ],
        width="stretch"
    )


# --------------------------------------------------
# Tab 3 - Salary Analysis
# --------------------------------------------------

with tab3:

    st.header(
        "Salary Trends by Skill"
    )

    st.write(
        "Median advertised hourly salary "
        "for internship postings mentioning "
        "each technical skill."
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
# Tab 4 - Job Category Predictor
# --------------------------------------------------

with tab4:

    st.header(
        "Job Category Predictor"
    )

    st.write(
        "Paste an internship job description "
        "below. The machine-learning model "
        "will predict the most likely career category."
    )

    job_description = st.text_area(
        "Job Description",
        height=200,
        placeholder=(
            "Example: We are looking for an intern "
            "with experience in Python, SQL, pandas, "
            "statistics, and data visualization..."
        )
    )

    if st.button(
        "Predict Category"
    ):

        if not job_description.strip():

            st.warning(
                "Please enter a job description."
            )

        else:

            # Predict category
            prediction = model.predict(
                [job_description]
            )[0]

            # Get prediction probabilities
            probabilities = model.predict_proba(
                [job_description]
            )[0]

            st.success(
                f"Predicted Category: {prediction}"
            )

            # Build probability DataFrame
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

            # Display confidence chart
            st.bar_chart(
                probability_df,
                x="Category",
                y="Confidence"
            )

            # Display confidence table
            st.dataframe(
                probability_df,
                width="stretch"
            )


# --------------------------------------------------
# Methodology
# --------------------------------------------------

st.markdown("---")

with st.expander(
    "Methodology"
):

    st.markdown(
        """
        ### Data Processing

        Internship postings were filtered to
        Software Engineering, Data Science,
        and Machine Learning roles.

        ### Skill Extraction

        Technical skills were identified from
        job descriptions using regular-expression
        pattern matching.

        ### Salary Analysis

        Salary comparisons use postings with
        valid hourly salary ranges. The midpoint
        of each advertised salary range was calculated,
        and median salaries were compared across skills.

        ### Machine Learning

        Job titles and descriptions were converted
        into numerical features using TF-IDF.

        A multiclass Logistic Regression model was
        trained to classify internship postings into:

        - Software Engineering
        - Data Science & Analytics
        - Machine Learning & AI

        The model was evaluated using a stratified
        train/test split, accuracy, precision,
        recall, F1 score, and a confusion matrix.
        """
    )