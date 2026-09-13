# Internship Market Analyzer

A Python data science and machine learning application that analyzes internship job postings to identify in-demand technical skills, compare salary trends, and classify internship roles.

The project combines data analysis, machine learning, cloud storage, and an interactive Streamlit dashboard.

**Live Demo:** (https://internship-market-analyzer.streamlit.app)

## Dashboard Screenshots

### Skill Demand
![Skill Demand](screenshots/skill-analysis.png)

### Career Comparison
![Career Comparison](screenshots/career-comparison.png)

### Salary Analysis
![Salary Analysis](screenshots/salary-analysis.png)

### Job Category Predictor
![Job Predictor](screenshots/predictor.png)

## Features

* Analyze the most frequently requested technical skills in internship postings
* Compare skill demand across Software Engineering, Data Science, and Machine Learning roles
* Analyze median hourly salary by technical skill
* Classify internship descriptions using machine learning
* Display prediction confidence for each job category
* Store and retrieve processed analysis data using Amazon S3
* Authenticate cloud access using AWS IAM permissions
* Explore results through an interactive Streamlit dashboard

## Tech Stack

### Programming and Data Science

* Python
* Pandas
* Matplotlib
* scikit-learn
* TF-IDF
* Logistic Regression
* Joblib

### Web Application

* Streamlit

### Cloud

* AWS S3
* AWS IAM
* boto3
* Streamlit Community Cloud

### Development Tools

* Git
* GitHub
* Python virtual environments

## Project Architecture

```text
Raw Internship Dataset
        |
        v
Python / Pandas
        |
        v
Data Cleaning & Filtering
        |
        +--------------------------+
        |                          |
        v                          v
Skill Extraction             Salary Analysis
        |                          |
        +-------------+------------+
                      |
                      v
                Analysis CSVs
                      |
                      v
                  Amazon S3
                      |
                    boto3
                      |
                      v
             Streamlit Dashboard
                      |
        +-------------+-------------+
        |                           |
        v                           v
Market Visualizations      ML Job Classifier


Job Title + Description
        |
        v
      TF-IDF
        |
        v
Logistic Regression
        |
        v
Job Category Prediction
```

## Data Analysis

The application analyzes internship job postings across three main career areas:

* Software Engineering
* Data Science & Analytics
* Machine Learning & AI

Technical skills are extracted from job descriptions using regular-expression pattern matching.

Examples include:

* Python
* Java
* C++
* JavaScript
* SQL
* AWS
* Azure
* Git
* Docker
* Kubernetes
* React
* Pandas
* NumPy
* TensorFlow
* PyTorch

The application calculates both the number and percentage of internship postings mentioning each skill.

## Career Comparison

Skill demand is compared across internship categories.

For example, the application can compare how frequently Python appears in:

* Software Engineering internships
* Data Science internships
* Machine Learning and AI internships

Users can select individual skills from the Streamlit dashboard and view category-specific demand.

## Salary Analysis

For postings with usable hourly salary information, the project:

1. Extracts minimum and maximum salary values
2. Calculates the midpoint of each advertised salary range
3. Removes unrealistic salary observations
4. Compares median hourly salary across technical skills
5. Uses minimum sample-size requirements to reduce misleading comparisons

Salary differences represent associations in the dataset and should not be interpreted as causal relationships.

## Machine Learning

The project uses supervised machine learning to classify internship job postings into:

* Software Engineering
* Data Science & Analytics
* Machine Learning & AI

Job titles and descriptions are converted into numerical features using **TF-IDF**.

A multiclass **Logistic Regression** classifier is trained using an 80/20 stratified train-test split.

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 score
* Confusion matrix

Current test accuracy:

**75.91%**

The trained model is saved using Joblib so the Streamlit application can make predictions without retraining the model for every request.

## Model Interpretation

The application also examines the Logistic Regression coefficients to identify which words and phrases most strongly influence predictions for each job category.

This makes it possible to inspect what the classifier learned instead of treating the model as a complete black box.

Examples of potentially important features include terms related to:

* backend development
* data analytics
* machine learning
* statistics
* neural networks
* APIs
* software development

The strongest learned features are exported to:

```text
data/model_top_features.csv
```

## AWS Integration

Processed analysis results are stored in **Amazon S3** and retrieved by the application using **boto3**, the AWS SDK for Python.

The cloud workflow is:

```text
Python Analysis
      |
      v
Generated CSV Files
      |
      v
Amazon S3
      |
    boto3
      |
      v
Pandas DataFrames
      |
      v
Streamlit Dashboard
```

The S3 bucket stores analysis outputs such as:

```text
analysis/
|-- skill_demand.csv
|-- category_skill_demand.csv
|-- skill_salary_analysis.csv
`-- model_top_features.csv
```

AWS IAM is used to restrict the application's permissions to only the required S3 bucket.

The project uses permissions such as:

* `s3:ListBucket`
* `s3:GetObject`
* `s3:PutObject`

AWS credentials are not stored in the source code or GitHub repository.

Local development uses the standard AWS credential configuration, while the deployed Streamlit application uses secure environment variables / Streamlit secrets.

## Interactive Dashboard

The Streamlit application contains four main tabs.

### Skill Demand

Displays the most frequently requested technical skills across internship postings.

### Career Comparison

Allows users to select a technology and compare demand across different internship categories.

### Salary Analysis

Displays median hourly salaries associated with technical skills.

### Job Category Predictor

Allows users to paste an internship job description and receive:

* Predicted career category
* Confidence score for each category
* Interactive probability visualization

## Project Structure

```text
Internship-Market-Analyzer/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- data/
|   |-- skill_demand.csv
|   |-- category_skill_demand.csv
|   |-- skill_salary_analysis.csv
|   `-- model_top_features.csv
|
|-- models/
|   `-- job_classifier_model.pkl
|
|-- src/
|   |-- analyze.py
|   |-- skill_analysis.py
|   |-- clean_data.py
|   |-- real_data_analysis.py
|   |-- category_analysis.py
|   |-- salary_analysis.py
|   |-- job_classifier.py
|   |-- model_analysis.py
|   |-- predict_job.py
|   |-- s3_storage.py
|   |-- s3_download.py
|   |-- read_s3_data.py
|   `-- test_s3.py
|
`-- screenshots/
    |-- skill-analysis.png
    |-- career-comparison.png
    |-- salary-analysis.png
    `-- predictor.png
```

## Running the Project Locally

### 1. Clone the repository

```bash
git clone YOUR-GITHUB-REPOSITORY-URL
cd Internship-Market-Analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure AWS credentials

Install and configure the AWS CLI:

```bash
aws configure
```

The configured IAM user must have permission to read the project's S3 analysis files.

### 5. Train the machine-learning model

```bash
python src/model_analysis.py
```

This creates:

```text
models/job_classifier_model.pkl
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Security

AWS credentials and other secrets are never hard-coded into the Python source code.

The project uses:

* AWS IAM permissions
* AWS CLI credential configuration for local development
* Streamlit secrets/environment variables for cloud deployment
* `.gitignore` to prevent sensitive files from being committed

## What I Learned

This project provided hands-on experience with:

* Python data analysis
* Pandas DataFrames
* Cleaning real-world datasets
* Exploratory data analysis
* Regular-expression based text extraction
* Data visualization
* TF-IDF feature engineering
* Supervised machine learning
* Multiclass classification
* Logistic Regression
* Train/test splitting
* Accuracy, precision, recall, and F1 score
* Confusion matrices
* Model interpretation
* Model persistence with Joblib
* Interactive web applications with Streamlit
* AWS S3 cloud storage
* AWS IAM permissions
* boto3 integration
* Secure credential management
* Git and GitHub
* Cloud deployment

## Future Improvements

Potential future additions include:

* Resume-to-job skill matching
* Missing-skill recommendations
* Additional internship categories
* More advanced NLP models
* Automatic ingestion of newer job-posting datasets
* Additional AWS services
* Historical skill-demand tracking
* Location-based internship analysis

