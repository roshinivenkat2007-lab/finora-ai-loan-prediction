# 💳 Finora AI – Intelligent Credit Risk Assessment and Smart Loan Recommendation Framework

Finora AI is a Machine Learning based loan assessment system that helps analyze loan applications, predict loan approval status, assess financial risk, and provide smart loan recommendations.

## 🚀 Project Overview

Finora AI uses Machine Learning to evaluate customer financial information such as income, loan amount, CIBIL score, assets, employment status, education, and dependents.

The system provides:

- Loan Approval / Rejection Prediction
- AI-based Risk Assessment
- Risk Score
- Approval Probability
- Financial Health Analysis
- Smart Loan Recommendation
- What-If Analysis
- Explainable AI insights
- Prediction History
- Feature Importance
- Model Performance Analysis
- Interactive Financial Analytics

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- HTML & CSS

## 🤖 Machine Learning Model

The project uses a **Random Forest Classifier** for loan approval prediction.

### Dataset

The model is trained using a loan approval dataset containing financial and personal information of loan applicants.

Important features include:

- Number of Dependents
- Education
- Self Employed
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets
- Commercial Assets
- Luxury Assets
- Bank Assets

## 📊 Model Performance

The project includes model evaluation using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Feature Importance

The dashboard also provides visual analysis of the model and dataset.

> Note: The displayed performance metrics are based on the dataset evaluation implemented in the project.

## ✨ Key Features

### 1. Loan Prediction
Users can enter applicant information and receive a loan approval prediction.

### 2. Risk Assessment
The system calculates a financial risk score and classifies the application into:

- Low Risk
- Medium Risk
- High Risk

### 3. Smart Loan Recommendation
Finora AI provides financial insights and recommendations based on the applicant's profile.

### 4. What-If Analysis
Users can change values such as:

- CIBIL Score
- Annual Income
- Loan Amount
- Loan Tenure

and observe how the financial assessment changes.

### 5. Explainable AI
The system highlights positive and risk factors influencing the financial assessment.

### 6. Prediction History
Previous loan predictions are stored during the current application session.

### 7. Financial Analytics
Interactive charts are provided for:

- Income vs Loan Amount
- Asset Distribution
- EMI vs Monthly Income
- Risk and Approval Analysis

## 📁 Project Structure

```text
finora-ai-loan-prediction/
│
├── app.py
├── loan_model.pkl
├── loan_approval_dataset (1).csv
├── model_performance.py
├── ui_style.css
├── finora_logo.png.png
└── feature_importance.png
