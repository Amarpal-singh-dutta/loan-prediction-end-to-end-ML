📌 Loan Prediction System – End-to-End ML Project (Flask Deployment)

This project builds a complete Loan Approval Prediction System using Machine Learning + Flask Web Deployment.
It covers the entire pipeline: Data Cleaning → EDA → Feature Engineering → Model Training → Pipeline Building → Flask UI Deployment.

🚀 Project Overview

The goal of the project is to predict whether a loan application should be Approved (1) or Rejected (0) based on applicant details such as income, loan amount, credit history, employment status, etc.

This project demonstrates:

✔ Real-world ML workflow
✔ Feature engineering
✔ Data preprocessing pipeline
✔ Best model selection (Random Forest – 86% accuracy)
✔ Flask web app for real-time predictions
✔ Full deployment-ready architecture

📂 Project Structure
Loandatsetprediction/
│── train_pipeline.py          # Builds preprocessing + model pipeline and saves pipeline.pkl
│── app.py                    # Flask app backend
│── pipeline.pkl              # Saved ML pipeline (preprocessing + model)
│── label_encoder.pkl         # Encodes Loan_Status
│── Loanprediction.csv.csv    # Dataset
│── templates/
│     └── index.html          # Frontend form for user input
│── README.md                 # Project documentation

📊 Dataset Description

The dataset contains 614 loan applications with the following features:

Feature	Description
ApplicantIncome	Applicant’s income
CoapplicantIncome	Co-applicant’s income
LoanAmount	Loan amount requested
Loan_Amount_Term	Duration of loan
Credit_History	1 = Good, 0 = Bad
Education	Graduate / Not Graduate
Property_Area	Urban / Semiurban / Rural
Dependents	Number of dependents
Loan_Status	Target variable
🧹 Data Cleaning Steps

✔ Missing values handled
✔ Numerical missing → Median imputation
✔ Categorical missing → Mode imputation
✔ Removed duplicates
✔ Converted "3+" → 3 in Dependents
✔ Outliers treated using IQR and Z-score
✔ Data types fixed

🧠 Feature Engineering Performed

New engineered features:

Total_Income = ApplicantIncome + CoapplicantIncome

Income_Loan_Ratio = Total_Income / LoanAmount

Loan_Term_Years = Loan_Amount_Term / 12

EMI = LoanAmount / Loan_Amount_Term

Income_Bin (binned income category)

Encoding applied:

Label Encoding for Loan_Status

Ordinal encoding for Education

One-Hot encoding for Gender, Married, Self_Employed, Property_Area

Scaling:

StandardScaler applied to all numerical features via the pipeline.

📈 Modeling

Trained models:

Logistic Regression

Random Forest (Best)

Decision Tree

SVM

✔ Best Model: Random Forest

Train Accuracy: 100%

Test Accuracy: ~86%

Saved as: pipeline.pkl

🧪 5 Key Insights from EDA

Higher Credit History strongly correlates with loan approval.

Applicants with higher income tend to get approved more often.

Most loans have a term of 360 months, making it the dominant category.

Semiurban applicants had the highest approval rate.

LoanAmount and ApplicantIncome show moderate positive correlation.
🚀 9. Deployment (Flask Web App)

The model was deployed using Flask.



Run Instructions
python train_pipeline.py
python app.py


Then open:

http://127.0.0.1:5000


Web UI accepts user inputs and predicts:
✔ Loan Approved
or
✔ Loan Not Approved

📝 10. Challenges & Learnings

Handling categorical encoding consistency

Managing outliers without harming model performance

Fixing feature mismatch errors in deployment

Building a production-ready ML pipeline with preprocessing

Integrating Python model with Flask interface

🎯 11. Conclusion

Random Forest emerged as the most reliable model for loan approval prediction.
