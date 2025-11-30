from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load pipeline and label encoder (must exist in project root)
PIPELINE_PATH = "pipeline.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

if not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError(f"{PIPELINE_PATH} not found. Run train_pipeline.py first.")
if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(f"{LABEL_ENCODER_PATH} not found. Run train_pipeline.py first.")

pipe = joblib.load(PIPELINE_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

# Columns expected by the pipeline BEFORE transformation
# (these match the columns used in train_pipeline.py)
EXPECTED_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Dependents",
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area"
]

def parse_form_to_df(form: dict) -> pd.DataFrame:
    """
    Convert incoming form dict to a DataFrame with the expected columns and types.
    This includes light validation & defaulting.
    """
    # Helper to get value from form with fallback
    def get(key, default=""):
        return form.get(key, default).strip()

    # Numeric fields (attempt to convert; fallback to median-like defaults)
    def to_float(val, default=0.0):
        try:
            return float(val)
        except Exception:
            return float(default)

    def to_int(val, default=0):
        try:
            return int(float(val))
        except Exception:
            return int(default)

    # Read inputs
    applicant_income = to_float(get("ApplicantIncome", "0"))
    coapplicant_income = to_float(get("CoapplicantIncome", "0"))
    loan_amount = to_float(get("LoanAmount", "0"))
    loan_term = to_float(get("Loan_Amount_Term", "360"))  # months
    credit_history = get("Credit_History", "")
    # credit_history should be 1 or 0
    if credit_history in ["1", "1.0", "yes", "Yes", "Y", "y", "True", "true"]:
        credit_history = 1.0
    elif credit_history in ["0", "0.0", "no", "No", "N", "n", "False", "false"]:
        credit_history = 0.0
    else:
        # if unspecified, default to 1.0 (most borrowers in your dataset had 1)
        credit_history = 1.0

    dependents = get("Dependents", "0")
    # allow "3+" -> make it 3, and safe int conversion
    dependents = dependents.replace("+", "")
    dependents = to_int(dependents, default=0)

    # Categorical fields: normalize common inputs
    gender = get("Gender", "Male").capitalize()
    if gender not in ["Male", "Female"]:
        # allow numeric 1/0 mapping
        if gender in ["1", "1.0", "true", "True", "m", "M"]:
            gender = "Male"
        else:
            gender = "Female"

    married = get("Married", "Yes").capitalize()
    if married not in ["Yes", "No"]:
        married = "Yes" if married in ["1", "true", "True", "y", "Y"] else "No"

    education = get("Education", "Graduate").capitalize()
    # we'll map 'Graduate'/'Not graduate' — keep case close to training
    if education not in ["Graduate", "Not graduate", "Not Graduate"]:
        # accept 1/0 or '0' as Not Graduate
        if education in ["0", "no", "n", "false", "False"]:
            education = "Not Graduate"
        else:
            education = "Graduate"
    # normalize exact label in training ('Graduate' vs 'Not Graduate')
    if education.lower() == "not graduate" or education.lower() == "not graduate":
        education = "Not Graduate"
    else:
        education = "Graduate"

    self_employed = get("Self_Employed", "No").capitalize()
    if self_employed not in ["Yes", "No"]:
        self_employed = "Yes" if self_employed in ["1", "true", "True", "y", "Y"] else "No"

    property_area = get("Property_Area", "Semiurban").capitalize()
    if property_area not in ["Urban", "Rural", "Semiurban"]:
        # support 'semi-urban', 'semi urban'
        if "semi" in property_area.lower():
            property_area = "Semiurban"
        elif "urb" in property_area.lower():
            property_area = "Urban"
        else:
            property_area = "Rural"

    # Build a single-row dataframe with expected columns
    data = {
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [float(credit_history)],
        "Dependents": [int(dependents)],
        "Gender": [gender],
        "Married": [married],
        "Education": [education],
        "Self_Employed": [self_employed],
        "Property_Area": [property_area]
    }

    df = pd.DataFrame(data)
    # Make sure columns order matches EXPECTED_COLUMNS (not strictly necessary but tidy)
    df = df[EXPECTED_COLUMNS]
    return df

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_form():
    # Parse form into dataframe for pipeline
    input_df = parse_form_to_df(request.form)
    try:
        pred_proba = pipe.predict_proba(input_df)[0, 1]  # probability of class '1' (approved)
        pred_label_num = pipe.predict(input_df)[0]
        pred_label_text = le.inverse_transform([pred_label_num])[0]
    except Exception as e:
        # Return a friendly error if prediction fails
        return render_template("index.html", error=str(e))

    # Build nice output
    result = {
        "prediction_label": pred_label_text,
        "prediction_numeric": int(pred_label_num),
        "probability_approved": float(pred_proba)
    }
    # render result on page
    return render_template("index.html", result=result, inputs=request.form)

@app.route("/predict_api", methods=["POST"])
def predict_api():
    """
    Example JSON request body:
    {
      "ApplicantIncome": 5000,
      "CoapplicantIncome": 0,
      "LoanAmount": 128,
      "Loan_Amount_Term": 360,
      "Credit_History": 1,
      "Dependents": 0,
      "Gender": "Male",
      "Married": "Yes",
      "Education": "Graduate",
      "Self_Employed": "No",
      "Property_Area": "Urban"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "send JSON body"}), 400

    input_df = parse_form_to_df(data)
    try:
        pred_proba = pipe.predict_proba(input_df)[0, 1]
        pred_label_num = pipe.predict(input_df)[0]
        pred_label_text = le.inverse_transform([pred_label_num])[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "prediction_label": pred_label_text,
        "prediction_numeric": int(pred_label_num),
        "probability_approved": float(pred_proba)
    })

if __name__ == "__main__":
    # Run in debug=False for production; set debug=True while developing
    app.run(host="0.0.0.0", port=5000, debug=True)
