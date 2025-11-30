import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ============================================================
# 1. Load Dataset
# ============================================================
df = pd.read_csv("Loanprediction.csv.csv")

# ============================================================
# 2. Clean Missing Values
# ============================================================
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Convert Dependents "3+" → 3
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

# ============================================================
# 3. Encode Target Variable
# ============================================================
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])  # Y=1, N=0

# Save label encoder
joblib.dump(le, "label_encoder.pkl")

# ============================================================
# 4. Train-Test Split
# ============================================================
X = df.drop(columns=['Loan_Status', 'Loan_ID'])  # Remove ID
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 5. Identify Column Types
# ============================================================
numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                    'Loan_Amount_Term', 'Credit_History', 'Dependents']

categorical_features = ['Gender', 'Married', 'Education',
                        'Self_Employed', 'Property_Area']

# ============================================================
# 6. Build Preprocessor
# ============================================================
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ============================================================
# 7. Full Pipeline (Preprocessing + RandomForest)
# ============================================================
model = RandomForestClassifier(n_estimators=200, random_state=42)

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# ============================================================
# 8. Train Model
# ============================================================
pipe.fit(X_train, y_train)

# ============================================================
# 9. Evaluate Model
# ============================================================
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Train accuracy: {pipe.score(X_train, y_train)}")
print(f"Test accuracy: {acc}")

# ============================================================
# 10. Save Pipeline
# ============================================================
joblib.dump(pipe, "pipeline.pkl")
print("pipeline.pkl and label_encoder.pkl saved successfully!")
