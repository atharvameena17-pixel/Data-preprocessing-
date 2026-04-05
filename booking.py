# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv("hotel_booking.csv")

# Drop duplicates (important)
df = df.drop_duplicates()

# 🚨 REMOVE DATA LEAKAGE HERE
df = df.drop(columns=["reservation_status", "reservation_status_date"])

# ==============================
# 3. TARGET VARIABLE
# ==============================
target = "is_canceled"

X = df.drop(columns=[target])
y = df[target]

# ==============================
# 4. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 5. COLUMN TYPES
# ==============================
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# ==============================
# 6. PREPROCESSING PIPELINE
# ==============================

# Numeric pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# ==============================
# 7. MODEL PIPELINE
# ==============================
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ==============================
# 8. TRAIN
# ==============================
model.fit(X_train, y_train)

# ==============================
# 9. PREDICT
# ==============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==============================
# 10. EVALUATION
# ==============================
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
baseline_results = {
    "Accuracy": accuracy,
    "ROC-AUC": roc_auc
}

print(baseline_results)