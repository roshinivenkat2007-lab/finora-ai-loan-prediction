import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("loan_approval_dataset (1).csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# =========================================================
# CLEAN DATA
# =========================================================

# Remove rows with missing values
df = df.dropna()

# =========================================================
# ENCODE CATEGORICAL FEATURES
# =========================================================

# Education
df["education"] = df["education"].map({
    " Graduate": 0,
    "Graduate": 0,
    " Not Graduate": 1,
    "Not Graduate": 1
})

# Self employed
df["self_employed"] = df["self_employed"].map({
    " No": 0,
    "No": 0,
    " Yes": 1,
    "Yes": 1
})

# =========================================================
# ENCODE TARGET
# =========================================================

df["loan_status"] = df["loan_status"].map({
    " Approved": 0,
    "Approved": 0,
    " Rejected": 1,
    "Rejected": 1
})

# Remove rows where encoding failed
df = df.dropna()

# =========================================================
# SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop("loan_status", axis=1)
y = df["loan_status"].astype(int)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("loan_model.pkl")

print("\nModel loaded successfully!")

# =========================================================
# MATCH MODEL FEATURES
# =========================================================

if hasattr(model, "feature_names_in_"):

    expected_features = list(model.feature_names_in_)

    print("\nFeatures expected by model:")
    print(expected_features)

    # Keep only required model features
    X = X.reindex(
        columns=expected_features,
        fill_value=0
    )

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X)

# =========================================================
# PERFORMANCE METRICS
# =========================================================

accuracy = accuracy_score(y, y_pred)

precision = precision_score(
    y,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    zero_division=0
)

# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n==========================================")
print("       MODEL PERFORMANCE RESULTS")
print("==========================================")

print(f"\nAccuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y, y_pred)

print("\n==========================================")
print("          CONFUSION MATRIX")
print("==========================================")

print(cm)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n==========================================")
print("        CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y,
        y_pred,
        target_names=["Approved", "Rejected"],
        zero_division=0
    )
)

print("\n==========================================")
print("        EVALUATION COMPLETED")
print("==========================================")
import matplotlib.pyplot as plt

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

if hasattr(model, "feature_importances_"):

    importance = model.feature_importances_

    feature_names = X.columns

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n==========================================")
    print("          FEATURE IMPORTANCE")
    print("==========================================")

    print(feature_importance)

    # Plot
    plt.figure(figsize=(10, 6))

    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Loan Approval - Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig("feature_importance.png")

    plt.show()

    print("\nFeature importance chart saved as:")
    print("feature_importance.png")

else:
    print("Model does not support feature importance.")