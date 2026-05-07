# ============================================
# Student Performance Prediction Project
# ============================================

# ===== 1. Imports =====
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns
from ucimlrepo import fetch_ucirepo

# ===== 2. Load Dataset =====
# TODO: Replace this path with your dataset file
# Example: df = pd.read_csv("student-mat.csv", sep=';')
student_performance = fetch_ucirepo(id=320) 

X = student_performance.data.features
y = student_performance.data.targets

df = pd.concat([X, y], axis=1)

print("Dataset shape:", df.shape)
print(df.head())
df.to_csv("student_performance.csv", index=False)

# ===== 3. Target Variable =====
# Predict final grade (G3)
target = "G3"

# OPTIONAL: Prevent data leakage (VERY IMPORTANT)
# Remove G1, G2 if present
leakage_cols = ["G1", "G2"]
df = df.drop(columns=[col for col in leakage_cols if col in df.columns])
df.to_csv("student_performanceEdited.csv", index=False)

# ===== 4. Features & Labels =====
X = df.drop(columns=[target])
y = df[target]

# ===== 5. Identify Feature Types =====
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# ===== 6. Preprocessing =====
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# ===== 7. Train-Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== 8. Models =====
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

# ===== 9. Train & Evaluate =====
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    
    results[name] = {
        "R2": r2,
        "MSE": mse,
        "CV Mean": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "pipeline": pipeline
    }

# ===== 10. Print Results =====
print("\n===== Model Performance =====")
for name, res in results.items():
    print(f"{name}:")
    print(f"  R2: {res['R2']:.4f}")
    print(f"  MSE: {res['MSE']:.4f}")
    print(f"  CV Mean: {res['CV Mean']:.4f}")
    print(f"  CV Std: {res['CV Std']:.4f}")
    print()

# ===== 11. Feature Importance (Random Forest) =====
rf_pipeline = results["Random Forest"]["pipeline"]

# Get feature names after preprocessing
ohe = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
encoded_cat_cols = ohe.get_feature_names_out(categorical_cols)

all_features = np.concatenate([numerical_cols, encoded_cat_cols])

# Extract importance
rf_model = rf_pipeline.named_steps["model"]
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n===== Top 10 Important Features =====")
print(feature_importance_df.head(10))

# ===== 12. Plot Feature Importance =====
plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance_df.head(10)
)
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("FeatureImportance.png")

# ===== 13. Correlation Heatmap (EDA) =====
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("CorrelationHeatmap.png")


#Compare R2 scores of all models
model_names = list(results.keys())
r2_scores = [results[m]["R2"] for m in model_names]

plt.figure(figsize=(8,5))
plt.bar(model_names, r2_scores)
plt.title("Model R² Comparison")
plt.xticks(rotation=30)
plt.savefig("model_r2_comparison.png")

#Compare MSE scores of all models
mse_scores = [results[m]["MSE"] for m in model_names]

plt.figure(figsize=(8,5))
plt.bar(model_names, mse_scores)
plt.title("Model MSE Comparison (lower is better)")
plt.xticks(rotation=30)
plt.savefig("model_mse_comparison.png")

#Compare CV scores of all models
cv_means = [results[m]["CV Mean"] for m in model_names]

plt.figure(figsize=(8,5))
plt.bar(model_names, cv_means)
plt.title("Cross-Validation R² Comparison")
plt.xticks(rotation=30)
plt.savefig("model_cv_comparison.png")