
# Student Performance Prediction Project


# imports
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
from ucimlrepo import fetch_ucirepo   #we are using the UCI Machine Learning repository to load dataset

#Note: Please pip install the ucimlrepo.


student_performance = fetch_ucirepo(id=320) 

X = student_performance.data.features
y = student_performance.data.targets

df = pd.concat([X, y], axis=1)

print("Dataset shape:", df.shape)
print(df.head())
df.to_csv("student_performance.csv", index=False)

# The final grade G3 will be the target
target = "G3"

# This is a failsafe to prevent data leaks
leakage_cols = ["G1", "G2"]
df = df.drop(columns=[col for col in leakage_cols if col in df.columns])
df.to_csv("student_performanceEdited.csv", index=False)

# X and Y, features and labels
X = df.drop(columns=[target])
y = df[target]


categorical_cols = X.select_dtypes(include=["object"]).columns #we differentiated the feature types
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing stge, where we use transformers.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Train Test Split, using 42 as the random seed state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# These are the different models we are using: Linear Regression, Ride, Lasso and Random Forest
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

# Here we train the models themselves.
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse= np.sqrt(mse)  #root mean squared error
    
    # Cross validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    
    results[name] = {
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "CV Mean": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "pipeline": pipeline
    }

# Results
print("\n Model Performance Metrics")
for name, resu in results.items():
    print(f"{name}:")
    print(f"  R2: {resu['R2']:.4f}")
    print(f"  MSE: {resu['MSE']:.4f}")
    print(f"  RMSE: {resu['RMSE']:.4f}")
    print(f"  CV Mean: {resu['CV Mean']:.4f}")
    print(f"  CV Std: {resu['CV Std']:.4f}")
    print()

# Feature Importance (Random Forest) 
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

# Plotting Feature Importance here
plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance_df.head(10)
)
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("FeatureImportance.png")

# Correlation Heatmap (EDA)
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("CorrelationHeatmap.png")


#Compare R2 scores of all models
model_names = list(results.keys())
r2_scores = [results[m]["R2"] for m in model_names]
plt.figure(figsize=(8, 6))
r2_bar = sns.barplot(x=model_names, y=r2_scores)

plt.title("Model R² Comparison (Higher is Better)")
plt.ylabel("R² Score")

min_r2 = min(r2_scores)
max_r2 = max(r2_scores)
plt.ylim(min_r2 - 0.01, max_r2 + 0.01)
for container in r2_bar.containers:
    r2_bar.bar_label(container, fmt='%.4f', padding=3)

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_r2_comparison.png")


#Compare RMSE scores of all models
rmse_scores = [results[rm]["RMSE"] for rm in model_names]

plt.figure(figsize=(8, 6))
rmse_bar = sns.barplot(x=model_names, y=rmse_scores)

plt.title("Model RMSE Comparison (Lower is Better)")
plt.ylabel("Error")

min_rmse = min(rmse_scores)
max_rmse = max(rmse_scores)
plt.ylim(min_rmse - 0.1, max_rmse + 0.1)

for container2 in rmse_bar.containers:
    rmse_bar.bar_label(container2, fmt='%.3f', padding=3)

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_rmse_comparison.png")

#Compare CV scores of all models
cv_means = [results[m]["CV Mean"] for m in model_names]

plt.figure(figsize=(8, 6))
CV_bar = sns.barplot(x=model_names, y=cv_means)

plt.title("Cross-Validation R² Comparison (Higher is Better)")
plt.ylabel("Average R² Score")

min_cv = min(cv_means)
max_cv = max(cv_means)

plt.ylim(min_cv - 0.02, max_cv + 0.02)

for container3 in CV_bar.containers:
    CV_bar.bar_label(container3, fmt='%.4f', padding=3)

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_cv_comparison.png")
