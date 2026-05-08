import pandas as pd
from ucimlrepo import fetch_ucirepo

# Testing UCI ML Repo Fetching and viewing dataset
student_performance = fetch_ucirepo(id=320) 

X = student_performance.data.features
y = student_performance.data.targets

# Combine into one dataframe
df = pd.concat([X, y], axis=1)

# View it
print(df.head())

df.to_csv("student_performance.csv", index=False)

