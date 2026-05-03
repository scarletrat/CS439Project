from ucimlrepo import fetch_ucirepo

# Example dataset (you can change ID later)
student_performance = fetch_ucirepo(id=320) 

X = student_performance.data.features
y = student_performance.data.targets

print(X.head())
print(y.head())
  
# metadata 
print(student_performance.metadata) 
  
# variable information 
print(student_performance.variables) 
