## Name:Nivetha N
## Reg.No:212225040290
# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data 
2. Split Dataset into Training and Testing Sets
3. Train the Model Using Stochastic Gradient Descent (SGD)
4. Make Predictions and Evaluate Accuracy
5.Generate Confusion Matrix
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Rrenita Ireen G
RegisterNumber:  212225220083
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Sample dataset
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1],  # 0 = No, 1 = Yes
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]    # Target: 0 = Not Placed, 1 = Placed
}

df = pd.DataFrame(data)

# Split into features and target
X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train SGDClassifier for Logistic Regression
sgd_model = SGDClassifier(loss='log_loss',       # 'log' loss → logistic regression
                          max_iter=1000,
                          learning_rate='optimal',
                          random_state=42)
sgd_model.fit(X_train, y_train)

# Make predictions
y_pred = sgd_model.predict(X_test)
y_prob = sgd_model.predict_proba(X_test)   # Probability of placement

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict placement for a new student
new_student = np.array([[6, 68, 1]])  # Example: 6 hours, 68 prev score, Internship yes
new_student_scaled = scaler.transform(new_student)
placement_pred = sgd_model.predict(new_student_scaled)
placement_prob = sgd_model.predict_proba(new_student_scaled)

print(f"\nPredicted Placement Status: {'Placed' if placement_pred[0]==1 else 'Not Placed'}")
print(f"Probability of Placement: {placement_prob[0][1]:.2f}")
```

## Output:
<img width="680" height="411" alt="image" src="https://github.com/user-attachments/assets/0d6b172f-5fda-45ea-b3e2-85ffe82c3c13" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
