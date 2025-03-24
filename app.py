import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\hemas\Desktop\dabeties\diabetes.csv")

# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# MODEL TRAINING
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Check accuracy
accuracy = accuracy_score(y_test, rf.predict(x_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(rf, file)
print("Model saved as 'diabetes_model.pkl'")