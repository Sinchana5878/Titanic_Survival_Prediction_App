# Titanic Survival Prediction Project - Local Version for VS Code

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

print("🚀 Starting Titanic model training...")

# Check if the dataset exists
if not os.path.exists('train.csv'):
    print("❌ ERROR: 'train.csv' not found! Please place the dataset in this folder.")
    exit()

# Load the dataset
titanic_data = pd.read_csv('train.csv')

# Check basic info
print("✅ Data loaded successfully!")
print(f"📊 Dataset shape: {titanic_data.shape}")
print()

# Data preprocessing
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Encode categorical columns
titanic_data.replace({
    'Sex': {'male': 0, 'female': 1},
    'Embarked': {'S': 0, 'C': 1, 'Q': 2}
}, inplace=True)

# Separate features and target
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
print("🧠 Training Logistic Regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

# Evaluate accuracy
train_acc = accuracy_score(Y_train, model.predict(X_train))
test_acc = accuracy_score(Y_test, model.predict(X_test))

print(f"✅ Training Accuracy: {train_acc:.2f}")
print(f"✅ Testing Accuracy: {test_acc:.2f}")

# Save the trained model as a pickle file
with open('titanic_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("🎉 Model saved successfully as 'titanic_model.pkl' in your project folder!")
print("✅ You can now use this model in your Streamlit app (app.py)")
