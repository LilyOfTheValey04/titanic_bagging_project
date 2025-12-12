import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from src.decision_tree import DecisionTree
from src.bagging import BaggingClassifier
from src.random_forest import RandomForestClassifier

print("Script started...")


# Load dataset
data = pd.read_csv('data/train.csv')

# Features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Encode categorical features
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])

# Prepare X and y
X = data[features].values
y = data[target].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train decision tree
print("Training Decision Tree...")
tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_tree)

# Train bagging classifier
print("Training Bagging Classifier...")
bagging = BaggingClassifier(base_estimator=DecisionTree, n_estimators=20, max_depth=5)
bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
acc_bag = accuracy_score(y_test, y_pred_bagging)

# Train random forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=20, max_depth=5, max_features='sqrt')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Evaluate all models
print("Decision Tree Accuracy:", acc_dt)
print("Bagging Accuracy:", acc_bag)
print("Random Forest Accuracy:", acc_rf)

# Plot a bar chart
models = ['Decision Tree', 'Bagging', 'Random Forest']
accuracy = [accuracy_score(y_test, y_pred_tree),
            accuracy_score(y_test, y_pred_bagging),
            accuracy_score(y_test, y_pred_rf)]

plt.bar(models, accuracy)
plt.ylabel('Accuracy')
plt.title('Model Comparison on Titanic Dataset')
plt.show()