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
n_runs = 10
acc_dt_list, acc_bag_list, acc_rf_list = [], [], []

for i in range(n_runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # Train decision tree
    print("Training Decision Tree...")
    tree = DecisionTree(max_depth=None)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_tree)
    acc_dt_list.append(acc_dt)

    # Train bagging classifier
    print("Training Bagging Classifier...")
    bagging = BaggingClassifier(base_estimator=DecisionTree, n_estimators=150, max_depth=None)
    bagging.fit(X_train, y_train)
    y_pred_bagging = bagging.predict(X_test)
    acc_bag = accuracy_score(y_test, y_pred_bagging)
    acc_bag_list.append(acc_bag)

    # Train random forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=None, max_features='sqrt')
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_rf_list.append(acc_rf)

# Evaluate all models
print("Decision Tree Accuracy:", acc_dt)
print("Bagging Accuracy:", acc_bag)
print("Random Forest Accuracy:", acc_rf)

# Compute mean and standard deviation
print(f"Decision Tree Accuracy: {np.mean(acc_dt_list):.3f} ± {np.std(acc_dt_list):.3f}")
print(f"Bagging Accuracy: {np.mean(acc_bag_list):.3f} ± {np.std(acc_bag_list):.3f}")
print(f"Random Forest Accuracy: {np.mean(acc_rf_list):.3f} ± {np.std(acc_rf_list):.3f}")

# Plot a bar chart
models = ['Decision Tree', 'Bagging', 'Random Forest']
accuracy_mean = [np.mean(acc_dt_list), np.mean(acc_bag_list), np.mean(acc_rf_list)]
accuracy_std = [np.std(acc_dt_list), np.std(acc_bag_list), np.std(acc_rf_list)]

plt.bar(models, accuracy_mean, yerr=accuracy_std, capsize=5)
plt.ylabel('Accuracy')
plt.title('Model Comparison on Titanic Dataset')
plt.show()
