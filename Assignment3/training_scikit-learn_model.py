import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from datasets import load_dataset
from icecream import ic
import seaborn as sns
import pandas as pd

def preprocess_data(X):
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    ic(X_scaled)
    return pd.DataFrame(X_scaled, columns=X.columns)

def plot_feature_importances(X, y):
    importances = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(10, 5))
    plt.barh(features, importances, color="green")
    plt.xlabel("Importance Score")
    plt.title("Feature Importance (Random Forest)")
    plt.show()


    features_dict = {features[i]:importances[i] for i in range(0, len(features))}
    top3_features = sorted(features_dict, key=features_dict.get, reverse=True)[:3]
    # Extract corresponding feature values
    X_top3 = X[top3_features]
    ic(X_top3)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using the three most important features
    ax.scatter(X_top3.iloc[:, 0], X_top3.iloc[:, 1], X_top3.iloc[:, 2], c=y, cmap='viridis', edgecolors='black')

    # Label axes with feature names
    ax.set_xlabel(top3_features[0])
    ax.set_ylabel(top3_features[1])
    ax.set_zlabel(top3_features[2])
    ax.set_title("3D Scatter Plot of Top 3 Important Features")
    ax.legend(title="Classes")


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    ic(cm)
    # Display confusion matrix using pandas
    cm_df = pd.DataFrame(cm, index=["Actual 1", "Actual 2", "Actual 3"], columns=["Predicted 1", "Predicted 2", "Predicted 3"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.show()
    for i in range(0,2):
        TP = cm[i][i]
        FP = sum(cm[i])-cm[i][i]

dataset = load_dataset("katossky/wine-recognition")
df = pd.DataFrame(dataset['train'])
print(df.head())

X = df.drop(columns=["label"])
y = df["label"].values


X_scaled = preprocess_data(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.35, random_state=12)

model = RandomForestClassifier(max_depth=2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



plot_feature_importances(X_scaled, y)
plot_confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
