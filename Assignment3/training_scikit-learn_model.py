import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from datasets import load_dataset
from icecream import ic
import seaborn as sns
import pandas as pd

dataset = load_dataset("katossky/wine-recognition")
df = pd.DataFrame(dataset['train'])
print(df.head())

X = df.drop(columns=["label"])
y = df["label"].values


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=12)

#model = SVC(kernel='rbf', gamma='scale')
model = RandomForestClassifier(max_depth=2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 5))
plt.barh(features, importances, color="green")
plt.xlabel("Importance Score")
plt.title("Feature Importance (Random Forest)")
plt.show()

cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix using pandas
cm_df = pd.DataFrame(cm, index=["Actual 1", "Actual 2", "Actual 3"], columns=["Predicted 1", "Predicted 2", "Predicted 3"])
plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Plot data points
