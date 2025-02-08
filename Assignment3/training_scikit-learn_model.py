import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
from icecream import ic
import pandas as pd

dataset = load_dataset("alfredodeza/wine-ratings", streaming=False, trust_remote_code=True)

df = pd.DataFrame(dataset['train'])
df["year"] = df["name"].str.extract(r"(\d{4})$")
df["year"] = pd.to_numeric(df["year"])
df["year"] = df["year"].fillna(int(df["year"].mean()))
df['name'] = df['name'].fillna("")
df['notes'] = df['notes'].fillna("")
df['region'] = df['region'].fillna("")
print(df[df["name"] == "Avignonesi Vin Santo Occhio di Pernice (375ML half-bottle) 1999"])
print("\n")
print(df.head())


X = df.drop(columns=['rating'])
y = df['rating']
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
preprocess = ColumnTransformer([
    ('name_process', TfidfVectorizer(), 'name'),
    ('notes_process', TfidfVectorizer(), 'notes'),
    ('region_process', TfidfVectorizer(), 'region'),
    ('year', StandardScaler(), ['year']),
    ('variety', OneHotEncoder(handle_unknown="ignore"), ['variety']),
], remainder='passthrough')


pipeline = Pipeline([
    ("preprocessor", preprocess),
    ('regressor', LinearRegression())
])



pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

y_pred = np.round(y_pred)
ic(y_pred)
ic(type(y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
