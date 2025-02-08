# import necessary libraries

from datasets import load_dataset
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from icecream import ic
import re
import string
import pandas as pd

dataset = load_dataset('imdb')
df = pd.DataFrame(dataset['train'])
ic(df)

def clean_text(text):
    """
    This function cleans up the text
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

X = df['text'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                    test_size=0.2, random_state=42)

# the model can't handle text, so we have to embed it in text
pipeline = Pipeline([
    # Vectorize or embed the text using TF-IDF
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10_000)),

    # This classifier either classifies the text as neg or pos, so logistic regression binary classifier is used
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def pos_or_neg(text):
    cleaned_text = clean_text(text)
    prediction = pipeline.predict([cleaned_text])
    return "Positive" if prediction[0] == 1 else "Negative"

while True:
    user_input = input("Enter a review (or type 'exit' to quit): ")
    if user_input == 'exit':
        break
    print("Sentiment", pos_or_neg(user_input))
