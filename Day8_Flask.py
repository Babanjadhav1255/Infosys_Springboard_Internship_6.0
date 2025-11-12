# Day 8: Saving Model and Vectorizer
 
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
 
# Load preprocessed data

df = pd.read_csv('fake_job_postings.csv')

df = df.dropna(subset=['description'])
 
# Train model quickly (TF-IDF + Logistic Regression)

vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform(df['description'])

y = df['fraudulent']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)
 
# Save model and vectorizer

joblib.dump(model, 'fake_job_model.pkl')

joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
 
print("✅ Model and Vectorizer Saved Successfully!")

 
import joblib
 
# Load saved files

model = joblib.load('fake_job_model.pkl')

vectorizer = joblib.load('tfidf_vectorizer.pkl')
 
# Example inference

test_jobs = [

    "Work from home with high pay, no experience required! Apply immediately!",

    "We are hiring a software engineer with 2+ years of Python experience."

]
 
X_test_jobs = vectorizer.transform(test_jobs)

predictions = model.predict(X_test_jobs)
 
for job, pred in zip(test_jobs, predictions):

    label = "Fake Job" if pred == 1 else "Real Job"

    print(f"\nJob: {job}\nPrediction: {label}")

 
# Save as app.py

from flask import Flask, request, jsonify

import joblib
 
app = Flask(__name__)
 
# Load model and vectorizer

model = joblib.load('fake_job_model.pkl')

vectorizer = joblib.load('tfidf_vectorizer.pkl')
 
@app.route('/')

def home():

    return "Fake Job Detection API is running!"
 
@app.route('/predict', methods=['POST'])

def predict():

    data = request.get_json()

    job_text = data.get('description', '')

    if not job_text:

        return jsonify({"error": "No job description provided"}), 400

    # Transform and predict

    X_input = vectorizer.transform([job_text])

    prediction = model.predict(X_input)[0]

    label = "Fake Job" if prediction == 1 else "Real Job"

    return jsonify({

        "prediction": label

    })
 
if __name__ == '__main__':

    app.run(debug=True)

 