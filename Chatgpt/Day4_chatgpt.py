# ===========================================================
# 📅 Day 4: Feature Extraction (BoW + TF-IDF)
# Tasks:
#   1️⃣ Create BoW and TF-IDF from company_profile and compare them.
#   2️⃣ Find top 20 most frequent words in job descriptions using BoW.
# ===========================================================

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter

# -----------------------------------------------------------
# Step 1: Load preprocessed dataset (from Day 3)
# -----------------------------------------------------------
df = pd.read_csv('fake_job_postings.csv')

# Fill missing values for both columns to avoid NaN issues
df['company_profile'] = df['company_profile'].fillna('')
df['description'] = df['description'].fillna('')

# ===========================================================
# 🧩 Task 1: BoW and TF-IDF on company_profile
# ===========================================================

# -----------------------
# Bag-of-Words (BoW)
# -----------------------
bow_vectorizer = CountVectorizer(max_features=2000)  # keep top 2000 frequent words
X_bow = bow_vectorizer.fit_transform(df['company_profile'])

print("📦 BoW Shape (company_profile):", X_bow.shape)
print("Sample BoW Feature Names:", bow_vectorizer.get_feature_names_out()[:10])

# -----------------------
# TF-IDF
# -----------------------
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf_vectorizer.fit_transform(df['company_profile'])

print("\n📊 TF-IDF Shape (company_profile):", X_tfidf.shape)
print("Sample TF-IDF Feature Names:", tfidf_vectorizer.get_feature_names_out()[:10])

# -----------------------
# Discussion / Notes
# -----------------------
"""
📘 BoW (Bag of Words):
    - Counts how many times each word appears in a text.
    - Simple frequency representation.
    - Example: if 'innovation' appears 3 times → value = 3
    - Does NOT consider importance across all documents.

📘 TF-IDF (Term Frequency–Inverse Document Frequency):
    - Gives higher weight to rare but meaningful words.
    - Common words like 'company', 'team' get lower weight.
    - Better for capturing semantic importance.

✅ TF-IDF captures meaning better because it highlights unique and relevant terms.
"""

# ===========================================================
# 🧩 Task 2: Top 20 Most Frequent Words in Job Descriptions
# ===========================================================

# Create BoW for job descriptions
bow_vectorizer_desc = CountVectorizer(max_features=5000)  # larger vocabulary for better word coverage
X_bow_desc = bow_vectorizer_desc.fit_transform(df['description'])

# Sum frequencies across all documents
word_freq = X_bow_desc.sum(axis=0)
word_freq = [(word, word_freq[0, idx]) for word, idx in bow_vectorizer_desc.vocabulary_.items()]

# Sort and get top 20 most frequent words
sorted_words = sorted(word_freq, key=lambda x: x[1], reverse=True)[:20]

print("\n🔥 Top 20 Most Frequent Words in Job Descriptions:")
for word, freq in sorted_words:
    print(f"{word:<15} : {int(freq)}")

# -----------------------
# Discussion / Notes
# -----------------------
"""
🧾 Observations:
    - Most frequent words reflect job posting patterns (e.g., 'job', 'experience', 'team', 'skills').
    - These words highlight what employers emphasize most.

💡 BoW Shape:
    (Number of job postings, Number of selected features)
💡 TF-IDF Shape:
    Same shape but with weighted values (float numbers).

✅ Key Takeaway:
    - BoW is simpler and good for frequency analysis.
    - TF-IDF is more informative for text classification or meaning-based models.
"""
