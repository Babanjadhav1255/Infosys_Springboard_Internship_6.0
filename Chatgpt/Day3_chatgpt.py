# Day 3: Text Cleaning and Preprocessing

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# ==============================
# Load dataset
# ==============================
df = pd.read_csv('fake_job_postings.csv')

# ==============================
# Task 1: Define text cleaning function
# ==============================
def clean_text(text):
    """Cleans text by removing HTML, URLs, punctuation, numbers, stopwords, and lemmatizing words."""
    if pd.isnull(text):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 4. Remove punctuation and numbers
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)
    
    # 5. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    
    return " ".join(words)

# Apply cleaning to description column
df['clean_description'] = df['description'].apply(clean_text)

# ==============================
# Show before and after example
# ==============================
print("Original Text:\n", df['description'].iloc[1][:300])
print("\nCleaned Text:\n", df['clean_description'].iloc[1][:300])

print("\nExample of Cleaned Data:")
print(df[['description', 'clean_description']].head(3))

# ==============================
# Task 2: Count average number of words before and after cleaning
# ==============================

def word_count(text):
    """Counts the number of words in a given text."""
    if pd.isnull(text):
        return 0
    return len(text.split())

# Calculate average word counts
avg_before = df['description'].apply(word_count).mean()
avg_after = df['clean_description'].apply(word_count).mean()

print("\nAverage words before cleaning:", round(avg_before, 2))
print("Average words after cleaning:", round(avg_after, 2))

# ==============================
# Visualization (optional)
# ==============================
plt.figure(figsize=(8,5))
plt.hist(df['description'].apply(word_count), bins=30, alpha=0.6, label='Before Cleaning')
plt.hist(df['clean_description'].apply(word_count), bins=30, alpha=0.6, label='After Cleaning')
plt.legend()

plt.title('Word Count Before vs After Cleaning')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

# ==============================
# Discussion Output
# ==============================
print("\nDiscussion:")
print("Cleaning usually removes HTML tags, punctuation, and stopwords — which are not useful for meaning.")
print("After cleaning, word count decreases, but text becomes more focused on keywords like 'engineer', 'python', 'data', etc.")
print("If word count dropped moderately (around 30–40%), cleaning is effective and not excessive.")


# --------------------------------------
# Task 1
# Custom cleaning function for company_profile
# --------------------------------------
from nltk.corpus import stopwords

def clean_company_profile(text):
    """
    Removes HTML, numbers, punctuation, and stopwords,
    converts text to lowercase.
    """
    if pd.isnull(text):
        return ""

    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]

    return " ".join(words)

# Apply cleaning to company_profile
df['clean_company_profile'] = df['company_profile'].apply(clean_company_profile)

# Display before and after
print("\n🏢 Original Company Profile:\n", df['company_profile'].iloc[0])
print("\n✅ Cleaned Company Profile:\n", df['clean_company_profile'].iloc[0])



# --------------------------------------
# Task  2
# Count words before and after cleaning for job descriptions
# --------------------------------------
df['desc_word_count_before'] = df['description'].apply(lambda x: len(str(x).split()))
df['desc_word_count_after'] = df['clean_description'].apply(lambda x: len(str(x).split()))

avg_before = df['desc_word_count_before'].mean()
avg_after = df['desc_word_count_after'].mean()

print(f"\n📊 Average words before cleaning: {avg_before:.2f}")
print(f"📉 Average words after cleaning: {avg_after:.2f}")
