# ==============================
# 🧠 Day 2: Understanding the Dataset
# ==============================

# Import required library
import pandas as pd

# --------------------------------------
# Step 1: Load the dataset
# --------------------------------------
# Make sure 'fake_job_postings.csv' is in your current directory.
df = pd.read_csv('fake_job_postings.csv')

# --------------------------------------
# Step 2: Display basic information
# --------------------------------------
print("✅ Sample Data (first 5 rows):")
print(df.head())

# Total number of records
print("\n📊 Total number of records:", len(df))

# --------------------------------------
# Step 3: Check for missing values
# --------------------------------------
print("\n❌ Missing Values per Column:")
print(df.isnull().sum())

# --------------------------------------
# Step 4: Check the target distribution
# --------------------------------------
# 'fraudulent' column: 0 = Real job, 1 = Fake job
print("\n🎯 Target (Fraudulent) Distribution:")
print(df['fraudulent'].value_counts())

# --------------------------------------
# Step 5: Show 3 examples of fake job descriptions
# --------------------------------------
print("\n🕵️ Examples of Fake Job Descriptions:")
print(df[df['fraudulent'] == 1]['description'].head(3))


# --------------------------------------
# Task 2
# Step 6: Which feature has the most missing values?
# --------------------------------------
print("\n📉 Feature with Most Missing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(5))

