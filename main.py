import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

print("EMPLOYEE DATA ANALYSIS\n")

# Load data
df = pd.read_csv(r"C:\Users\ycnit\Downloads\Employee_Messy.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

# DATA CLEANING
print("Missing values before:", df.isnull().sum().sum())

# Fix inconsistent text (case issues)
text_cols = ['Education', 'City', 'Gender', 'EverBenched']
for col in text_cols:
    df[col] = df[col].str.strip().str.title()

# Convert numeric columns and handle errors
df['JoiningYear'] = pd.to_numeric(df['JoiningYear'], errors='coerce')
df['PaymentTier'] = pd.to_numeric(df['PaymentTier'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['ExperienceInCurrentDomain'] = pd.to_numeric(df['ExperienceInCurrentDomain'], errors='coerce')
df['LeaveOrNot'] = pd.to_numeric(df['LeaveOrNot'], errors='coerce')

# Fill missing values
numeric_cols = ['Age', 'JoiningYear', 'PaymentTier', 'ExperienceInCurrentDomain', 'LeaveOrNot']
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in text_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values after:", df.isnull().sum().sum())

# Remove duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates} duplicates")

# Handle outliers
for col in ['Age', 'JoiningYear', 'PaymentTier', 'ExperienceInCurrentDomain']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

# DATA TRANSFORMATION
df['YearsInCompany'] = 2024 - df['JoiningYear']

df_encoded = df.copy()
for col in ['Education', 'City', 'Gender', 'EverBenched']:
    le = LabelEncoder()
    df_encoded[col + '_Encoded'] = le.fit_transform(df[col])

df.to_csv("Employee_Cleaned.csv", index=False)
print("\nCleaned data saved\n")

# EXPLORATORY DATA ANALYSIS
print("Summary Statistics:")
print(df[['Age', 'JoiningYear', 'PaymentTier', 'ExperienceInCurrentDomain']].describe())

# Histograms
for col in ['Age', 'JoiningYear', 'PaymentTier', 'ExperienceInCurrentDomain']:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Boxplots
for col in ['Age', 'JoiningYear', 'PaymentTier', 'ExperienceInCurrentDomain']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
corr_cols = ['Age', 'JoiningYear', 'PaymentTier', 'ExperienceInCurrentDomain',
             'Education_Encoded', 'Gender_Encoded', 'LeaveOrNot']
sns.heatmap(df_encoded[corr_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# INSIGHTS
attrition_rate = (df['LeaveOrNot'].sum() / len(df)) * 100
print(f"\nAttrition Rate: {attrition_rate:.2f}%")
print(f"Average Age: {df['Age'].mean():.1f} years")
print(f"Average Experience: {df['ExperienceInCurrentDomain'].mean():.1f} years\n")

print("Attrition by Payment Tier:")
print(df.groupby('PaymentTier')['LeaveOrNot'].mean() * 100)

print("\nAttrition by City:")
print(df.groupby('City')['LeaveOrNot'].mean() * 100)

print("\nAttrition by Gender:")
print(df.groupby('Gender')['LeaveOrNot'].mean() * 100)

print("\nAnalysis Complete!")
