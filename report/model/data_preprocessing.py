import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv("data/sample_patient_data.csv")

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df['age'] = imputer.fit_transform(df[['age']])

# One-hot encode diagnosis
df = pd.get_dummies(df, columns=['diagnosis_code'])

# Normalize numerical features
scaler = StandardScaler()
df[['age', 'length_of_stay']] = scaler.fit_transform(df[['age', 'length_of_stay']])

df.to_csv("data/processed_data.csv", index=False)
print("Data preprocessing complete.")
