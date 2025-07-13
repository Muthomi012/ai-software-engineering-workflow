import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load test data
df = pd.read_csv("data/processed_data.csv")
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Reload model
model = joblib.load('model/readmission_model.pkl')

# Predictions
y_pred = model.predict(X)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
