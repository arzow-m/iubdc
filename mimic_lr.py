import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the dataset
df_grouped = pd.read_csv('mimic_patient_data.csv')

# create target and features
y = df_grouped['had_adr']
features = ['gender', 'anchor_age', 'num_unique_diagnoses', 'num_unique_drugs', 'num_unique_procedures']
X = df_grouped[features]

# split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

# standardize the features (important for logistic regression)
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
val_X_scaled = scaler.transform(val_X)

# train logistic regression with class weight to address imbalance
logreg_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=1)
logreg_model.fit(train_X_scaled, train_y)

# make predictions
pred_val = logreg_model.predict(val_X_scaled)

# evaluate the model
accuracy = accuracy_score(val_y, pred_val)
print(f"Validation Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(val_y, pred_val))

print("Confusion Matrix:")
print(confusion_matrix(val_y, pred_val))

# plot feature importance (absolute value of coefficients)
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': abs(logreg_model.coef_[0])
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='magma')
plt.title("Logistic Regression Feature Importance")
plt.tight_layout()
plt.show()

feature_importance_df