import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df_grouped = pd.read_csv('mimic_patient_data.csv')
# create target object, call it y
y = df_grouped['had_adr']

# create features, call it X
features = ['gender', 'anchor_age', 'num_unique_diagnoses', 'num_unique_drugs', 'num_unique_procedures']
X = df_grouped[features]

# split into validation and training data (80% for training, 20% for validation)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state=1)

# define the model + fit it
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 1, class_weight = 'balanced')
rf_model.fit(train_X, train_y)

# make predictions and accuracy
pred_val = rf_model.predict(val_X)
accuracy = accuracy_score(val_y, pred_val)
print(f"Validation Accuracy: {accuracy:.4f}")

# classification report
print("\nClassification Report:")
print(classification_report(val_y, pred_val))

# confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(val_y, pred_val))

# get feature importances
importances = rf_model.feature_importances_
feature_names = ['gender', 'age', 'unique diagnoses', 'unique drugs', 'unique procedures']
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# plot feature importances
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

feature_importance_df