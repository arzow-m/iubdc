import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df_grouped = pd.read_csv('patient_data.csv')
# create target object, call it y
y = df_grouped['had_adr']

#create features, call it X
features = ['gender', 'anchor_age', 'num_unique_diagnoses', 'num_unique_drugs', 'num_unique_procedures']
X = df_grouped[features]

#split into validation and training data (80% for training, 20% for validation)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state=1)

#define the model + fit it
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 1, class_weight = 'balanced')
rf_model.fit(train_X, train_y)

#make predictions and accuracy
pred_val = rf_model.predict(val_X)
accuracy = accuracy_score(val_y, pred_val)
print(f"Validation Accuracy: {accuracy:.4f}")

#classification report
print("\nClassification Report:")
print(classification_report(val_y, pred_val))

#confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(val_y, pred_val))