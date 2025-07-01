import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# load dataset
df_grouped = pd.read_csv('mimic_patient_data.csv')

# create target (label) and features
y = df_grouped['had_adr']
features = ['gender', 'anchor_age', 'num_unique_diagnoses', 'num_unique_drugs', 'num_unique_procedures']
X = df_grouped[features]

# split into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

# initialize and train XGBoost classifier
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=(train_y == 0).sum() / (train_y == 1).sum(),  # balance ADR class
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=1
)
xgb_model.fit(train_X, train_y)

# make predictions
pred_val = xgb_model.predict(val_X)

# evaluate model
accuracy = accuracy_score(val_y, pred_val)
print(f"Validation Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(val_y, pred_val))

print("Confusion Matrix:")
print(confusion_matrix(val_y, pred_val))

# get and plot feature importances
importances = xgb_model.feature_importances_
feature_names = ['gender', 'age', ' # of unique diagnoses', '# of unique drugs', '# of unique procedures']
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

feature_importance_df
