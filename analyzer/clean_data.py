import pandas as pd
from sklearn.preprocessing import LabelEncoder

# load mimic-iv tables
df_patient = pd.read_csv('/patients.csv.gz', index_col=0, compression='gzip')
df_diagnose = pd.read_csv('/diagnoses_icd.csv.gz', index_col=0, compression='gzip')
df_prescription = pd.read_csv('/prescriptions.csv.gz', index_col=0, compression='gzip')
df_procedure = pd.read_csv('/procedures_icd.csv.gz', index_col=0, compression='gzip')
df_chartevents = pd.read_csv('/chartevents.csv.gz', index_col=0, compression='gzip')

# filter for elderly patients
df_patient = df_patient[df_patient['anchor_age'] >= 65]
df_mimic = df_patient[['gender', 'anchor_age']].copy()
df_mimic.reset_index(inplace=True)  # now has subject_id column

# diagnoses
diagnosis_counts = df_diagnose[df_diagnose.index.isin(df_mimic['subject_id'])] \
    .groupby('subject_id')['icd_code'].nunique().reset_index(name='num_unique_diagnoses')
df_mimic = df_mimic.merge(diagnosis_counts, on='subject_id', how='left')

# prescriptions
drug_counts = df_prescription[df_prescription.index.isin(df_mimic['subject_id'])] \
    .groupby('subject_id')['drug'].nunique().reset_index(name='num_unique_drugs')
drug_counts['polypharmacy'] = (drug_counts['num_unique_drugs'] >= 5).astype(int)
df_mimic = df_mimic.merge(drug_counts, on='subject_id', how='left')

# procedures
procedure_counts = df_procedure[df_procedure.index.isin(df_mimic['subject_id'])] \
    .groupby('subject_id')['icd_code'].nunique().reset_index(name='num_unique_procedures')
df_mimic = df_mimic.merge(procedure_counts, on='subject_id', how='left')

# lab averages
# define common lab item IDs (replace with actual MIMIC item IDs if different)
lab_ids = {
    'creatinine': 50912,
    'potassium': 50971
}

# adr label (from chart events - itemid 227968)
adr_event = df_chartevents[(df_chartevents['itemid'] == 227968) & (df_chartevents['value'] == "No")]
adr_patients = adr_event.index.unique().tolist()
df_mimic['had_adr'] = df_mimic['subject_id'].isin(adr_patients).astype(int)

# handle missing values
df_mimic.fillna({
    'num_unique_diagnoses': 0,
    'num_unique_drugs': 0,
    'num_unique_procedures': 0,
}, inplace=True)

# gender
le = LabelEncoder()
df_mimic['gender'] = le.fit_transform(df_mimic['gender'])  # F = 0, M = 1

# export final csv
df_mimic.to_csv('clean_mimic_extended.csv', index=False)
print(df_mimic.head())
