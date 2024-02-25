import sys

from sklearn.ensemble import RandomForestClassifier  # For classification
from sklearn.ensemble import RandomForestRegressor  # For regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score  # For classification
from sklearn.metrics import mean_squared_error  # For regression
import pandas as pd  # For data manipulation
from sklearn.preprocessing import OneHotEncoder
from missingpy import KNNImputer, MissForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

df = pd.read_csv('data/HCC dataset/hcc-data.csv')

df.columns = ['Gender', 'Symptoms', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B E Antigen',
              'Hepaititis B Core Antibody', 'Hepatitis C Virus Antibody', 'Cirrhosis', 'Endemic Countries', 'Smoking',
              'Diabetes', 'Obesity', 'Hemochromatosis', 'Arterial Hypertension', 'Chronic Renal Insuffeciency',
              'Human Inmmunodeficiency Virus', 'Nonalcoholic Steatohepatitis', 'Esophageal Varices', 'Splenomegaly',
              'Portal Hypertension', 'Portal Vein Thrombosis', 'Liver Metasis', 'Radiological Hallmark',
              'Age at diagnosis', 'Grams of Alcohol per day', 'Packs of cigarettes per year', 'Performance Status',
              'Encephalopathy Degree', 'Ascites Degree', 'International Normalised ratio', 'Alpha-Fetoprotein (ng/mL)',
              'Hemoglobin (g/dL)', 'Mean Corpuscular Volume', 'Leukocytes(G/L)', 'Platelets (G/L)', 'Albumin (mg/dL)',
              'Total Bilirubin (mg/dL)', 'Alanine Transaminase (U/L)', 'Asparate Transaminase (U/L)',
              'Gamma Glutamyl Transferase (U/L)', 'Alkaline Phosphatase (U/L)', 'Total Proteins (g/dL)',
              'Creatinine (mg/dL)', 'Number of Nodules', 'Major dimension of nodule (cm)', 'Direct Bilirubin (mg/dL)',
              'Iron (mcg/dL)', 'Oxygen Saturation (%)', 'Ferritin (ng/mL)', 'Class Attribute']
important_features = ['Gender', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B E Antigen',
                      'Hepaititis B Core Antibody', 'Hepatitis C Virus Antibody', 'Cirrhosis',
                      'Smoking', 'Diabetes', 'Obesity', 'Nonalcoholic Steatohepatitis', 'Radiological Hallmark', 'Age at diagnosis', 'Grams of Alcohol per day',
                      'Packs of cigarettes per year', 'Alpha-Fetoprotein (ng/mL)']

X = df[important_features]  # Use your defined features
y = df['Gender']  # Assuming "liver_cancer" is the target column

# Handle categorical features (if needed)
categorical_features = ['Gender', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B E Antigen',
                      'Hepaititis B Core Antibody', 'Hepatitis C Virus Antibody', 'Cirrhosis',
                      'Smoking', 'Diabetes', 'Obesity', 'Nonalcoholic Steatohepatitis', 'Radiological Hallmark', 'Age at diagnosis', 'Grams of Alcohol per day',
                      'Packs of cigarettes per year', 'Alpha-Fetoprotein (ng/mL)']
encoder = OneHotEncoder(handle_unknown="ignore")
X_encoded = encoder.fit_transform(X[categorical_features])
X = pd.concat([X.drop(categorical_features, axis=1), pd.DataFrame(X_encoded.toarray())], axis=1)

imputer = KNNImputer(n_neighbors=5)  # adjust k as needed
X_imputed = imputer.fit_transform(X)

imputer = MissForest(n_estimators=100, max_depth=5)  # adjust hyperparameters as needed
X_imputed = imputer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Create the model (adjust hyperparameters as needed)
model = RandomForestClassifier(n_estimators=100, max_depth=5)

# Train the model
model.fit(X_train, y_train)

# Generate risk scores for the testing set
risk_scores = model.predict_proba(X_test)[:, 1]  # Extract probabilities for the positive class

# Evaluate model performance
accuracy = accuracy_score(y_test, model.predict(X_test))
roc_auc = roc_auc_score(y_test, risk_scores, multi_class='ovo')
print("Accuracy:", accuracy)
print("ROC AUC:", roc_auc)



