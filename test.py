import pandas as pd
import numpy as np

# Number of data points to generate
num_data_points = 500

# Create empty DataFrame with desired features
data = pd.DataFrame(columns= ['Gender', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B E Antigen',
                      'Hepaititis B Core Antibody', 'Hepatitis C Virus Antibody', 'Cirrhosis',
                      'Smoking', 'Diabetes', 'Obesity', 'Nonalcoholic Steatohepatitis', 'Radiological Hallmark', 'Age at diagnosis', 'Grams of Alcohol per day',
                      'Packs of cigarettes per year', 'Alpha-Fetoprotein (ng/mL)', 'Liver Cancer Risk'])


# Function to generate a single data point with correlations
def generate_data_point():
    # 1 = male & 0 = female
    gender = np.random.choice([1, 0])
    # 1 = yes & 0 = no
    alcohol_category = np.random.choice([0, 1], p=[0.4, 0.6])
    alcohol_grams = 0  # Default for non-drinkers
    if alcohol_category == 1:
        alcohol_grams = np.random.normal(20, 5)
    # 1 = yes & 0 = no
    HepatitisBSurfaceAntigen_category = np.random.choice([1, 0], p=[0.02, 0.98])
    # 1 = yes & 0 = no
    HepatitisBeAntigen_category = np.random.choice([1, 0], p=[0.005, 0.995])
    # 1 = yes & 0 = no
    HepatitisBCoreAntibody_category = np.random.choice([1, 0], p=[0.07, 0.93])
    # 1 = yes & 0 = no
    HepatitisCVirusAntibody_category = np.random.choice([1, 0], p=[0.05, 0.95])
    # 1 = yes & 0 = no
    Cirhosis_category = np.random.choice([1, 0], p=[0.05, 0.95])
    # 1 = yes & 0 = no
    Smoking_category = np.random.choice([1, 0], p=[0.15, 0.85])
    cigarette_packs = 0
    if Smoking_category == 1:
        cigarette_packs = np.random.normal(150, 30)
    # 1 = yes & 0 = no
    Diabetes_category = np.random.choice([1, 0], p=[0.10, 0.90])
    # 1 = yes & 0 = no
    Obesity_category = np.random.choice([1, 0], p=[0.20, 0.80])
    # 1 = yes & 0 = no
    NASH_category = np.random.choice([1, 0], p=[0.10, 0.90])
    # 1 = yes & 0 = no
    Radiological_category = np.random.choice([1, 0], p=[0.025, 0.975])
    age_category = np.random.normal(46, 13)
    AFP_Category = np.random.normal(14.40, 222.60)
    if AFP_Category < 0:
        AFP_Category = 10;

    liver_cancer_risk = 0.1  # Base risk
    liver_cancer_risk += 0.05 * (alcohol_category == 1)
    liver_cancer_risk += 0.0001 * alcohol_grams
    liver_cancer_risk += 0.025 * (HepatitisBSurfaceAntigen_category == 1)
    liver_cancer_risk += 0.025 * (HepatitisBeAntigen_category == 1)
    liver_cancer_risk += 0.06 * (Cirhosis_category == 1)
    liver_cancer_risk += 0.01 * (HepatitisCVirusAntibody_category == 1)
    liver_cancer_risk += 0.0001 * (Smoking_category == 1)
    if Smoking_category == 1:
        liver_cancer_risk += 0.0005 * cigarette_packs
    liver_cancer_risk += 0.01 * (Diabetes_category == 1)
    liver_cancer_risk += 0.01 * (Obesity_category == 1)
    liver_cancer_risk += 0.05 * (NASH_category == 1)
    liver_cancer_risk += 0.05 * (Radiological_category == 1)
    liver_cancer_risk += 0.007 * age_category
    if AFP_Category > 400:
        liver_cancer_risk += 0.1
    if AFP_Category <= 500:
        liver_cancer_risk += 0.01 * (1 / 300000) * ((AFP_Category - 100)**2)
    if liver_cancer_risk > 1:
        liver_cancer_risk = 1

    liver_cancer_risk_category = 0
    if liver_cancer_risk > 0.7:
        liver_cancer_risk_category = 1


    return [gender, alcohol_category, HepatitisBSurfaceAntigen_category, HepatitisBeAntigen_category,HepatitisBCoreAntibody_category,
            HepatitisCVirusAntibody_category, Cirhosis_category, Smoking_category, Diabetes_category,
            Obesity_category, NASH_category, Radiological_category, age_category,alcohol_grams, cigarette_packs, AFP_Category,
            liver_cancer_risk_category]  # Return values for all features


# Generate data points and fill the DataFrame
for i in range(num_data_points):
    data.loc[i] = generate_data_point()

df = pd.read_csv('data/HCC dataset/hcc-data1.csv')
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
              'Iron (mcg/dL)', 'Oxygen Saturation (%)', 'Ferritin (ng/mL)', 'Class Attribute', 'Liver Cancer Risk']
important_features = ['Gender', 'Alcohol', 'Hepatitis B Surface Antigen', 'Hepatitis B E Antigen',
                      'Hepaititis B Core Antibody', 'Hepatitis C Virus Antibody', 'Cirrhosis',
                      'Smoking', 'Diabetes', 'Obesity', 'Nonalcoholic Steatohepatitis', 'Radiological Hallmark', 'Age at diagnosis', 'Grams of Alcohol per day',
                      'Packs of cigarettes per year', 'Alpha-Fetoprotein (ng/mL)', 'Liver Cancer Risk']
X = df[important_features]

main = pd.concat([X, data], axis=0)

# Save DataFrame to CSV file
main.to_csv('liver_cancer_data.csv', index=False)

print('Data generated and saved to liver_cancer_data.csv')