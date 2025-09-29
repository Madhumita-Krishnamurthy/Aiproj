# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Import pickle for saving the model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# 2. Load the dataset
file_path = "enhanced_box_office_data(2000-2024)u.csv"
df = pd.read_csv(file_path)

# 3. Data Preprocessing
threshold = df["$Worldwide"].quantile(0.70)
df["Hit_Flop"] = (df["$Worldwide"] >= threshold).astype(int)

# Identify numerical and categorical columns
numerical_features = ['Rank', '$Worldwide', '$Domestic', 'Domestic %', '$Foreign',
                      'Foreign %', 'Year', 'Vote_Count']

categorical_features = ['Release Group', 'Genres', 'Rating', 'Original_Language', 'Production_Countries']

# Label encoding for categorical features
label_encoder = LabelEncoder()
for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])

# One-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Splitting into features and target
X = df_encoded[numerical_features]
y = df_encoded['Hit_Flop']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training & Evaluation
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
# Model evaluation
y_pred = svm_model.predict(X_test_scaled)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Flop', 'Hit'], yticklabels=['Flop', 'Hit'])
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()


# Save the trained model as a pickle file
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)
    
# Save scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save imputer
with open('imputer.pkl', 'wb') as file:
    pickle.dump(imputer, file)
