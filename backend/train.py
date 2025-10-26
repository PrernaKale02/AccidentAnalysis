import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import json

print("Script started...")

# 1. Load Data
try:
    df = pd.read_csv('data/traffic_accidents.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: traffic_accidents.csv not found. Make sure it's in the 'backend/data/' folder.")
    exit()

# 2. Define Target and Features
df = df.dropna(subset=['most_severe_injury', 'weather_condition', 'lighting_condition', 'prim_contributory_cause'])

# Define our target variable (y)
target = 'most_severe_injury'
y = df[target]

# Define our features (X)
features = ['weather_condition', 'lighting_condition', 'prim_contributory_cause', 'crash_hour', 'crash_day_of_week']
X = df[features]

# 3. Preprocessing: One-Hot Encoding for Categorical Data
categorical_features = ['weather_condition', 'lighting_condition', 'prim_contributory_cause']
numerical_features = ['crash_hour', 'crash_day_of_week']

# Apply One-Hot Encoding to categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded_categorical = pd.DataFrame(encoder.fit_transform(X[categorical_features]), columns=encoder.get_feature_names_out())

# Combine encoded features with numerical features
X_numerical = X[numerical_features].reset_index(drop=True)
X_processed = pd.concat([X_encoded_categorical, X_numerical], axis=1)

from imblearn.over_sampling import SMOTE

print("Data processed. Total features:", len(X_processed.columns))

# --- FIX IMBALANCED DATA ---
print("\nApplying SMOTE to fix data imbalance...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_processed, y)
print("Resampling complete.")
print("Original y distribution:\n", y.value_counts())
print("Resampled y distribution:\n", y_resampled.value_counts())

# 4. --- TASK A: CLASSIFICATION (Predicting Severity) ---
print("\n--- Starting Classification ---")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize a Decision Tree model
model = DecisionTreeClassifier(max_depth=10, random_state=42)

# Train the model
model.fit(X_train, y_train)
print("Classification model trained.")

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 5. Save the Model and the Encoder
joblib.dump(model, 'accident_severity_model.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(X_processed.columns.tolist(), 'model_columns.pkl')
print("Model, encoder, and column list saved to .pkl files.")


# 6. --- TASK B: CLUSTERING (Finding Accident Groups) ---
print("\n--- Starting Clustering ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# Run K-Means to find types of accidents
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['accident_cluster'] = kmeans.fit_predict(X_scaled)

print("Clustering complete. 'accident_cluster' column added.")
print("\nCluster Analysis (showing mean values):")
print(df.groupby('accident_cluster')[['crash_hour', 'num_units', 'injuries_total']].mean())

print("\nSaving clustering models...")

# 1. Save the K-Means model
joblib.dump(kmeans, 'kmeans_model.pkl')

# 2. Save the Scaler
joblib.dump(scaler, 'scaler.pkl')

# 3. Save the cluster analysis results
cluster_analysis = df.groupby('accident_cluster')[['crash_hour', 'num_units', 'injuries_total']].mean().to_dict('index')

with open('cluster_analysis.json', 'w') as f:
    json.dump(cluster_analysis, f, indent=4)

print("Clustering models and analysis saved.")
print("\nScript finished.")
