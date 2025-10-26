from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# --- Define Cluster Descriptions ---
CLUSTER_DESCRIPTIONS = {
    "0": "Avg. Hour: ~12:15 PM, Injury Risk: Medium-Low (0.37)",
    "1": "Avg. Hour: ~2:45 PM, Injury Risk: High (0.46)",
    "2": "Avg. Hour: ~1:20 PM, Injury Risk: Medium-High (0.40)",
    "3": "Catch-all Cluster: Avg. hour ~1:00 PM, but includes many off-peak outliers.",
    "4": "Avg. Hour: ~2:20 PM, Injury Risk: Lowest (0.15)"
}

# --- Load All Models and Data ---
try:
    # Classification model
    model = joblib.load('accident_severity_model.pkl')
    encoder = joblib.load('encoder.pkl')
    model_columns = joblib.load('model_columns.pkl')
    
    # Clustering models
    kmeans_model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
        
    print("All models and data loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please run train_model.py first.")
    exit()

# --- Helper Function for Data Processing ---
def process_input_data(data):
    input_df = pd.DataFrame([data])
    categorical_features = ['weather_condition', 'lighting_condition', 'prim_contributory_cause']
    numerical_features = ['crash_hour', 'crash_day_of_week']
    
    encoded_data = pd.DataFrame(encoder.transform(input_df[categorical_features]), columns=encoder.get_feature_names_out())
    numerical_data = input_df[numerical_features].reset_index(drop=True)
    processed_df = pd.concat([encoded_data, numerical_data], axis=1)
    
    processed_df = processed_df.reindex(columns=model_columns, fill_value=0)
    return processed_df

# --- API Endpoint 1: Prediction (MODIFIED) ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        processed_df = process_input_data(data)
        
        # 1. Get the final prediction
        prediction = model.predict(processed_df)[0]
        
        # 2. Get the prediction probabilities (NEW)
        probabilities = model.predict_proba(processed_df)[0]
        
        # 3. Get the class names from the model
        class_names = model.classes_
        
        # 4. Combine class names and probabilities into a nice list
        prob_list = []
        for i, class_name in enumerate(class_names):
            prob_list.append({
                'class': class_name.title(), # e.g., "Fatal"
                'probability': round(probabilities[i] * 100, 2) # e.g., 15.2
            })
        
        # 5. Sort by probability, descending
        prob_list.sort(key=lambda x: x['probability'], reverse=True)
        
        # 6. Return the full data
        return jsonify({
            'predicted_severity': prediction.title(), # e.g., "Nonincapacitating Injury"
            'probabilities': prob_list
        })

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 400

# --- API Endpoint 2: Analysis (MODIFIED) ---
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        processed_df = process_input_data(data)
        
        # 1. Scale and predict cluster
        scaled_data = scaler.transform(processed_df)
        cluster_prediction = kmeans_model.predict(scaled_data)
        cluster_num = str(cluster_prediction[0])
        
        # 2. Get the description
        description = CLUSTER_DESCRIPTIONS.get(cluster_num, "A standard accident type.")
        
        return jsonify({
            'cluster_number': cluster_num,
            'description': description
        })

    except Exception as e:
        print(f"Error in /analyze: {e}")
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5000)