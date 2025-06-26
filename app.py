# server.py

import joblib
import pandas as pd
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load the trained model and label encoders
model = joblib.load('XGBoost_Treatment_Model.pkl')
label_encoders = joblib.load('label_encoders.pkl')  # save & load this as needed

# Define the order of features expected by the model
ordered_features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Nitrogen',
                    'Potassium', 'Phosphorous', 'Fertilizer Name', 'Disease']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Encode categorical features
        for col in ['Soil Type', 'Fertilizer Name', 'Disease']:
            data[col] = label_encoders[col].transform([data[col]])[0]
        
        # Create input DataFrame
        input_data = pd.DataFrame([[data[feature] for feature in ordered_features]], columns=ordered_features)

        # Predict
        pred = model.predict(input_data)[0]
        treatment = label_encoders['Recommended Treatment'].inverse_transform([pred])[0]

        return jsonify({'predicted_treatment': treatment})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

