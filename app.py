import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load model and encoders
model = joblib.load("XGBoost_Treatment_Model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Ordered features expected by the model
ordered_features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Nitrogen',
                    'Potassium', 'Phosphorous', 'Fertilizer Name', 'Disease']

# ✅ Health check route so Render doesn't throw 502
@app.route('/')
def home():
    return "✅ Soil Treatment API is running!"

# ✅ Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Encode categorical inputs
        for col in ['Soil Type', 'Fertilizer Name', 'Disease']:
            data[col] = label_encoders[col].transform([data[col]])[0]

        # Create input dataframe
        input_df = pd.DataFrame([[data[feat] for feat in ordered_features]], columns=ordered_features)

        # Predict and decode
        pred = model.predict(input_df)[0]
        treatment = label_encoders['Recommended Treatment'].inverse_transform([pred])[0]

        return jsonify({'predicted_treatment': treatment})
    except Exception as e:
        return jsonify({'error': str(e)})


# ✅ Correct way to bind the app to Render's dynamic port
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
