from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

model = load_model('mlp_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = np.array([[
            data['pregnancies'],
            data['glucose'],
            data['bloodPressure'],
            data['skinThickness'],
            data['insulin'],
            data['bmi'],
            data['diabetesPedigreeFunction'],
            data['age']
        ]])
    except Exception as e:
        return jsonify({"error": "Geçersiz girdi formatı", "details": str(e)}), 400

    input_data_scaled = scaler.transform(input_data)
    prediction_prob = model.predict(input_data_scaled)[0][0]
    prediction = 1 if prediction_prob > 0.5 else 0

    return jsonify({
        "prediction": prediction,
        "probability": float(prediction_prob)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
