from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow import keras

app = Flask(__name__)

# Load model
model = keras.models.load_model('models/stunting_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # features = np.array([[data['Gender'], data['Age'], data['Body Length']]]) 
    features = np.array([[data['Jenis Kelamin'], data['Umur (bulan)'], data['Tinggi Badan (cm)']]])

    # Tanpa scaler
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Mapping hasil
    labels = {0: "severely stunted", 1: "stunted", 2: "normal", 3: "tinggi"}
    stunting_status = labels[predicted_class]

    return jsonify({'stunting_status': stunting_status})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
