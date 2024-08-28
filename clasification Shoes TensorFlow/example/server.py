from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from model import TFModel  # Importar el modelo
import os

app = Flask(__name__)
CORS(app)  # Configura CORS para aceptar solicitudes desde cualquier origen

# Inicializar el modelo
model = TFModel(dir_path=os.getcwd())

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image = Image.open(file)
    prediction = model.predict(image)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
