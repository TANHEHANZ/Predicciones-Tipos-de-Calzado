from flask import Flask, request, jsonify
from PIL import Image
import os
import tensorflow as tf
import numpy as np
import json
from threading import Lock

app = Flask(__name__)

EXPORT_MODEL_VERSION = 1

class TFModel:
    def __init__(self, dir_path) -> None:
        self.model_dir = os.path.dirname(dir_path)
        with open(os.path.join(self.model_dir, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.model_file = os.path.join(self.model_dir, self.signature.get("filename"))
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"Model file does not exist")
        self.inputs = self.signature.get("inputs")
        self.outputs = self.signature.get("outputs")
        self.lock = Lock()

        self.model = tf.saved_model.load(tags=self.signature.get("tags"), export_dir=self.model_dir)
        self.predict_fn = self.model.signatures["serving_default"]

        version = self.signature.get("export_model_version")
        if version is None or version != EXPORT_MODEL_VERSION:
            print(f"There has been a change to the model format. Please use a model with a signature 'export_model_version' that matches {EXPORT_MODEL_VERSION}.")

    def predict(self, image: Image.Image) -> dict:
        image = self.process_image(image, self.inputs.get("Image").get("shape"))

        with self.lock:
            feed_dict = {}
            feed_dict[list(self.inputs.keys())[0]] = tf.convert_to_tensor(image)
            outputs = self.predict_fn(**feed_dict)
            return self.process_output(outputs)

    def process_image(self, image, input_shape) -> np.ndarray:
        width, height = image.size
        if image.mode != "RGB":
            image = image.convert("RGB")
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            image = image.crop((left, top, right, bottom))
        input_width, input_height = input_shape[1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))

        image = np.asarray(image) / 255.0
        return np.expand_dims(image, axis=0).astype(np.float32)

    def process_output(self, outputs) -> dict:
        out_keys = ["label", "confidence"]
        results = {}
        for key, tf_val in outputs.items():
            val = tf_val.numpy().tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        confs = results["Confidences"]
        labels = self.signature.get("classes").get("Label")
        output = [dict(zip(out_keys, group)) for group in zip(labels, confs)]
        sorted_output = {"predictions": sorted(output, key=lambda k: k["confidence"], reverse=True)}
        return sorted_output

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
