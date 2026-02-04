from flask import Flask, render_template, request, send_from_directory
import joblib
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

import os

app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained models
cnn_model = load_model("models/model_augmented.h5")


IMG_SIZE = 100  # must match training size

def preprocess_image(file):
    # Convert uploaded file to numpy array
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Save uploaded image
            filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.seek(0)
            file.save(save_path)

            # For prediction, re-open the file
            with open(save_path, 'rb') as f:
                img = preprocess_image(f)

            # CNN Prediction
            cnn_pred = cnn_model.predict(np.expand_dims(img, axis=0))[0][0]
            cnn_label = "✅ Not Defective" if cnn_pred < 0.5 else "❌ Defective"

            prediction = {
                "CNN": cnn_label,
            }
            image_url = f"/static/uploads/{filename}"

    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
