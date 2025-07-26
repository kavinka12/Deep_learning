from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

# Load model and class names
model = tf.keras.models.load_model("model.keras")
class_names = ['with_mask', 'without_mask']  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Prediction function
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            predicted_class, confidence = predict_image(filepath)

            return render_template("index.html", image_path=filepath,
                                   prediction=predicted_class
                                   )
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
