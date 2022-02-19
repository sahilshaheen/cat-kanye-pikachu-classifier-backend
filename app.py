from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import os

port = int(os.environ.get('PORT', 5000))
app = Flask(__name__)
CORS(app)

conv_base = tf.keras.applications.Xception(input_shape=(150, 150, 3), include_top=False)
top_model = tf.keras.models.load_model("xception-top.keras")
class_names = ['a cat', 'Kanye', 'Pikachu'] 

@app.route("/", methods=["POST"])
def home():
    if request.method == "POST":
        try:
            content = request.files['image'].read()
            im = Image.open(io.BytesIO(content)).convert("RGB").resize((150, 150))
            im_arr = np.array(im)
            im_preprocessed = tf.keras.applications.xception.preprocess_input(im_arr)
            im_batched = np.expand_dims(im_preprocessed, axis=0)
            im_features = conv_base.predict(im_batched)
            prediction = top_model.predict(im_features)
            label_idx = prediction.argmax(axis=-1)[0]
            label = class_names[label_idx]
            score = prediction[0][label_idx]
            return jsonify(label=label, score=f'{score}')
        except Exception as e:
            print(e)
            return "Internal Server Error", 500

if __name__ == "__main__":
    app.run(port=port, debug=True)
