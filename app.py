import io

import flask
import numpy as np
from flask import Flask, jsonify
from PIL import Image

from CenterNetEngine import CenterNetEngine

app = Flask(__name__)

clf = CenterNetEngine()

@app.route("/ready", methods=["GET"])
def check_connection():
    """End point to check connection
    """
    return jsonify({"status": "ready"})


@app.route("/predict", methods=["POST"])
def predict():
    """End point for image to be posted
    """
    ret = {"success": False, "detections": None}

    if flask.request.method == "POST":
        image_bytes = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image_bytes))
        #image_np = np.asarray(image, dtype="int32")
        image_np = np.asarray(image, dtype="uint8")

        pred = clf.show_image(image_np)
        ret["detections"] = pred
        ret["success"] = True
    else:
        print("Route only accepts POST")

    return jsonify(ret)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
