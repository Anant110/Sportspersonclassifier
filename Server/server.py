# from flask import Flask,request,jsonify
# from flask_cors import CORS
# from . import util
# app=Flask(__name__)
# CORS(app)

# @app.route("/classify_image",methods=['POST'])
# def classify_image():
#     image_data=request.form["image_data"]

#     response=jsonify(util.classify_image(image_data))

#     response.headers.add('Access-Control-Allow-Origin','*')

#     return response

# if __name__=="__main__":
#     print("Starting python flask for celebrity image classification")
#     util.load_saved_artifacts()
#     app.run(port=5000)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from . import util

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "UI")

@app.route("/")
def home():
    return send_from_directory(UI_DIR, "app.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(UI_DIR, path)

@app.route("/classify_image", methods=["POST"])
def classify_image():
    try:
        image_data = request.form["image_data"]
        return jsonify(util.classify_image(image_data))
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

