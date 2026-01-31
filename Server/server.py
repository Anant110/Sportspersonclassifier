from flask import Flask,request,jsonify
from flask_cors import CORS
import util
app=Flask(__name__)
CORS(app)

@app.route("/classify_image",methods=['POST'])
def classify_image():
    image_data=request.form["image_data"]

    response=jsonify(util.classify_image(image_data))

    response.headers.add('Access-Control-Allow-Origin','*')

    return response

if __name__=="__main__":
    print("Starting python flask for celebrity image classification")
    util.load_saved_artifacts()
    app.run(port=5000)