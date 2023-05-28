import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask_bootstrap import Bootstrap

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
Bootstrap(flask_app)


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [int(x) for x in request.form.values()]
    features = [np.array(float_features)]
    np.array([features])
    prediction = model.predict(features)
    output = prediction
    if output == 1:
        prediction = "good"
    elif output == 3:
        prediction = "very good"
    elif output == 0:
        prediction = "acceptable"
    else:
        prediction = "unacceptable"

    return render_template("index.html", prediction_text="THe Car Evaluation is ({})".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)
