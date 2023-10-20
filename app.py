import numpy as np
import pandas as pd
from flask import Flask, request, app, jsonify, url_for, render_template
import pickle
import os
import json
import matplotlib.pyplot as plt


# Creating the flask application instance
# __name__ defines the starting point of the website
app = Flask(__name__)

# Load the Prediction & Scaler Model
model = pickle.load(open('Regression.pkl','rb'))
scaler = pickle.load(open('standardizer.pkl','rb'))

# Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Prediction Page
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    scaled_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    prediction  = model.predict(scaled_data)
    print(prediction[0])
    return jsonify(prediction[0])

# Main function
if __name__ == "__main__":
    app.run(debug=True)