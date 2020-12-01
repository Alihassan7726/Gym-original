# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:52:18 2020

@author: Ali
"""


import pickle
import numpy as np
from flask import Flask, request

model = None
app = Flask(__name__)

def load_model():
    global model
    # model variable refers to the global variable
    with open('Ali-gym-diet.pkl', 'rb') as f:
        model = pickle.load(f)
        
@app.route('/')
def home_endpoint():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = np.array(data)[np.newaxis, :]  # converts shape from (3,) to (1, 3)
        prediction = model.predict(data)  # runs globally loaded model on the data
    return render_template('after.html',data=prediction)

if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)
    
