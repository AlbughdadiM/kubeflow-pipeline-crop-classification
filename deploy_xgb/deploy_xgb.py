from flask import Flask
from flask import request
import argparse
import joblib
import json
import os
import numpy as np


model_path = '/model/data'
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/inference', methods = ['POST'])
def inference():
    if request.method == 'POST':
        data = np.array(json.loads(request.data.decode())['data'])
        data = np.expand_dims(data,0)
        label = model.predict(data)
        return json.dumps({'label':int(label[0])})

    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))



