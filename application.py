import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            id=request.form.get('id'),
            Sex=request.form.get('Sex'),
            Age=float(request.form.get('Age')),
            Height=float(request.form.get('Height')),
            Weight=float(request.form.get('Weight')),
            Duration=float(request.form.get('Duration')),
            Heart_Rate=float(request.form.get('Heart_Rate')),
            Body_Temp=float(request.form.get('Body_Temp'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=round(results[0], 2))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))  # fallback for local testing
    app.run(host='0.0.0.0', port=port)
