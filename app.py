from flask import Flask, request, render_template , redirect , url_for
import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
# import pandas as pd
import pickle

app = Flask(__name__)

model = tf.keras.models.load_model('func_API.h5', compile=False)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST' ,'GET'])
def submit():
    
    Chemical_Composition = float(request.form['Chemical_Composition'])
    Casting_Temperature = float(request.form['Casting_Temperature'])
    Cooling_Water_Temperature = float(request.form['Cooling_Water_Temperature'])
    Casting_Speed = float(request.form['Casting_Speed'])
    Entry_Temperature = float(request.form['Entry_Temperature'])
    Emulsion_Temperature = float(request.form['Emulsion_Temperature'])
    Emulsion_Pressure = float(request.form['Emulsion_Pressure'])
    Emulsion_Concentration = float(request.form['Emulsion_Concentration'])
    Quench_Water_Pressure = float(request.form['Quench_Water_Pressure'])
    import numpy as np
    input_data = [Chemical_Composition , Casting_Temperature ,  Cooling_Water_Temperature , Casting_Speed , Entry_Temperature , Emulsion_Temperature ,Emulsion_Pressure ,Emulsion_Concentration ,Quench_Water_Pressure   ]
    input_data_reshaped = np.array(input_data).reshape(1, -1)  # Reshape to (1, number_of_features)

    predictions = model.predict(input_data_reshaped)

    # print(f"Predicted UTS: {predictions[0][0]}")
    # print(f"Predicted Elongation: {predictions[1][0]}")
    # print(f"Predicted Conductivity: {predictions[2][0]}")  
    UTS_Prediction = predictions[0][0]  
    Elongation_Prediction = predictions[1][0]
    Conductivity_Prediction = predictions[2][0]
    
    return render_template('index.html', UTS_Prediction=f'Prediction: {UTS_Prediction}' , Elongation_Prediction =f'Prediction: {Elongation_Prediction}',Conductivity_Prediction=f'Prediction: {Conductivity_Prediction}' )

if __name__ == "__main__":
    app.run(debug=True)