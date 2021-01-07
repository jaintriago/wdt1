from flask import Flask, jsonify, request, make_response
import sklearn
import pickle
import numpy as numpy
import pandas as pd 

### declarar el servidor 

app = Flask(__name__)
modelTest = pickle.load(open('RF_VF.pkl', 'rb'))

def predecir(**objeto):
    feat=dict(objeto)
    df_feat=pd.DataFrame(columns=(list(feat.keys())))
    df_feat.loc[len(df_feat)]=list(feat.values())
    print(df_feat)
    pr=modelTest.predict(df_feat)
    print(pr)
    return int(pr[0])

@app.route('/')

def index():
    print(modelTest)
    return "<h1>Hola este es el deploy del codigo de python con flask</h1>"

@app.route('/predict', methods=['POST'])

def pred():
    obj = request.json
    p=predecir(**obj)
    return jsonify({'prediction': p})


## ejecución del servidor

if __name__ == '__main__':
    app.run(debug=True)