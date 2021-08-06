from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import numpy as np
import sys
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
from sklearn.preprocessing import MinMaxScaler

from Net import Net
from calculate_descriptors import calculate_descriptors_for_single_smiles, calculate_descriptors_for_multiple_smiles
from test_DNN import test_DNN


app = Flask(__name__)

# load model
device='cpu'
model = torch.load('entire_model.pt', map_location=device)
#print('Model state dict imported')

scaler = joblib.load('min_max_scaler_1.joblib')
#print('scaler model imported')

descr_names = pd.read_csv('Descriptor_names_before_scaling.csv',header=None,index_col = None)
descr_names.columns = ['cols']
#print('\ndescr_names\n')
#print(descr_names.head())

descr_mean_values = pd.read_csv('X_TRAIN_mean.csv',header=0,index_col = None)
#print('\ndescr_mean_values\n')
#print(descr_mean_values.head())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #input_smiles = str(request.form.values()) 
    input_smiles = request.form['text']
    int_features = calculate_descriptors_for_single_smiles(input_smiles)
    #print('Descriptor calculation done!')
    # only keep columns that are present in descr_names table
    int_features = int_features[descr_names.cols]
    #print(f'new dimension {int_features.shape}')
    # fill missing values with mean of descr values
    int_features = int_features.fillna(descr_mean_values)
    #print('missing values filled')
    
    X_TEST = pd.DataFrame(scaler.transform(int_features.values), columns=int_features.columns, index=None)
    #print(f'Scaling finished! X_TEST shape : {X_TEST.shape}')
    
    # train
    Query_prob = test_DNN(model, X_TEST)
    torch.set_printoptions(sci_mode=False)
    #print(f'Query_prob : {Query_prob}')
    Query_prob_1 = Query_prob[0][1]
    
    #predicted_class = np.where(Query_prob_1 >= 0.0037, 1, 0)
    
    predicted_class = torch.argmax(Query_prob)
    result_to_show = None
    if predicted_class == 1:
        result_to_show = 'AR-AGONIST'
    else:
        result_to_show = 'NOT AR-AGONIST'
    
    return render_template('index.html', prediction_text='Query chemical is {}'.format(result_to_show))


if __name__ == "__main__":
    app.run(debug=True)
