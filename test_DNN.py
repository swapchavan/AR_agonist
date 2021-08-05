import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_DNN(my_model, test_df):
    # testing
    pred_class_for_epoch_val = np.empty((0), int)
    pred_prob_for_epoch_val = np.empty((0,2), int)
    
    my_model.eval()
    # turn off gradients for validation
    with torch.no_grad():
        #batch_X_val = torch.tensor(test_df.values).type(torch.FloatTensor)
        batch_X_val = torch.tensor(test_df.values).to(dtype=torch.float32)
        outputs = my_model.forward(batch_X_val)
        # pred porb
        predicted_prob_val = my_model.predict_prob(outputs)
    
    return predicted_prob_val