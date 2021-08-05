import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, initial_descriptors=196, n_hid_lay=2, neurons=256, dropout=0.2):
        super().__init__()
        self.name = "DNN"
        self.init_args = {"n_hid_lay": n_hid_lay, "neurons": neurons}
        
        self.input_layer = nn.Linear(initial_descriptors, neurons)
        self.fc = nn.ModuleList()
        self.fc.append(nn.BatchNorm1d(neurons))
        self.fc.append(nn.Dropout(p=dropout))

        for i in range(n_hid_lay):
            if i < n_hid_lay:
                self.fc.append(nn.Linear(neurons, neurons))
                self.fc.append(nn.BatchNorm1d(neurons))
                self.fc.append(nn.Dropout(p=dropout))
            elif i == n_hid_lay:
                self.fc.append(nn.Linear(neurons, neurons))
                self.fc.append(nn.BatchNorm1d(neurons))
        self.classifier = nn.Linear(neurons, 2)
        
    def forward(self, input):
        x = self.input_layer(input)
        for i, l in enumerate(self.fc):
            x = F.relu(self.fc[i](x))
        x = self.classifier(x)
        return x
    
    def predict_class(self, output):
        pred = F.softmax(output)
        ans = pred.argmax(1)
        return torch.tensor(ans)
    
    def predict_prob(self, output): 
        ans_prob = F.softmax(output)
        return torch.tensor(ans_prob)