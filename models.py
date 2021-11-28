# Define model classes here

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1(torch.nn.Module):
    def __init__(self): # Define all the adjustable params here
        super(CNN1, self).__init__()
        self.layer1 = nn.Linear(784,100) #784 input features, hidden_dim output features
        self.layer2 = nn.Linear(100, 10) #hidden_dim input features, 10 output features
        self.model = nn.Sequential(self.layer1, nn.ReLU(), self.layer2)#, nn.Softmax(dim=1))

    def forward(self, x):
        return self.model(x)

class CNN2(torch.nn.Module):
    def __init__(self): # Define all the adjustable params here
        super(CNN2, self).__init__()


    def forward(self, x):

        return
