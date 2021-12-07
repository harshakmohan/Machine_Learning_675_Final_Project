# Define model classes here

import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
gc.collect()
torch.cuda.empty_cache()

class CNN1(torch.nn.Module):
    def __init__(self, n1_chan = 80, n1_kern=5, n2_kern=10): # Define all the adjustable params here
        super(CNN1, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=n1_chan, kernel_size=n1_kern, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=n1_chan, out_channels=4, kernel_size=n2_kern, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.flatten1 = nn.Flatten()
        self.linear1 = torch.nn.Linear(in_features=4, out_features=4)
        self.pooling = torch.nn.MaxPool2d(kernel_size=8)
        self.cnn1 = None

    def forward(self, x):
        #print('x on cuda before? ', x.device)
        x = x.reshape(x.size()[0], 1, 496, 496)
        #print('x on cuda after? ', x.device)
        #self.cnn1 = nn.Sequential(self.conv1, nn.ReLU(inplace=True), self.conv2, nn.ReLU(inplace=True), self.pooling)
        #self.cnn1 = nn.Sequential(self.conv1, nn.ReLU(inplace=True), self.conv2, self.flatten1, self.linear1, self.pooling)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.flatten1(x)
        print('size of x after to flatten: ', x.size(-1))
        x = torch.nn.Linear(in_features=x.size(-1), out_features=4)(x)
        x = self.pooling(x)


        return x

class CNN2(torch.nn.Module):
    def __init__(self): # Define all the adjustable params here
        super(CNN2, self).__init__()


    def forward(self, x):

        return
