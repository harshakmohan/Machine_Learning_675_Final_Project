# Define model classes here

import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
gc.collect()
torch.cuda.empty_cache()

class CNN1(torch.nn.Module):
    def __init__(self, n1_chan = 80, n1_kern=10, n2_kern=5): # Define all the adjustable params here
        super(CNN1, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=n1_chan, kernel_size=n1_kern, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=n1_chan, out_channels=4, kernel_size=n2_kern, stride=2)
        self.pooling = torch.nn.MaxPool2d(kernel_size=8)
        self.cnn1 = None

    def forward(self, x):
        #print('x on cuda before? ', x.device)
        x = x.reshape(x.size()[0], 1, 496, 496)
        #print('x on cuda after? ', x.device)
        self.cnn1 = nn.Sequential(self.conv1, nn.ReLU(inplace=True), self.conv2, nn.ReLU(inplace=True), self.pooling)
        return self.cnn1(x)

class CNN2(torch.nn.Module):
    def __init__(self): # Define all the adjustable params here
        super(CNN2, self).__init__()


    def forward(self, x):

        return
