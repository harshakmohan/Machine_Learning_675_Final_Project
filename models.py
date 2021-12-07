# Define model classes here

import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
gc.collect()
torch.cuda.empty_cache()

class CNN1(torch.nn.Module):
    def __init__(self, n1_chan = 80, n1_kern=6, n2_kern=10): # Define all the adjustable params here
        super(CNN1, self).__init__()
        self.img = torch.zeros((1, 1, 64,64))
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=n1_chan, kernel_size=n1_kern, stride=1, padding=(n1_kern//2,n1_kern//2))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=n1_chan, out_channels=4, kernel_size=n2_kern, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.flatten1 = nn.Flatten()
        self.pooling = torch.nn.MaxPool2d(kernel_size=8)


        test = self.conv1(self.img)
        test = self.relu1(test)
        test= self.conv2(test)
        test = self.pooling(test)
        test= self.relu2(test)
        test = self.flatten1(test) # test.size(-1)

        self.linear1 = torch.nn.Linear(in_features=test.size(-1), out_features=4)
        self.cnn1 = None

    def forward(self, x):
        #print('x on cuda before? ', x.device)
        print('size of x before reshape: ', x.size())
        x = x.reshape(x.size()[0], 1, 64, 64)
        print('size of x after reshape: ', x.size())
        #print('x on cuda after? ', x.device)
        #self.cnn1 = nn.Sequential(self.conv1, nn.ReLU(inplace=True), self.conv2, nn.ReLU(inplace=True), self.pooling)
        #self.cnn1 = nn.Sequential(self.conv1, nn.ReLU(inplace=True), self.conv2, self.flatten1, self.linear1, self.pooling)
        x = self.conv1(x)
        print('size of x after conv1: ', x.size())
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pooling(x)
        print('size of x before flatten: ', x.size())
        x = self.flatten1(x)
        print('size of x after to flatten: ', x.size())
        x = self.linear1(x)
        print('size of x after linear layer: ', x.size())
        #x = self.pooling(x)
        return x

class CNN2(torch.nn.Module):
    def __init__(self): # Define all the adjustable params here
        super(CNN2, self).__init__()


    def forward(self, x):

        return
