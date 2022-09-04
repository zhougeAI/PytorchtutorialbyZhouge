import torch
import torch.nn as nn

# 把重复的层封装起来，层的集合
class Conv_layer(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding,use_pooling=True):
        super().__init__()
        self.conv = nn.Conv2d( in_channels= in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d((2,2))
        self.usepooling = use_pooling
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.usepooling:
            output = self.pool(x)
        else:
            output = (x)
        return output


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # conv1
        self.conv1 = Conv_layer(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        # conv2
        self.conv2 = Conv_layer(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0)
        # conv3 
        self.conv3 = Conv_layer(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0)
        # conv4
        self.conv4 = Conv_layer(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=0,use_pooling=False)
        # linear
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=12800,out_features=100)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        output = self.dense(x)
        return output


class ConvTranspose_layer(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.ConvTranspose2d( in_channels= in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        output = (x)
        return output

if __name__ == '__main__':
    pass