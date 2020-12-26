"""encoder of GLC GAN"""
"""copy right by Eric Yu"""
import torch
import torch.nn as nn

class resblock(nn.Module):
    def __init__(self,in_channels,res_num):
        super(resblock,self).__init__()
        self.resblock=nn.ModuleList([nn.ReflectionPad2d(1),
                                     nn.Conv2d(in_channels,in_channels,3),
                                     nn.InstanceNorm2d(in_channels),
                                     ])

class encoderNet(nn.Module):
    def __init__(self,input_channel=3,output_channel=3):
        super(encoderNet,self).__init__()
        self.refpad=nn.ReflectionPad2d(3)
        self.conv1=nn.Conv2d(input_channel,64,7)
        self.insnorm=nn.InstanceNorm2d(64)
        self.relu=nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(64,128,3,padding=1,stride=2)
        self.conv3=nn.Conv2d(128,256,3,padding=1,stride=2)
    def forward(self,inputs):
        """conv1 layer"""
        x=self.refpad(inputs)
        x=self.conv1(x)
        x=self.insnorm(x)
        x=self.relu(x)
        """conv2+conv3"""
        x=self.conv2(x)
        x=self.insnorm(x)
        x=self.relu(x)

        x=self.conv3(x)
        x=self.insnorm(x)
        x=self.relu(x)
        return x

if __name__ == '__main__':
    model=encoderNet()
    print(model)
    x=torch.rand(4,3,128,128)
    print(x.shape)
    y=model(x)
    print(y.shape)