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
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(in_channels,in_channels,3),
                                     nn.InstanceNorm2d(in_channels)
                                     ])
        self.res_num=res_num
    def forward(self,inputs):
        x=inputs
        for _ in range(self.res_num):
            for block in self.resblock:
                x=block(x)
        return x



class GeneratorNet(nn.Module):
    def __init__(self,input_channel=3,output_channel=3):
        super(GeneratorNet,self).__init__()
        self.refpad=nn.ReflectionPad2d(3)
        self.conv1=nn.Conv2d(input_channel,64,7)
        self.insnorm64=nn.InstanceNorm2d(64)
        self.relu=nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(64,128,3,padding=1,stride=2)
        self.insnorm128=nn.InstanceNorm2d(128)
        self.conv3=nn.Conv2d(128,256,3,padding=1,stride=2)
        self.insnorm256=nn.InstanceNorm2d(256)
        self.resblocks=resblock(256,6)

        self.deconv1=nn.ConvTranspose2d(256,128,3,stride=2,padding=1,output_padding=1)
        self.deconv2=nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1)
        self.deconv3=nn.ConvTranspose2d(64,output_channel,7,stride=1,padding=3,output_padding=0)
        self.conv4=nn.Conv2d(64,output_channel,7,stride=1)
        self.tanh=nn.Tanh()
    def forward(self,inputs):
        """conv1 layer"""
        x=self.refpad(inputs)
        x=self.conv1(x)
        x=self.insnorm64(x)
        x=self.relu(x)
        conv1_out=x
        """conv2+conv3"""
        x=self.conv2(x)
        x=self.insnorm128(x)
        x=self.relu(x)
        conv2_out=x

        x=self.conv3(x)
        x=self.insnorm256(x)
        x=self.relu(x)
        conv3_out=x

        """resblock"""
        x=self.resblocks(x)

        x=x+conv3_out

        """deconv1"""
        x=self.deconv1(x)
        x=self.insnorm128(x)
        x=self.relu(x)

        x=x+conv2_out

        """deconv2"""
        x=self.deconv2(x)
        x=self.insnorm64(x)
        x=self.relu(x)

        x=x+conv1_out

        """deconv3"""
        x=self.refpad(x)
        x=self.conv4(x)
        x=self.tanh(x)

        # x=self.deconv3(x)
        # x=self.relu(x)

        return x

class Discriminator_global(nn.Module):
    def __init__(self,input_channel=3):
        super(Discriminator_global,self).__init__()
        self.conv1=nn.Conv2d(input_channel,64,4,stride=2,padding=1)
        self.conv2=nn.Conv2d(64,128,4,stride=2,padding=1)
        self.conv3=nn.Conv2d(128,256,4,stride=2,padding=1)
        self.conv4=nn.Conv2d(256,512,4,stride=2,padding=1)
        self.lkrelu=nn.LeakyReLU(0.2,inplace=True)
        self.insnorm=nn.ModuleList([nn.InstanceNorm2d(128,affine=False),
                                    nn.InstanceNorm2d(256,affine=False),
                                    nn.InstanceNorm2d(512,affine=False)])
        self.conv5=nn.Conv2d(512,1,4,padding=1)

    def forward(self,inputs):
        x=self.conv1(inputs)
        x=self.lkrelu(x)
        x=self.conv2(x)
        x=self.insnorm[0](x)
        x=self.lkrelu(x)
        x=self.conv3(x)
        x=self.insnorm[1](x)
        x=self.lkrelu(x)
        x=self.conv4(x)
        x=self.insnorm[2](x)
        x=self.lkrelu(x)
        x=self.conv5(x)
        return x

class Discriminator_local(nn.Module):
    def __init__(self,input_channel=3):
        super(Discriminator_local,self).__init__()
        self.conv1=nn.Conv2d(input_channel,64,4,stride=2,padding=1)
        self.conv2=nn.Conv2d(64,128,4,stride=2,padding=1)
        self.conv3=nn.Conv2d(128,256,4,stride=2,padding=1)
        self.conv4=nn.Conv2d(256,512,4,stride=2,padding=1)
        self.lkrelu=nn.LeakyReLU(0.2,inplace=True)
        self.insnorm=nn.ModuleList([nn.InstanceNorm2d(128,affine=False),
                                    nn.InstanceNorm2d(256,affine=False),
                                    nn.InstanceNorm2d(512,affine=False)])
        self.conv5=nn.Conv2d(512,1,4,padding=1)

    def forward(self,inputs):
        x=self.conv1(inputs)
        x=self.lkrelu(x)
        x=self.conv2(x)
        x=self.insnorm[0](x)
        x=self.lkrelu(x)
        x=self.conv3(x)
        x=self.insnorm[1](x)
        x=self.lkrelu(x)
        x=self.conv4(x)
        x=self.insnorm[2](x)
        x=self.lkrelu(x)
        x=self.conv5(x)
        return x

if __name__ == '__main__':
    model=Discriminator_global()
    print(model)
    x=torch.rand(4,3,32,32)
    print(x.shape)
    y=model(x)
    print(y.shape)