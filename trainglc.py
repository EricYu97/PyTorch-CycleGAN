import torch
import torch.nn as nn
from dehazenet1 import GeneratorNet,Discriminator_global,Discriminator_local
from loss import Blur,ColorLoss,LossNetwork
from torchvision.models import vgg16
from torch.autograd import Variable
import itertools
from utils import ReplayBuffer
from torch.utils.data import DataLoader
from datasets import ImageDataset
import torchvision.transforms as transforms
epochs=100
dataroot="./dataset/RICE/"
lr=0.0002
input_channel=3
output_channel=3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
size=(128,128)
local_size=(32,32)
batchSize=16

blur_rgb=Blur(3)
cl=ColorLoss()

netGAtoB=GeneratorNet(input_channel,output_channel)
netGBtoA=GeneratorNet(input_channel,output_channel)

netD_A=Discriminator_global(input_channel)
netD_B=Discriminator_global(input_channel)

netD_A_L=Discriminator_local(input_channel)
netD_B_L=Discriminator_local(input_channel)

netGAtoB.to(device)
netGBtoA.to(device)
netD_A.to(device)
netD_B.to(device)
netD_A_L.to(device)
netD_B_L.to(device)

criterionGAN=nn.MSELoss()
criterionCycle=nn.L1Loss()
def criterionColor(pred,label):
    blur_rgb1=blur_rgb(pred)
    blur_rgb2=blur_rgb(label)
    return cl(blur_rgb1,blur_rgb2)
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False
criterionPer=LossNetwork(vgg16)
criterion_identity = torch.nn.L1Loss()

optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

optimizer_D_A_L = torch.optim.Adam(netD_A_L.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B_L.parameters(), lr=lr, betas=(0.5, 0.999))

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(batchSize,input_channel,size[0],size[1])
input_B = Tensor(batchSize,output_channel,size[0],size[1])
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                transforms.RandomCrop(opt.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=batchSize, shuffle=True, num_workers=8)

for i in range(epochs):
    for i,batch in enumerate(dataloader):
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        optimizer_G.zero_grad()
        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netGAtoB(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netGBtoA(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0
        # GAN loss
        fake_B = netGAtoB(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterionGAN(pred_fake, target_real)

        fake_A = netGBtoA(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterionGAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netGBtoA(fake_B)
        loss_cycle_ABA = criterionCycle(recovered_A, real_A)*10.0

        recovered_B = netGAtoB(fake_A)
        loss_cycle_BAB = criterionCycle(recovered_B, real_B)*10.0

        #PerLoss
        loss_per_ABA=criterionPer(recovered_A,real_A)*10.0
        loss_per_BAB=criterionPer(recovered_B,real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB +loss_per_ABA +loss_per_BAB
        loss_G.backward()

        optimizer_G.step()

        """Train D"""
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterionGAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterionGAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterionGAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterionGAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        """Local"""
        """将一张输出裁切为多张局部输出计算loss"""

    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netGAtoB.state_dict(), 'output/netG_A2B.pth')
    torch.save(netGBtoA.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')




