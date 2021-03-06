import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchvision.models import vgg16

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    out_filter = np.repeat(out_filter, channels, axis=3)
    return out_filter


class Blur(nn.Module):
    def __init__(self, nc):
        super(Blur, self).__init__()
        kernel = gauss_kernel(21, 3, nc)
        kernel = torch.from_numpy(kernel).permute(3, 2, 0, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, stride=1, padding=10)
        return x


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x1, x2):
        return torch.sum(torch.pow((x1 - x2), 2)).div(2 * x1.size()[0])


# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '9': "maxpool2",
            '30': "maxpool5"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss)/len(loss)

if __name__ == '__main__':
    cl = ColorLoss()

    # rgb example
    blur_rgb = Blur(3)
    img_rgb1 = torch.randn(4, 3, 40, 40)
    img_rgb2 = torch.randn(4, 3, 40, 40)
    blur_rgb1 = blur_rgb(img_rgb1)
    blur_rgb2 = blur_rgb(img_rgb2)
    mse=nn.L1Loss()
    print(mse(img_rgb1,img_rgb2))
    print(cl(blur_rgb1, blur_rgb2))

    # gray example
    blur_gray = Blur(1)
    img_gray1 = torch.randn(4, 1, 40, 40)
    img_gray2 = torch.randn(4, 1, 40, 40)
    blur_gray1 = blur_gray(img_gray1)
    blur_gray2 = blur_gray(img_gray2)
    print(cl(blur_gray1, blur_gray2))

    vgg_model = vgg16(pretrained=False).features[:50]
    # vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    criterionPer = LossNetwork(vgg_model)
    out=criterionPer(img_rgb1,img_rgb2)
    print(criterionPer(img_rgb1,img_rgb2))