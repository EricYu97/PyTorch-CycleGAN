import torch
import torch.nn as nn
from torchvision.models import vgg16

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
    img_rgb1 = torch.randn(4, 3, 40, 40)
    img_rgb2 = torch.randn(4, 3, 40, 40)
    vgg_model = vgg16(pretrained=False).features[:50]
    # vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    criterionPer = LossNetwork(vgg_model)
    out = criterionPer(img_rgb1, img_rgb2)