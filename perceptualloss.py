import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from style_transfer_perceptual_loss.image_dataset import get_transform
from src.utils.train_utils import get_device


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


class PerceptualLoss:
    def __init__(self, args):
        self.content_layer = args.content_layer
        device = get_device(args)
        self.vgg = nn.DataParallel(Vgg16())
        self.vgg.eval()
        self.mse = nn.DataParallel(nn.MSELoss())
        self.mse_sum = nn.DataParallel(nn.MSELoss(reduction='sum'))
        style_image = Image.open(args.style_image).convert('RGB')
        _, transform = get_transform(args)
        style_image = transform(style_image).repeat(args.batch_size, 1, 1, 1).to(device)

        with torch.no_grad():
            self.style_features = self.vgg(style_image)
            self.style_gram = [gram(fmap) for fmap in self.style_features]
        pass

    def __call__(self, x, y_hat):
        b, c, h, w = x.shape
        y_content_features = self.vgg(x)
        y_hat_features = self.vgg(y_hat)

        recon = y_content_features[self.content_layer]
        recon_hat = y_hat_features[self.content_layer]
        L_content = self.mse(recon_hat, recon)

        y_hat_gram = [gram(fmap) for fmap in y_hat_features]
        L_style = 0
        for j in range(len(y_content_features)):
            _, c_l, h_l, w_l = y_hat_features[j].shape
            L_style += self.mse_sum(y_hat_gram[j], self.style_gram[j]) / float(c_l * h_l * w_l)

        L_pixel = self.mse(y_hat, x)

        # calculate total variation regularization (anisotropic version)
        # https://www.wikiwand.com/en/Total_variation_denoising
        diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
        L_tv = (diff_i + diff_j) / float(c * h * w)

        return L_content, L_style, L_pixel, L_tv