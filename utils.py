import aircv as ac
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as ts
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision.transforms as ts

#crop and gen mask
def random_crop(img, size): 
    trans = ts.ToTensor()
    img = trans(img)
    n, h, w = img.size()
    # print(img.size())
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)
    part = img[:, y:y+size, x:x+size]
    mask = torch.FloatTensor(img.size()).fill_(0.0)
    ones = torch.FloatTensor(n, size, size).fill_(1.0)
    mask[:, y:y+size, x:x+size] = ones
    mask = mask*img
    # mask[:, y:y+size, x:x+size] = part
    
    return x, y, part, mask, img

def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def resize(tensor, channels, size):
    trans1 = ts.ToPILImage()
    trans2 = ts.ToTensor()
    img = trans1(tensor)
    img = img.resize((size, size))
    return trans2(img)

if __name__ == "__main__":
    # trans = ts.Compose([
    #     ts.ToTensor(),
    # ])
    # img = Image.open('/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/yl/facades/test/1.jpg')
    # # img = trans(img)
    # # print(img.size())

    # x, y, part, mask, img = random_crop(img, 100)
    
    # print(x,y,part.size(), mask.size())
    # out = torch.cat((img, mask), -2)
    # save_image(out, '/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/yl/facades/test/000.jpg')
    a = Image.open('/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/yl/facades/test/1.jpg')
    # Image._show(a)
    # a = torch.rand(3, 100, 100)
    transs = ts.ToTensor()
    a = transs(a)
    out = Resize(a, 3, 556)
    save_image(out, "/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/yl/spiral_try/out.jpg")
