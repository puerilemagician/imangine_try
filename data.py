from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
from utils import *
import config
import aircv as ac
import torchvision.transforms as ts
from torchvision.utils import save_image
import torch

cfg = config.Config

class dataset(Dataset):


    def __init__(self, rootpath, transform=None, size=cfg.part_size):

        super(dataset, self).__init__()
        self.rootpath = rootpath
        self.imgs_list = glob.glob(rootpath+'/*.*')
        self.size = size
        self.transform = transform

    def __getitem__(self, index):

        img = Image.open(self.imgs_list[index % len(self.imgs_list)])
        x, y, part, mask, img = random_crop(img, self.size)
        part = resize(part, cfg.channels, cfg.img_size)
        # part = part.resize(cfg.channels, cfg.part_size, cfg.part_size)
        
        # part = part.reshape(cfg.channels, cfg.img_size, cfg.img_size)
        return x, y, part, mask, img
    
    def __len__(self):

        return len(self.imgs_list)



if __name__ =="__main__":
    transform = ts.Compose([
        ts.ToTensor(),
        # trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    rootpath = '/media/disk/gds/data/CelebA-HQ/celeba-256'

    dataset_facades = dataset(rootpath, transform=transform)
    # print(dataset_facades[0][0])
    dataloader = DataLoader(dataset_facades, batch_size=4)
    for i, (x, y, part, mask, img) in enumerate(dataloader):
        print(x, y, part.size(), img.size())
        # a = x[0]
        # print(a.item())
        # print(type(mask), type(img))
        out = torch.cat((img, mask), -1)
        # save_image(mask, "/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/yl/spiral_try/masksssssssss%d.jpg" % i)
        save_image(out, "/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/yl/spiral_try/mask-img%d.jpg"% i,nrow=2)
        print('tohere')
        # break

        