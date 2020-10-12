import torch
from networks import *
from vgg import Vgg16 
import torch.nn
from torch.utils.data import Dataset, DataLoader
import config
from data import dataset
from torchvision.transforms import transforms
from torchvision.utils import save_image
from utils import *
import random
import os


cfg = config.Config
path = cfg.rootpath

random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed(cfg.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.cuda.set_device(cfg.cuda)

transform = transforms.Compose([
    # transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])

#dataset and loader
dataset = dataset(path,transform=transform)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

gen = Generator(cfg.channels, cfg.dim, cfg.num_resnet).cuda()
dis = Discriminator(cfg.channels, cfg.dim).cuda()

optimizer_G = torch.optim.Adam(gen.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
optimizer_D = torch.optim.Adam(dis.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))

if not os.path.exists('/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/xyl/output/checkpoints/celeba_imangine/dict'):
    gen.apply(weights_init_normal)
    dis.apply(weights_init_normal)
    start_epoch = 0
else:
    checkpoint = torch.load('/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/xyl/output/checkpoints/celeba_imangine/dict')
    gen.load_state_dict(checkpoint['gen_state_dict'])
    dis.load_state_dict(checkpoint['dis_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_d_state_dict'])
    start_epoch = checkpoint['epoch']
    print("start_epoch:", start_epoch)



valid = torch.ones(cfg.batch_size, 1, cfg.patch_size, cfg.patch_size).cuda()
fake = torch.zeros(cfg.batch_size, 1, cfg.patch_size, cfg.patch_size).cuda()

vgg = Vgg16().cuda()

mse_loss = nn.MSELoss().cuda()
# pix_loss = nn.L1Loss().cuda()


gen.train()
for epoch in range(start_epoch, cfg.epochs):
    accu_style_loss = 0.0
    accu_content_loss = 0.0
    accu_var_loss = 0.0

    # torch.save({
    #         'gen_state_dict': gen.state_dict(),
    #         'dis_state_dict': dis.state_dict(),
    #         'optimizer_g_state_dict': optimizer_G.state_dict(),
    #         'optimizer_d_state_dict': optimizer_D.state_dict(),
    #         'epoch': epoch,

    #         }, '/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/xyl/output/checkpoints/celeba_imangine/dict')
    

    for i, (x, y, part, mask, img) in enumerate(dataloader):###############

        # img_batch_read = len(img)
        # print("~~", img_batch_read)
        # break
        # save_image(mask, '/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/yl/spiral_try/%d.jpg'%i)
        # print(mask.size())
        z = torch.rand(cfg.batch_size, 1,  cfg.img_size, cfg.img_size).cuda()

        part = part.cuda()
        mask = mask.cuda()
        img = img.cuda()
        optimizer_G.zero_grad()
        fake_img = gen(mask, part, z)
        #calculate g_loss
        g_loss = mse_loss(dis(fake_img, mask), valid)
        #calculate style_loss
        real_features = vgg(img)
        real_gram = [gram(fmap) for fmap in real_features]
        fake_features = vgg(fake_img)
        fake_gram = [gram(fmap) for fmap in fake_features]
        style_loss = 0.0
        for j in range(4):
            style_loss += mse_loss(fake_gram[j], real_gram[j])
            # style_loss += loss_mse(fake_img_gram[j], style_gram[j][:img_batch_read])#with no style,use input img
        style_loss = style_loss * cfg.style_loss_weight
        # print(style_loss.item())
        # break
        accu_style_loss += style_loss.item()
        #calculate content_loss(2nd layer)
        real_2_feature = real_features[1]
        fake_2_feature = fake_features[1]
        content_loss = cfg.content_loss_weight * mse_loss(real_2_feature, fake_2_feature)
        accu_content_loss += content_loss.item()
        #calculate total variation regularization
        diff_i = torch.sum(torch.abs(fake_img[:, :, :, 1:] - fake_img[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(fake_img[:, :, 1:, :] - fake_img[:, :, :-1, :]))
        var_loss = cfg.var_loss_weight * (diff_i + diff_j)
        accu_var_loss += var_loss.item()

        G_loss = g_loss + style_loss + content_loss + var_loss
        G_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        D_loss_real = mse_loss(dis(img, mask), valid)
        D_loss_fake = mse_loss(dis(fake_img.detach(), mask), fake)
        D_loss = (D_loss_fake + D_loss_real) / 2
        D_loss.backward()
        optimizer_D.step()


        print("[Epoch %d/%d] [Batch %d/%d] [G_loss %f] [D_loss %f]" %(epoch, cfg.epochs, i, len(dataloader), G_loss, D_loss))
        batchesdone = epoch * len(dataloader) + i
        if batchesdone % cfg.sample_interval == 0:
            torch.save({
            'gen_state_dict': gen.state_dict(),
            'dis_state_dict': dis.state_dict(),
            'optimizer_g_state_dict': optimizer_G.state_dict(),
            'optimizer_d_state_dict': optimizer_D.state_dict(),
            'epoch': epoch,

            }, '/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/xyl/output/checkpoints/celeba_imangine/dict')
            out = torch.cat((img, mask, fake_img), 2)
            save_image(out, "/media/s/c9c8cd8e-a5ab-4592-b6f1-8ea96609bf5d/xyl/output/imgs/train/celeba_imangine/%d.jpg"%batchesdone, nrow=4, normalize=False)

