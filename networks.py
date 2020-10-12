import torch.nn as nn
import torch
from spectral_normalization import SpectralNorm

def weights_init_normal(m):

    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        pass
        # torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Resnetblock(nn.Module):

    def __init__(self, in_channels):

        super(Resnetblock, self).__init__()
        self.in_channels = in_channels

        model = [nn.ReflectionPad2d(1), nn.Conv2d(in_channels, in_channels, 3, 1, 1, dilation=2), nn.InstanceNorm2d(in_channels)]
        
        model += [nn.ReflectionPad2d(1), nn.Conv2d(in_channels, in_channels, 3), nn.ReLU(), nn.InstanceNorm2d(in_channels)]
        self.model = nn.Sequential(
            *model
        )
    def forward(self, x):

        return x + self.model(x)


class Generator(nn.Module):

    def __init__(self, channels, dim, num_resnet):

        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels*2+1, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]

        for _ in range(2):
            model += [
                nn.Conv2d(dim, 2*dim, 3, stride=2, padding=1),
                nn.InstanceNorm2d(2*dim),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        for _ in range(num_resnet):
            model += [Resnetblock(dim)] 
        
        for _ in range(2):
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, dim//2, 3, stride=1, padding=1),
                nn.InstanceNorm2d(dim//2),
                nn.ReLU(inplace=True),
            ]
            dim //= 2
        
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(dim, channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, part, mask, z):

        x = torch.cat((part, mask, z), 1)
        return self.model(x)



class Discriminator(nn.Module):

    def __init__(self, channels, dim):
        super(Discriminator, self).__init__()
        # model = [nn.Conv2d(channels, 8*dim, 3, 1, 1)]
        def dis_block(in_channels, out_channels, norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
            # if norm:
                # layers.append(nn.BatchNorm2d(out_channels))
            # layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers
        self.model = nn.Sequential(

            
            SpectralNorm(*dis_block(channels*2, dim, False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(*dis_block(dim, 2*dim, True)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(*dis_block(2*dim, 4*dim, True)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(*dis_block(4*dim, 8*dim, True)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(8*dim, 1, 4, 1, 1, bias=False),
            # nn.utils.spectral_norm()
        )

    def forward(self, x, y):

        x = torch.cat((x,y), 1)
        return self.model(x)

# aa = Discriminator(3, 64)
# print(aa)
class slice_Generator():
    pass
class slice_Discriminator():
    pass
class extra_Discriminator():
    pass
        

