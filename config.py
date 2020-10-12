
class Config(object):
    part_size = 128
    img_size = 256
    rootpath = '/media/disk/gds/data/CelebA-HQ/celeba-256'
    cuda = 0
    batch_size = 4
    mode = 'train' #test
    start_epoch = 0
    lr = 2e-5
    beta1 = 0.5
    patch_size = 16
    epochs = 200
    channels = 3
    dim = 64
    num_resnet = 8
    style_loss_weight = 1e5
    content_loss_weight = 1e0
    var_loss_weight = 1e-7
    sample_interval = 100     #interval to sample images
    seed = 3
    save_interval = 5             #interval to save model
