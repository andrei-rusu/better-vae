import argparse
import os
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import visdom
import utils
import itertools

vis = visdom.Visdom(port=6007)
vis.env = 'vae_dcgan'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--saveInt', type=int, default=25, help='number of epochs between checkpoints')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--limit', type=int, default=100, help='limit iterations per epoch')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _Sampler(nn.Module):
    def __init__(self):
        super(_Sampler, self).__init__()
        
    def forward(self,input):
        mu = input[0]
        logvar = input[1]
        
        std = logvar.mul(0.5).exp_() #calculate the STDEV
        if opt.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_() #random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_() #random normalized noise
        return eps.mul(std).add_(mu) 


class _Encoder(nn.Module):
    def __init__(self,imageSize):
        super(_Encoder, self).__init__()
        
        n = math.log2(imageSize)
        
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)


        self.conv1 = nn.Conv2d(ngf * 2**(n-3), nz, 4)
        self.conv2 = nn.Conv2d(ngf * 2**(n-3), nz, 4)

        self.encoder = nn.Sequential()
        # input is (nc) x 64 x 64
        self.encoder.add_module('input-conv',nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('input-relu',nn.LeakyReLU(0.2, inplace=True))
        for i in range(n-3):
            # state size. (ngf) x 32 x 32
            self.encoder.add_module('pyramid_{0}-{1}_conv'.format(ngf*2**i, ngf * 2**(i+1)), nn.Conv2d(ngf*2**(i), ngf * 2**(i+1), 4, 2, 1, bias=False))
            self.encoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2**(i+1)), nn.BatchNorm2d(ngf * 2**(i+1)))
            self.encoder.add_module('pyramid_{0}_relu'.format(ngf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf*8) x 4 x 4

    def forward(self,input):
        output = self.encoder(input)
        return [self.conv1(output),self.conv2(output)]


class _netG(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.encoder = _Encoder(imageSize)
        self.sampler = _Sampler()
        
        n = math.log2(imageSize)
        
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)

        self.decoder = nn.Sequential()
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2**(n-3), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2**(n-3)))
        self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(n-3, 0, -1):
            self.decoder.add_module('pyramid_{0}-{1}_conv'.format(ngf*2**i, ngf * 2**(i-1)),nn.ConvTranspose2d(ngf * 2**i, ngf * 2**(i-1), 4, 2, 1, bias=False))
            self.decoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2**(i-1)), nn.BatchNorm2d(ngf * 2**(i-1)))
            self.decoder.add_module('pyramid_{0}_relu'.format(ngf * 2**(i-1)), nn.LeakyReLU(0.2, inplace=True))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False))
        self.decoder.add_module('output-tanh', nn.Tanh())


    def forward(self, input):
        if isinstance(input, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            output = nn.parallel.data_parallel(self.sampler, output, range(self.ngpu))
            output = nn.parallel.data_parallel(self.decoder, output, range(self.ngpu))
        else:
            output = self.encoder(input)
            output = self.sampler(output)
            output = self.decoder(output)
        return output
    
    def make_cuda(self):
        self.encoder.cuda()
        self.sampler.cuda()
        self.decoder.cuda()

netG = _netG(opt.imageSize,ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        n = math.log2(imageSize)
        
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)
        self.main = nn.Sequential()

        # input is (nc) x 64 x 64
        self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.main.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ndf) x 32 x 32
        for i in range(n-3):
            self.main.add_module('pyramid_{0}-{1}_conv'.format(ngf*2**(i), ngf * 2**(i+1)), nn.Conv2d(ndf * 2 ** (i), ndf * 2 ** (i+1), 4, 2, 1, bias=False))
            self.main.add_module('pyramid_{0}_batchnorm'.format(ngf * 2**(i+1)), nn.BatchNorm2d(ndf * 2 ** (i+1)))
            self.main.add_module('pyramid_{0}_relu'.format(ngf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))

        self.main.add_module('output-conv', nn.Conv2d(ndf * 2**(n-3), 1, 4, 1, 0, bias=False))
        self.main.add_module('output-sigmoid', nn.Sigmoid())
        

    def forward(self, input):
        if isinstance(input, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1)
    
def prepare_img(img):
    inp = img.cpu()
    inp = inp.detach().numpy().transpose((0, 2, 3, 1))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = inp.transpose((0, 3, 1, 2))
    return inp

netD = _netD(opt.imageSize,ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()
MSECriterion = nn.MSELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.make_cuda()
    criterion.cuda()
    MSECriterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
optimizerEnc = optim.Adam(netG.encoder.parameters(), lr=2e-4, betas=(opt.beta1, 0.999))
optimizerDec = optim.Adam(netG.decoder.parameters(), lr=2e-4, betas=(opt.beta1, 0.999))
optimizerDis = optim.Adam(netD.parameters(), lr=1e-4, betas=(opt.beta1, 0.999))

gen_win = None
rec_win = None

for epoch in range(opt.niter):
    for i, data in enumerate(itertools.islice(dataloader, opt.limit)):
        
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        input.resize_(real_cpu.size()).copy_(real_cpu)
        label.resize_(batch_size)
        
        recon_modules = list(netD.main.modules())[1:7]
        
        ############################
        # (1) Update G network: Encoder
        ############################
        netG.zero_grad()
        netD.zero_grad()
        
        encoded = netG.encoder(input)
        mu = encoded[0]
        logvar = encoded[1]
        
        KLD = utils.kld(mu, logvar)
        
        sampled = netG.sampler(encoded)
        recons = netG.decoder(sampled)
        
        # Reconstruction loss through Discriminator l-depth
        inp = input
        rec = recons
        for mod in recon_modules:
            inp = mod(inp)
            rec = mod(rec)
        err_recons = F.mse_loss(rec, inp.detach(), reduction='sum') / input.shape[0]
        
        gamma = 1
        VAEerr = KLD + gamma * err_recons
        
        VAEerr.backward()
        optimizerEnc.step()
        
        ############################
        # (2) Update G network: maximize log(D(G(z))) - Decoder
        ############################

        netG.zero_grad()
        netD.zero_grad()
        
        recons2 = netG(input)
        disc_recons = netD(recons2)
        
        D_G_z2 = disc_recons.mean()
        
        # Reconstruction loss through Discriminator l-depth
        inp2 = input
        rec2 = recons2
        for mod in recon_modules:
            inp2 = mod(inp2)
            rec2 = mod(rec2)
        err_recons2 = F.mse_loss(rec2, inp2.detach(), reduction='sum') / input.shape[0]
        
        label.fill_(real_label)  # fake labels are real for generator cost
        errG_gan = criterion(disc_recons, label)
        
        gamma = 1e-4
        errG = errG_gan + gamma * err_recons2
        
        errG.backward()
        optimizerDec.step()
        
        ############################
        # (3) Update D network: maximize log(D(x)) + log(1 - D(G(z))) - Discriminator
        ###########################
        netG.zero_grad()
        netD.zero_grad()
        
        label = label.clone()
        label.fill_(real_label)  # real labels

        # train with real
        disc_inp = netD(input)
        errD_real = criterion(disc_inp, label)
        D_x = disc_inp.mean()
    
        label = label.clone()
        label.fill_(fake_label)  # fake labels

        # train with fake
        noise.resize_(batch_size, nz, 1, 1)
        noise.normal_(0, 1)
        gen = netG.decoder(noise)
        disc_gen = netD(gen.detach())
        
        errD_fake = criterion(disc_gen, label)
        D_G_z1 = disc_gen.mean()
        
        errD = errD_real + errD_fake
        errD.backward()
        optimizerDis.step()
        
        # Log original MSE reconstruction loss
        with torch.no_grad():
            MSEerr_f = F.mse_loss(recons, input, reduction='sum') / input.shape[0]
        
        if (i % 50 == 0):
            rec_win = vis.images(prepare_img(recons), win = rec_win, opts=dict(title='Reconstructed Images'))
            gen_win = vis.images(prepare_img(gen), win = gen_win, opts=dict(title='Generated Images'))
            
        utils.debug(KLD, err_recons, errG_gan, gamma * err_recons2)
            
        print('[%d/%d][%d/%d] MSE: %.4F Loss_VAE: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 MSEerr_f.item(), VAEerr.item(), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        
        del errG, errD, VAEerr, MSEerr_f, D_x, D_G_z1, D_G_z2, 

    if epoch%opt.saveInt == 0 and epoch!=0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
