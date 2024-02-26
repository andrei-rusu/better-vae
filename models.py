# VAE models
# Credits to: Jonathon Hare

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import utils, layers
import torchbearer as tb

from tensorboard_logging import ReconstructionsLogger, TensorBoardModelLogger, LatentSpaceReconLogger, RandomReconLogger
from torchbearer import Trial
from torchbearer.callbacks.tensor_board import TensorBoard
from torchbearer import state_key
from utils import AEDatasetWrapper

LOSS_E = state_key('encoder_loss')
LOSS_D = state_key('decoder_loss')


# PixelCNN modules

class CGated(nn.Module):

    def __init__(self, input_size, cond_size, channels=63, num_layers=6, k=7, padding=3):
        super().__init__()

        c, h, w = input_size

        self.conv1 = nn.Conv2d(c, channels, 1, groups=c)

        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gated_layers.append(
                layers.CMaskedConv2d(
                    (channels, h, w),
                    cond_size,
                    channels, colors=c, self_connection=i > 0,
                    res_connection= i > 0,
                    gates=True,
                    hv_connection=True,
                    k=k, padding=padding)
            )

        self.conv2 = nn.Conv2d(channels, 256*c, 1, groups=c)

    def forward(self, x, cond):

        b, c, h, w = x.size()

        x = self.conv1(x)

        xh, xv = x, x

        for layer in self.gated_layers:
            xv, xh = layer(xv, xh, cond)

        x = self.conv2(xh)

        return x.view(b, c, 256, h, w).transpose(1, 2)
    
class LGated(nn.Module):

    def __init__(self, input_size, conditional_channels, channels=63, num_layers=6, k=7, padding=3):
        super().__init__()

        c, h, w = input_size

        self.conv1 = nn.Conv2d(c, channels, 1, groups=c)

        self.gated_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gated_layers.append(
                layers.LMaskedConv2d(
                    (channels, h, w),
                    conditional_channels,
                    channels, colors=c, self_connection=i > 0,
                    res_connection= i > 0,
                    gates=True,
                    hv_connection=True,
                    k=k, padding=padding)
            )

        self.conv2 = nn.Conv2d(channels, 256*c, 1, groups=c)

    def forward(self, x, cond):

        b, c, h, w = x.size()

        x = self.conv1(x)

        xh, xv = x, x

        for layer in self.gated_layers:
            xv, xh = layer(xv, xh, cond)

        x = self.conv2(xh)

        return x.view(b, c, 256, h, w).transpose(1, 2)


# Modified VAE networks
class ImEncoder(nn.Module):

    """
    Encoder for a VAE
    """
    def __init__(self, in_size, zsize=32, use_res=False, use_bn=False, depth=0, colors=3):
        a, b, c = 16, 64, 128  # channel sizes
        p, q, r = 2, 2, 2  # up/downsampling

        super().__init__()
        self.zsize = zsize

        # - Encoder
        modules = [
            layers.Block(colors, a, use_res=use_res, batch_norm=use_bn),
            nn.MaxPool2d((p, p)),
            layers.Block(a, b, use_res=use_res, batch_norm=use_bn),
            nn.MaxPool2d((q, q)),
            layers.Block(b, c, use_res=use_res, batch_norm=use_bn),
            nn.MaxPool2d((r, r)),
        ]

        for i in range(depth):
            modules.append(layers.Block(c, c, use_res=use_res, batch_norm=use_bn))

        modules.extend([
            layers.Flatten(),
            nn.Linear((in_size[0] // (p*q*r)) * (in_size[1] //  (p*q*r)) * c, zsize * 2)
        ])

        self.encoder = nn.Sequential(*modules)

    def forward(self, image):

        zcomb = self.encoder(image)
        return zcomb[:, :self.zsize], zcomb[:, self.zsize:]

class ImDecoder(nn.Module):
    """
    Decoder for a VAE
    """
    def __init__(self, in_size, zsize=32, use_res=False, use_bn=False, depth=0, out_channels=3):
        super().__init__()

        a, b, c = 60, 64, 128  # channel sizes
        p, q, r = 2, 2, 2  # up/downsampling

        self.zsize = zsize

        #- Decoder
        upmode = 'bilinear'
        modules = [
            nn.Linear(zsize, (in_size[0] // (p*q*r)) * (in_size[1] // (p*q*r)) * c), nn.ReLU(),
            layers.Reshape((c, in_size[0] // (p*q*r), in_size[1] // (p*q*r)))
        ]

        for _ in range(depth):
            modules.append(layers.Block(c, c, deconv=True, use_res=use_res, batch_norm=use_bn) )


        modules.extend([
            layers.Interpolate(scale_factor=r, mode=upmode),
            layers.Block(c, c, deconv=True, use_res=use_res, batch_norm=use_bn),
            layers.Interpolate(scale_factor=q, mode=upmode),
            layers.Block(c, b, deconv=True, use_res=use_res, batch_norm=use_bn),
            layers.Interpolate(scale_factor=p, mode=upmode),
            layers.Block(b, a, deconv=True, use_res=use_res, batch_norm=use_bn),
            nn.ConvTranspose2d(a, out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        ])

        self.decoder = nn.Sequential(*modules)

    def forward(self, zsample):

        return self.decoder(zsample)


class SimpleVAE(nn.Module):
    def __init__(self, imgSize, zsize, depth=0):
        super().__init__()

        self.enc = ImEncoder(in_size=(imgSize, imgSize), zsize=zsize, depth=depth)
        self.dec = ImDecoder(in_size=(imgSize, imgSize), zsize=zsize, depth=depth, out_channels=3)

    # Sampling function (using the reparameterisation trick)
    def sample(self, mu, log_sigma2):
        if self.training:
            return utils.sample(mu, log_sigma2)
        else:
            return mu

    def forward(self, x):
        mu, log_sigma2 = self.enc(x)
        z = self.sample(mu, log_sigma2)
        return self.dec(z), mu, log_sigma2
    
class PixelVAE(nn.Module):
    
    def __init__(self, imgSize, zsize, depth=0, outcn=-1):
        super().__init__()
        
        self.enc = ImEncoder(in_size=(imgSize, imgSize), zsize=zsize, depth=depth)
        
        if outcn > 0:
            self.dec = ImDecoder(in_size=(imgSize, imgSize), zsize=zsize, depth=depth, out_channels=outcn)
            self.pixcnn = LGated((3, imgSize, imgSize), outcn)
        else:
            self.dec = layers.Lambda(lambda x : x)
            self.pixcnn = CGated((3, imgSize, imgSize), (zsize,))
        
    # Sampling function (using the reparameterisation trick)
    def sample(self, mu, log_sigma2):
        if self.training:
            eps = torch.randn(mu.shape, device=mu.device)
            return mu + torch.exp(log_sigma2 / 2) * eps
        else:
            return mu
        
    def draw_sample(self, results):
        
        b, colors, c, h, w = results.size()
        
        sample = torch.randn(b, c, h, w, device=results.device)
        
        for i in range(h):
            for j in range(w):
                for channel in range(c):
                    probs = F.softmax(results[:, :, channel, i, j], dim=1)
                    pixel_sample = torch.multinomial(probs, 1).float() / 255.
                    sample[:, channel, i, j] = pixel_sample.squeeze()

        return sample
    
    def forward(self, x):
        mu, log_sigma2 = self.enc(x)
        z = self.sample(mu, log_sigma2)
        cond = self.dec(z)
        return self.pixcnn(x, cond), mu, log_sigma2
    
    @staticmethod
    def loss_ce(y_pred, y_true):
        """VAE Loss"""
        
        recon_x, mu, log_sigma2 = y_pred
        
        target = (y_true * 255).long()
        
        recon = F.cross_entropy(recon_x, target, reduction='none').view(y_true.shape[0], -1).sum(dim=1).mean()
        kld = utils.kld(mu, log_sigma2)
        
        loss = recon + kld

        return loss


# Normal VAE layers

class Encoder(nn.Module):
    """simple convolutional encoder"""

    def __init__(self, conv_ch, output_dim):
        super().__init__()
        self.dim = output_dim
        self.conv1 = nn.Conv2d(3, 3, 2, stride=1)
        self.conv2 = nn.Conv2d(3, conv_ch, 2, stride=2)
        self.conv3 = nn.Conv2d(conv_ch, conv_ch, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(conv_ch, conv_ch, 3, stride=1, padding=1)
        self.hidden_dim = 16 * 16 * conv_ch
        self.intermediate_dim = 128
        self.inter = nn.Linear(self.hidden_dim, self.intermediate_dim)
        self.mu = nn.Linear(self.intermediate_dim, output_dim)
        self.log_sigma2 = nn.Linear(self.intermediate_dim, output_dim)

    def forward(self, x):
        x = F.pad(x, (1, 0, 1, 0))
        x = self.conv1(x)
        x = F.relu(x)
        x = F.pad(x, (1, 0, 1, 0))
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = x.view(-1, self.hidden_dim)
        x = self.inter(x)
        x = F.relu(x)

        mu_ = self.mu(x)
        log_sigma2_ = self.log_sigma2(x)

        return mu_, log_sigma2_


class Decoder(nn.Module):
    """simple convolutional decoder"""

    def __init__(self, input_dim, conv_ch):
        super().__init__()
        self.hidden_dim = 16 * 16 * conv_ch
        self.intermediate_dim = 128
        self.conv_ch = conv_ch
        self.inter = nn.Linear(input_dim, self.intermediate_dim)
        self.hidden = nn.Linear(self.intermediate_dim, self.hidden_dim)
        self.deconv1 = nn.ConvTranspose2d(conv_ch, conv_ch, 3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(conv_ch, conv_ch, 3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(conv_ch, conv_ch, 2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(conv_ch, 3, 1, stride=1)

    def forward(self, x):
        x = self.inter(x)
        x = F.relu(x)
        x = self.hidden(x)
        x = F.relu(x)
        x = x.view(-1, self.conv_ch, 16, 16)
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = F.relu(x)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x


class ConvVAE(nn.Module):
    def __init__(self, conv_ch, embedding_dim):
        super().__init__()

        self.enc = Encoder(conv_ch, embedding_dim)
        self.dec = Decoder(embedding_dim, conv_ch)

    # Sampling function (using the reparameterisation trick)
    def sample(self, mu, log_sigma2):
        if self.training:
            return utils.sample(mu, log_sigma2)
        else:
            return mu

    def forward(self, x):
        mu, log_sigma2 = self.enc(x)
        z = self.sample(mu, log_sigma2)
        return self.dec(z), mu, log_sigma2

    
    @staticmethod
    def loss_bce(y_pred, y_true):
        """VAE Loss"""
        recon_x, mu, log_sigma2 = y_pred
        # E[log P(X|z)] - as images are binary it makes most sense to use binary cross entropy
        # we need to be a little careful - by default torch averages over every observation
        # (e.g. each  pixel in each image of each batch), whereas we want the average over entire
        # images instead
        recon = F.binary_cross_entropy(recon_x, y_true, reduction='sum') / y_true.shape[0]

        loss = recon + ConvVAE.kld(mu, log_sigma2)

        return loss

    @staticmethod
    def loss_mse(y_pred, y_true):
        """VAE Loss"""
        recon_x, mu, log_sigma2 = y_pred
        
        recon = F.mse_loss(recon_x, y_true, reduction='sum') / y_true.shape[0]
        
        loss = recon + ConvVAE.kld(mu, log_sigma2)

        return loss
    
    @staticmethod
    def kld(mu, log_sigma2):
        
        # D_KL(Q(z|X) || P(z|X)) - calculate in closed form
        kl = torch.mean(0.5 * torch.sum(torch.exp(log_sigma2) + mu ** 2 - 1. - log_sigma2, 1))
        return kl
    
    
# IntroVAE implementation
    
class IntroVAE(ConvVAE):
    def __init__(self, imgSize, zsize, depth=0, alpha=0.25, beta=0.5, gamma=1, margin=110, lr=1e-3, amsgrad=False):
        super().__init__(imgSize, zsize)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.optim_encoder = optim.Adam(self.enc.parameters(), lr=lr, amsgrad=amsgrad)
        self.optim_decoder = optim.Adam(self.dec.parameters(), lr=lr, amsgrad=amsgrad)

# Classes to be used with torchbearer
class encoder_step(tb.Callback):
    
    def __init__(self, pretrain=False):
        self.pretrain = pretrain

    def on_sample(self, state):
        model = state[tb.MODEL]
        x = state[tb.X]
        
        # 1. update encoder
        mu, log_sigma2 = model.enc(x)
        z = model.sample(mu, log_sigma2)
        xr = model.dec(z)
        
        zp = torch.randn_like(z)
        xp = model.dec(zp)

        loss_ae = F.mse_loss(xr, x, reduction='sum') / x.shape[0]
        reg_ae = utils.kld(mu, log_sigma2)
        
        # Needed to optimize parameters
        if (self.pretrain and state[tb.EPOCH] >= 2):
            utils.debug(reg_ae)

        mur_ng, log_sigma2r_ng = model.enc(xr.detach())
        regr_ng = utils.kld(mur_ng, log_sigma2r_ng)
        # max(0, margin - l)
        regr_ng = torch.clamp(model.margin - regr_ng, min=0)
        
        mupp_ng, log_sigma2pp_ng = model.enc(xp.detach())
        regpp_ng = utils.kld(mupp_ng, log_sigma2pp_ng)
        # max(0, margin - l)
        regpp_ng = torch.clamp(model.margin - regpp_ng, min=0)


        encoder_adv = regr_ng + regpp_ng
        encoder_loss = model.gamma * reg_ae + model.alpha * encoder_adv + model.beta * loss_ae
        
        state[LOSS_E] = encoder_loss
        
        # Adam Updates
        model.optim_encoder.zero_grad()
        encoder_loss.backward()
        model.optim_encoder.step()

# Classes to be used with torchbearer
class decoder_step(tb.Callback):
    
    def on_sample(self, state):
        model = state[tb.MODEL]
        x = state[tb.X]
        
        # 2. update decoder
        mu, log_sigma2 = model.enc(x)
        z = model.sample(mu, log_sigma2)
        xr = model.dec(z)
        
        zp = torch.randn_like(z)
        xp = model.dec(zp)
        
        loss_ae = F.mse_loss(xr, x, reduction='sum') / x.shape[0]
        
        mur, log_sigma2r = model.enc(xr)
        regr = utils.kld(mur, log_sigma2r)
        
        mupp, log_sigma2pp = model.enc(xp)
        regpp = utils.kld(mupp, log_sigma2pp)

        # by Eq.12, the 1st term of loss
        decoder_adv = regr + regpp
        decoder_loss = model.alpha * decoder_adv + model.beta * loss_ae
        
        state[LOSS_D] = decoder_loss
        
        # Adam Updates
        model.optim_decoder.zero_grad()
        decoder_loss.backward()
        model.optim_decoder.step()
        
# Classes to be used with torchbearer
class training_step(tb.Callback):
    
    def on_sample(self, state):
        model = state[tb.MODEL]
        x = state[tb.X]
        
        mu, log_sigma2 = model.enc(x)
        z = model.sample(mu, log_sigma2)
        xr = model.dec(z)
        
        zp = torch.randn_like(z)
        xp = model.dec(zp)
        
        loss_ae = F.mse_loss(xr, x, reduction='sum') / x.shape[0]
        reg_ae = utils.kld(mu, log_sigma2)
        
        # Needed to optimize parameters
        if state[tb.EPOCH] >= 2:
            utils.debug(reg_ae)
        
        mur, log_sigma2r = model.enc(xr.detach())
        # max(0, margin - l)
        regr_ng = torch.clamp(model.margin - utils.kld(mur, log_sigma2r), min=0)
        
        mupp, log_sigma2pp = model.enc(xp.detach())
        # max(0, margin - l)
        regpp_ng = torch.clamp(model.margin - utils.kld(mur, log_sigma2r), min=0)
        
        # Encoder step
        encoder_adv = model.gamma * reg_ae + model.alpha * (regr_ng + regpp_ng)
        encoder_loss = encoder_adv + model.beta * loss_ae
        state[LOSS_E] = encoder_loss
        
        # Adam Updates
        model.optim_encoder.zero_grad()
        encoder_loss.backward(retain_graph=True)
        model.optim_encoder.step()
        
        mur, log_sigma2r = model.enc(xr)
        
        mupp, log_sigma2pp = model.enc(xp)
        
        # Decoder step
        decoder_adv = model.alpha * (utils.kld(mur, log_sigma2r) + utils.kld(mupp, log_sigma2r))
        decoder_loss = model.alpha * decoder_adv + model.beta * loss_ae
        state[LOSS_D] = decoder_loss
        
        # Adam Updates
        model.optim_decoder.zero_grad()
        decoder_loss.backward(retain_graph=False)
        model.optim_decoder.step()
        
        del z, reg_ae, loss_ae, encoder_loss, decoder_loss, xr, xp, mur, log_sigma2r, mupp, log_sigma2pp, mu, log_sigma2
        torch.cuda.empty_cache()
        
# Classes to be used with torchbearer
class forward_step(tb.Callback):
    
    def on_sample(self, state):
        with torch.no_grad():
            state[tb.Y_PRED] = state[tb.MODEL](state[tb.X])
        
@tb.metrics.running_mean
@tb.metrics.mean
class encoder_loss(tb.metrics.Metric):
    def __init__(self):
        super().__init__('encoder_loss')

    def process(self, state):
        return state[LOSS_E]

@tb.metrics.running_mean
@tb.metrics.mean
class decoder_loss(tb.metrics.Metric):
    def __init__(self):
        super().__init__('decoder_loss')

    def process(self, state):
        return state[LOSS_D]
    
@tb.metrics.running_mean
@tb.metrics.mean
class recons_loss(tb.metrics.Metric):
    def __init__(self):
        super().__init__('recons_loss')

    def process(self, state):
        with torch.no_grad():
            y_true = state[tb.Y_TRUE]
            recon_x, mu, log_sigma2 = state[tb.Y_PRED]
            recon = F.mse_loss(recon_x, y_true, reduction='sum') / y_true.shape[0]
            return recon
        
@tb.metrics.running_mean
@tb.metrics.mean
class sampled_recons_loss(tb.metrics.Metric):
    def __init__(self):
        super().__init__('recons_loss')

    def process(self, state):
        with torch.no_grad():
            y_true = state[tb.Y_TRUE]
            recon_x, mu, log_sigma2 = state[tb.Y_PRED]
            recon_x = state[tb.MODEL].draw_sample(recon_x)
            recon = F.mse_loss(recon_x, y_true, reduction='sum') / y_true.shape[0]
            return recon