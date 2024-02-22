# VAE models
# Credits to: Jonathon Hare

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import utils

from tensorboard_logging import ReconstructionsLogger, TensorBoardModelLogger, LatentSpaceReconLogger, RandomReconLogger
from torchbearer import Trial
from torchbearer.callbacks.tensor_board import TensorBoard
from utils import AEDatasetWrapper


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
    def loss_ce(y_pred, y_true):
        """VAE Loss"""
        recon_x, mu, log_sigma2 = y_pred
        # E[log P(X|z)] - as images are binary it makes most sense to use binary cross entropy
        # we need to be a little careful - by default torch averages over every observation
        # (e.g. each  pixel in each image of each batch), whereas we want the average over entire
        # images instead
        recon = F.binary_cross_entropy(recon_x, y_true, reduction='sum') / y_true.shape[0]

        loss = recon + kld(mu, log_sigma2)

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
    
class IntroVAE(ConvVAE):
    def __init__(self, conv_ch, embedding_dim, alpha=0.25, beta=0.5, gamma=1, margin=110, lr=1e-3):
        super().__init__(conv_ch, embedding_dim)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.optim_encoder = optim.Adam(self.enc.parameters(), lr=lr)
        self.optim_decoder = optim.Adam(self.dec.parameters(), lr=lr)
    
    def train_iter(self, x):
        # 1. update encoder
        mu, log_sigma2 = self.enc(x)
        z = self.sample(mu, log_sigma2)
        xr = self.dec(z)
        
        zp = torch.randn_like(z)
        xp = self.dec(zp)

        loss_ae = F.mse_loss(xr, x, reduction='sum').sqrt()
        reg_ae = ConvVAE.kld(mu, log_sigma2)

        mur_ng, log_sigma2r_ng = self.enc(xr.detach())
        zr_ng = self.sample(mur_ng, log_sigma2r_ng)
        regr_ng = ConvVAE.kld(mur_ng, log_sigma2r_ng)
        # max(0, margin - l)
        regr_ng = torch.clamp(self.margin - regr_ng, min=0)
        
        mupp_ng, log_sigma2pp_ng = self.enc(xp.detach())
        zpp_ng = self.sample(mupp_ng, log_sigma2pp_ng)
        regpp_ng = ConvVAE.kld(mupp_ng, log_sigma2pp_ng)
        # max(0, margin - l)
        regpp_ng = torch.clamp(self.margin - regpp_ng, min=0)


        encoder_adv = regr_ng + regpp_ng
        encoder_loss = self.gamma * reg_ae + self.alpha * encoder_adv + self.beta * loss_ae
        
        # Adam Updates
        self.optim_encoder.zero_grad()
        encoder_loss.backward()
        self.optim_encoder.step()


        # 2. update decoder
        mu, log_sigma2 = self.enc(x)
        z = self.sample(mu, log_sigma2)
        xr = self.dec(z)
        
        zp = torch.randn_like(z)
        xp = self.dec(zp)
        
        loss_ae = F.mse_loss(xr, x, reduction='sum').sqrt()
        
        mur, log_sigma2r = self.enc(xr)
        zr = self.sample(mur, log_sigma2r)
        regr = ConvVAE.kld(mur, log_sigma2r)
        
        mupp, log_sigma2pp = self.enc(xp)
        zpp  = self.sample(mupp, log_sigma2pp)
        regpp = ConvVAE.kld(mupp, log_sigma2pp)

        # by Eq.12, the 1st term of loss
        decoder_adv = regr + regpp
        decoder_loss = self.alpha * decoder_adv + self.beta * loss_ae
        
        # Adam Updates
        self.optim_decoder.zero_grad()
        decoder_loss.backward()
        self.optim_decoder.step()