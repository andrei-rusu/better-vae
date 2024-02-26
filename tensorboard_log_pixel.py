import torchbearer
import torchvision.utils as utils
from torchbearer.callbacks.tensor_board import AbstractTensorBoard, TensorBoard
import torchbearer.callbacks as callbacks
import torch
from textwrap import indent
import os
import copy
import random

import tqdm

import utils as ut

class TensorBoardModelLogger(AbstractTensorBoard):
    """
    Add some info about the model to the log
    """
    @callbacks.once
    def on_step_validation(self, state):
        super(TensorBoardModelLogger, self).on_start(state)
        self.writer.add_text("Model", indent(repr(state[torchbearer.MODEL]), '    '))


class ReconstructionsLogger(AbstractTensorBoard):
    """
    Visualise reconstructions of the validation set
    """
    def __init__(self, log_dir='./logs', comment='torchbearer', nrow=8, nrecon=64, normalize=True,
                 output_shape=(1, 28, 28), title='Validation Reconstructions'):
        super(ReconstructionsLogger, self).__init__(log_dir, comment)
        self.nrow = nrow
        self.nrecon = nrecon
        self.output_shape = output_shape
        self.normalize = normalize
        self.title = title

    @callbacks.once_per_epoch
    def on_step_validation(self, state):
        super(ReconstructionsLogger, self).on_step_validation(state)
        
        sample = state[torchbearer.MODEL].draw_sample(state[torchbearer.Y_PRED][0])
        
        recon = sample.view(-1, *self.output_shape)[:self.nrecon, :, :, :]

        image = utils.make_grid(
            recon,
            nrow=self.nrow,
            normalize=self.normalize
        )
        self.writer.add_image(self.title, image, state[torchbearer.EPOCH])


class LatentSpaceReconLogger(AbstractTensorBoard):
    """
    Visualise 2 dimensions of the latent space by sampling z's on a grid
    and creating reconstructions from these.
    """
    def __init__(self, log_dir='./logs', comment='torchbearer', size=21, sigmas=4, dims=(0, 1),
                 latent_dim=2, output_shape=(1, 28, 28), normalize=True, title='Latent Space'):
        super(LatentSpaceReconLogger, self).__init__(log_dir, comment)
        self.size = size
        self.sigmas = sigmas
        assert len(dims) is 2, "can only draw in two dimensions"
        self.dims = dims
        self.latent_dim = latent_dim if isinstance(latent_dim, tuple) else (latent_dim,)
        self.output_shape = output_shape
        self.normalize = normalize
        self.title = title

    def on_end_epoch(self, state):
        super(LatentSpaceReconLogger, self).on_end_epoch(state)
        model = state[torchbearer.MODEL]
        size = self.size

        device = "cuda:0" if next(model.parameters()).is_cuda else "cpu"

        z = torch.zeros(size * size, *self.latent_dim, device=device)
        for j, y in enumerate(range(int(-size / 2), int(size / 2 + 1))):
            for i, x in enumerate(range(int(-size / 2), int(size / 2 + 1))):
                z[j * size + i, self.dims[0]] = x / ((size - 1) / 2) * self.sigmas
                z[j * size + i, self.dims[1]] = y / ((size - 1) / 2) * self.sigmas

        X = next(iter(state[torchbearer.VALIDATION_GENERATOR]))[0]
        b, c, h, w = X.size()
        seed = torch.zeros(size * size, c, h, w, device=device)
        
        dim = random.randint(0, b-size-1)
        parts = X[dim:dim+size, :, :h//2, :]
        
        for i in range(1, size+1):
            seed[(i-1)*size:i*size, :, :h//2, :] = parts 
        
        samples = model.draw_sample(model.pixcnn(seed, z).data)

        recon = torch.zeros(size*size, *self.output_shape, device=device)
        for j in range(0, size):
            for i in range(0, size):
                idx = i + j * size
                recon[idx] = samples[idx].reshape(*self.output_shape)

        image = utils.make_grid(
            recon,
            nrow=size,
            normalize=self.normalize
        )
        self.writer.add_image(self.title, image, state[torchbearer.EPOCH])


class RandomReconLogger(AbstractTensorBoard):
    """
    Visualise a set of reconstructions from randomly sampled z's.
    """
    def __init__(self, log_dir='./logs', comment='torchbearer', nrow=8, ncol=8,
                 latent_dim=2, output_shape=(1, 28, 28), normalize=True, title='Random z Reconstructions'):
        super(RandomReconLogger, self).__init__(log_dir, comment)
        self.nrow = nrow
        self.ncol = ncol
        self.latent_dim = latent_dim if isinstance(latent_dim, tuple) else (latent_dim,)
        self.output_shape = output_shape
        self.normalize = normalize
        self.title = title

    def on_end_epoch(self, state):
        super(RandomReconLogger, self).on_end_epoch(state)
        model = state[torchbearer.MODEL]
        size = self.nrow * self.ncol

        device = "cuda:0" if next(state[torchbearer.MODEL].parameters()).is_cuda else "cpu"
        
        z = torch.randn(size, *self.latent_dim, device=device)

        X = next(iter(state[torchbearer.VALIDATION_GENERATOR]))[0]
        b, c, h, w = X.size()
        seed = torch.zeros(size, c, h, w, device=device)
        
        dim = random.randint(0, b-size-1)
        
        seed[:, :, :h//2, :] = X[dim:dim+size, :, :h//2, :]
        
        samples = model.draw_sample(model.pixcnn(seed, z).data)

        recon = torch.zeros(size, *self.output_shape, device=device)
        for j in range(0, self.nrow):
            for i in range(0, self.ncol):
                idx = i + j * self.ncol
                recon[idx] = samples[idx].reshape(*self.output_shape)

        image = utils.make_grid(
            recon,
            nrow=self.nrow,
            normalize=self.normalize
        )
        self.writer.add_image(self.title, image, state[torchbearer.EPOCH])
        
class RandomPixReconLogger(AbstractTensorBoard):
    """
    Visualise a set of reconstructions from randomly sampled z's.
    """
    def __init__(self, log_dir='./logs', comment='torchbearer', nrow=8, ncol=8,
                 latent_dim=2, output_shape=(1, 28, 28), normalize=True, title='Random z Pix Reconstructions'):
        super().__init__(log_dir, comment)
        self.nrow = nrow
        self.ncol = ncol
        self.latent_dim = latent_dim if isinstance(latent_dim, tuple) else (latent_dim,)
        self.output_shape = output_shape
        self.normalize = normalize
        self.title = title

    def on_end_epoch(self, state):
        super().on_end_epoch(state)
        epoch = state[torchbearer.EPOCH]
        if ((epoch + 1) % 5 == 0 and epoch != 0):
            model = state[torchbearer.MODEL]
            size = self.nrow * self.ncol

            b, C, H, W = state[torchbearer.X].size()

            device = "cuda:0" if next(state[torchbearer.MODEL].parameters()).is_cuda else "cpu"

            ###
            sample_zs = torch.randn(12, *self.latent_dim, device=device)
            sample_zs = sample_zs.unsqueeze(1).expand(12, 6, -1).contiguous().view(72, 1, -1).squeeze(1)

            # A sample of 144 square images with 3 channels, of the chosen resolution
            # (144 so we can arrange them in a 12 by 12 grid)
            sample_init_zeros = torch.zeros(72, C, H, W)
            
            # Vary first pixel in the random zeros images
            for i in range(1, 18):
                sample_init_zeros[(i*4):((i+1)*4), :, 0, 0] = torch.tensor([i*8 + 13, i*11 + 3, i * 14 + 7])
            
            sample_init_seeds = torch.zeros(72, C, H, W)

            sh, sw = H//2, W//2

            # Init second half of sample with patches from test set, to seed the sampling
            testbatch = ut.readn(state[torchbearer.VALIDATION_GENERATOR], 12)
            testbatch = testbatch.unsqueeze(1).expand(12, 6, C, H, W).contiguous().view(72, 1, C, H, W).squeeze(1)
            sample_init_seeds[:, :, :sh, :] = testbatch[:, :, :sh, :]

            sample_zeros = draw_pix_sample(sample_init_zeros, model.dec, model.pixcnn, sample_zs)
            sample_seeds = draw_pix_sample(sample_init_seeds, model.dec, model.pixcnn, sample_zs, seedsize=(sh, W))
            sample = torch.cat([sample_zeros, sample_seeds], dim=0)

            image = utils.make_grid(
                sample,
                nrow=8,
                normalize=self.normalize
            )

            self.writer.add_image(self.title, image, state[torchbearer.EPOCH])
        
def draw_pix_sample(seeds, decoder, pixcnn, zs, seedsize=(0,0)):

    b, c, h, w = seeds.size()

    sample = seeds.clone()
    if torch.cuda.is_available():
        sample, zs = sample.cuda(), zs.cuda()

    cond = decoder(zs)

    for i in tqdm.trange(h):
        for j in range(w):

            if i < seedsize[0] and j < seedsize[1]:
                continue

            for channel in range(c):

                result = pixcnn(sample, cond)
                probs = torch.nn.functional.softmax(result[:, :, channel, i, j]).data

                pixel_sample = torch.multinomial(probs, 1).float() / 255.
                sample[:, channel, i, j] = pixel_sample.squeeze().cuda()

    return sample