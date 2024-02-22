import torchbearer
import torchvision.utils as utils
from torchbearer.callbacks.tensor_board import AbstractTensorBoard, TensorBoard
import torchbearer.callbacks as callbacks
import torch
from textwrap import indent
import os
import copy

class TensorBoardExtension(TensorBoard):
    
    def direct_call(self, model, trainloader):
        if self.writer == None:
            self.log_dir = os.path.join(self.log_dir, model.__class__.__name__ + '_' + self.comment)
            self.writer = self.get_writer(visdom=self.visdom, visdom_params=self.visdom_params)
            
        inputs, _ = next(iter(trainloader))
            
        dummy = torch.rand(inputs.size(), requires_grad=False)
        model_copy = copy.deepcopy(model).to('cpu')
        self.writer.add_graph(model_copy, (dummy,))

class ReconstructionLossLogger(AbstractTensorBoard):
    
    def direct_call(self, epoch, metrics):
        if self.writer == None:
            self.log_dir = os.path.join(self.log_dir, model.__class__.__name__ + '_' + self.comment)
            self.writer = self.get_writer(visdom=self.visdom, visdom_params=self.visdom_params)
        for metric in metrics:
            self.writer.add_scalar('epoch/' + metric, metrics[metric], epoch)
        
class TensorBoardModelLogger(AbstractTensorBoard):
    """
    Add some info about the model to the log
    """
    @callbacks.once
    def on_step_validation(self, state):
        super(TensorBoardModelLogger, self).on_start(state)
        self.writer.add_text("Model", indent(repr(state[torchbearer.MODEL]), '    '))
        
    def direct_call(self, model):
        if self.writer == None:
            self.log_dir = os.path.join(self.log_dir, model.__class__.__name__ + '_' + self.comment)
            self.writer = self.get_writer(visdom=self.visdom, visdom_params=self.visdom_params)
            
        self.writer.add_text("Model", indent(repr(model), '    '))

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
        recon = state[torchbearer.Y_PRED][0].view(-1, *self.output_shape)[:self.nrecon, :, :, :]

        image = utils.make_grid(
            recon,
            nrow=self.nrow,
            normalize=self.normalize
        )
        self.writer.add_image(self.title, image, state[torchbearer.EPOCH])
        
    def direct_call(self, model, epoch, y_pred):
        if self.writer == None:
            self.log_dir = os.path.join(self.log_dir, model.__class__.__name__ + '_' + self.comment)
            self.writer = self.get_writer(visdom=self.visdom, visdom_params=self.visdom_params)
        
        recon = y_pred[0].view(-1, *self.output_shape)[:self.nrecon, :, :, :]

        image = utils.make_grid(
            recon,
            nrow=self.nrow,
            normalize=self.normalize
        )
        self.writer.add_image(self.title, image, epoch)

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

        samples = model.dec(z).data

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
        
    def direct_call(self, model, epoch):
        if self.writer == None:
            self.log_dir = os.path.join(self.log_dir, model.__class__.__name__ + '_' + self.comment)
            self.writer = self.get_writer(visdom=self.visdom, visdom_params=self.visdom_params)
        
        size = self.size

        device = "cuda:0" if next(model.parameters()).is_cuda else "cpu"

        z = torch.zeros(size * size, *self.latent_dim, device=device)
        for j, y in enumerate(range(int(-size / 2), int(size / 2 + 1))):
            for i, x in enumerate(range(int(-size / 2), int(size / 2 + 1))):
                z[j * size + i, self.dims[0]] = x / ((size - 1) / 2) * self.sigmas
                z[j * size + i, self.dims[1]] = y / ((size - 1) / 2) * self.sigmas

        samples = model.dec(z).data

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
        self.writer.add_image(self.title, image, epoch)

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
        samples = model.dec(z).data

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
        
    def direct_call(self, model, epoch):
        if self.writer == None:
            self.log_dir = os.path.join(self.log_dir, model.__class__.__name__ + '_' + self.comment)
            self.writer = self.get_writer(visdom=self.visdom, visdom_params=self.visdom_params)
        
        size = self.nrow * self.ncol

        device = "cuda:0" if next(model.parameters()).is_cuda else "cpu"

        z = torch.randn(size, *self.latent_dim, device=device)
        samples = model.dec(z).data

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
        self.writer.add_image(self.title, image, epoch)