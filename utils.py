import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.utils.data.dataset import Dataset

def imshow(inp, normalized=False, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    if normalized:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    

# cuda helper
def cuda(obj):
    if torch.cuda.is_available():
        return obj.cuda()
    else:
        return obj
    
# Sampling function (using the reparameterisation trick)
def sample(mu, log_sigma2):
    eps = cuda(torch.randn(mu.shape[0], mu.shape[1]))
    return mu + torch.exp(log_sigma2 / 2) * eps

# function to plot 16 reconstructions 
def draw_samples(outputs, epoch, image_size, name):
    samples = outputs.data.cpu().numpy()[:16]

    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, smpl in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(smpl.reshape(3, image_size, image_size).transpose(1,2,0))
        
    folder = 'out-' + name + '/'

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(folder + '{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)
    
def draw_random_samples(dec, epoch, image_size, name):
    size = 21
    z = torch.randn(size*size, embedding_dim, 4, 4)

    samples = dec(cuda(z)).data.cpu().numpy()

    canvas = np.empty((image_size*size, image_size*size, 3))
    for j in range(0, size):
        for i in range(0, size):
            canvas[(size-1-j)*image_size:(size-j)*image_size,i*image_size:i*image_size+image_size,:] = samples[i + j*size].transpose(1, 2, 0)

    fig = plt.figure(figsize=(10, 10))        
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    
    folder = 'out-' + name + '/'

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(folder + 'random_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)

# function to plot by sampling a grid from -4.sigma ... +4.sigma
def draw_space(dec, image_size, name):
    size = 21
    z = torch.zeros(size*size, 2)
    for j, y in enumerate(range(int(-size/2), int(size/2 + 1))):
        for i, x in enumerate(range(int(-size/2), int(size/2 + 1))):
            z[j*size + i, 0] = x / ((size-1)/2) * 4
            z[j*size + i, 1] = y / ((size-1)/2) * 4

    samples = dec(cuda(z)).data.cpu().numpy()

    canvas = np.empty((image_size*size, image_size*size, 3))
    for j in range(0, size):
        for i in range(0, size):
            canvas[(size-1-j)*image_size:(size-j)*image_size,i*image_size:i*image_size+image_size,:] = samples[i + j*size].reshape(3, image_size, image_size).transpose(1, 2, 0)

    fig = plt.figure(figsize=(10, 10))        
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    
    folder = 'out-' + name + '/'

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(folder + 'space_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)

# function to plot a labelled scatter of the space of test images
def draw_scatter(enc, epoch, name):
    all_labels = []
    all_coords = np.empty((0,2))

    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = cuda(inputs)

        all_labels = np.append(all_labels, labels, axis=0)

        mu, log_sigma2 = enc(inputs)
        z = sample(mu, log_sigma2)

        all_coords = np.append(all_coords, z.data.cpu().numpy(), axis=0)

    fig = plt.figure()
    all_labels = all_labels
    plt.scatter(all_coords[:,0], all_coords[:,1], c=all_labels, cmap=plt.get_cmap("tab10"))
    plt.colorbar()
    
    folder = 'out-' + name + '/'

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(folder + 'scatter_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)
    
def splitset(dataset):
    master_len = len(dataset.imgs)
    train_len = int(0.8*master_len)
    valid_len = master_len - train_len
    train_set_holder, test_set_holder = data.random_split(dataset, (train_len, valid_len))
    train_set = train_set_holder.dataset
    test_set = test_set_holder.dataset
    return train_set, test_set

def save(model, filename):
    torch.save(model.state_dict(), "./weights/" + filename)
    
def load(model, filename):
    model.load_state_dict(torch.load("./weights/" + filename))
    
class AEDatasetWrapper(Dataset):
    """Wrapper for datasets used with auto encoders that return the data as the target"""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset.__getitem__(index)
        return data, data

    def __len__(self):
        return len(self.dataset)
