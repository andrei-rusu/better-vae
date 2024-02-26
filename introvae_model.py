import  torch
from    torch import nn, optim
from    torch.nn import functional as F
import  math
from    layers import IntroReshape, Flatten, ResBlk
import utils


# IntroVAE implementation based on dragen1846

class Encoder(nn.Module):

    def __init__(self, imgsz, ch, z_dim):
        """
        :param imgsz:
        :param ch: base channels
        """
        super(Encoder, self).__init__()

        x = torch.randn(2, 3, imgsz, imgsz)
        print('Encoder:', list(x.shape), end='=>')

        layers = [
            nn.Conv2d(3, ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=None, padding=0),
        ]
        # just for print
        out = nn.Sequential(*layers)(x)
        print(list(out.shape), end='=>')

        # [b, ch_cur, imgsz, imgsz] => [b, ch_next, mapsz, mapsz]
        mapsz = imgsz // 2
        ch_cur = ch
        ch_next = ch_cur * 2

        while mapsz > 4: # util [b, ch_, 4, 4]
            # add resblk
            layers.extend([
                ResBlk([1, 3, 3], [ch_cur, ch_next, ch_next, ch_next]),
                nn.AvgPool2d(kernel_size=2, stride=None)
            ])
            mapsz = mapsz // 2
            ch_cur = ch_next
            ch_next = ch_next * 2 if ch_next < 512 else 512 # set max ch=512

            # for print
            out = nn.Sequential(*layers)(x)
            print(list(out.shape), end='=>')

        layers.extend([
            ResBlk([3, 3], [ch_cur, ch_next, ch_next]),
            nn.AvgPool2d(kernel_size=2, stride=None),
            ResBlk([3, 3], [ch_next, ch_next, ch_next]),
            nn.AvgPool2d(kernel_size=2, stride=None),
            Flatten()
        ])

        # to init linear layer
        out = nn.Sequential(*layers)(x)
        layers.extend([
            nn.Linear(out.size(1), 2*z_dim)
        ])
        
        self.net = nn.Sequential(*layers)

        # for printing
        mu, log_sigma2 = self.net(x).chunk(2, dim=1)
        print(list(mu.shape), ' x 2')


    def forward(self, x):
        """
        :param x:
        :return:
        """
        return self.net(x).chunk(2, dim=1)


class Decoder(nn.Module):


    def __init__(self, imgsz, z_dim):
        """
        :param imgsz:
        :param z_dim:
        """
        super(Decoder, self).__init__()

        mapsz = 4
        ch_next = z_dim
        print('Decoder:', [z_dim], '=>', [2, ch_next, mapsz, mapsz], end='=>')

        # z: [b, z_dim] => [b, z_dim, 4, 4]
        layers = [
            # z_dim => z_dim * 4 * 4 => [z_dim, 4, 4] => [z_dim, 4, 4]
            nn.Linear(z_dim, z_dim * mapsz * mapsz),
            nn.BatchNorm1d(z_dim * mapsz * mapsz),
            nn.ReLU(inplace=True),
            IntroReshape(z_dim, mapsz, mapsz),
            ResBlk([3, 3], [z_dim, z_dim, z_dim])
        ]


        # scale imgsz up while keeping channel untouched
        # [b, z_dim, 4, 4] => [b, z_dim, 8, 8] => [b, z_dim, 16, 16]
        for i in range(2):
            layers.extend([
                nn.Upsample(scale_factor=2),
                ResBlk([3, 3], [ch_next, ch_next, ch_next])
            ])
            mapsz = mapsz * 2

            # for print
            tmp = torch.randn(2, z_dim)
            net = nn.Sequential(*layers)
            out = net(tmp)
            print(list(out.shape), end='=>')
            del net

        # scale imgsz up and scale imgc down
        # [b, z_dim, 16, 16] => [z_dim//2, 32, 32] => [z_dim//4, 64, 64] => [z_dim//8, 128, 128]
        # => [z_dim//16, 256, 256] => [z_dim//32, 512, 512]
        while mapsz < imgsz//2:
            ch_cur = ch_next
            ch_next = ch_next // 2 if ch_next >=32 else ch_next # set mininum ch=16
            layers.extend([
                # [2, 32, 32, 32] => [2, 32, 64, 64]
                nn.Upsample(scale_factor=2),
                # => [2, 16, 64, 64]
                ResBlk([1, 3, 3], [ch_cur, ch_next, ch_next, ch_next])
            ])
            mapsz = mapsz * 2

            # for print
            tmp = torch.randn(2, z_dim)
            net = nn.Sequential(*layers)
            out = net(tmp)
            print(list(out.shape), end='=>')
            del net


        # [b, ch_next, 512, 512] => [b, 3, 1024, 1024]
        layers.extend([
            nn.Upsample(scale_factor=2),
            ResBlk([3, 3], [ch_next, ch_next, ch_next]),
            nn.Conv2d(ch_next, 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        ])

        self.net = nn.Sequential(*layers)

        # for print
        tmp = torch.randn(2, z_dim)
        out = self.net(tmp)
        print(list(out.shape))

    def forward(self, x):
        """
        :param x: [b, z_dim]
        :return:
        """
        # print('before forward:', x.shape)
        x =  self.net(x)
        # print('after forward:', x.shape)
        return x


class IntroVAE(nn.Module):


    def __init__(self, imgSize, zsize, depth, alpha=0.25, beta=0.5, gamma=1, margin=110, lr=1e-3):
        """
        :param imgsz:
        :param z_dim: h_dim is the output dim of encoder, and we use z_net net to convert it from
        h_dim to 2*z_dim and then splitting.
        """
        super().__init__()

        imgsz = imgSize
        z_dim = zsize

        # set first conv channel as 16
        self.enc = Encoder(imgsz, 16, z_dim)

        # create decoder by z_dim
        self.dec = Decoder(imgsz, z_dim)


        self.alpha = alpha # for adversarial loss
        self.beta = beta # for reconstruction loss
        self.gamma = gamma # for variational loss
        self.margin = margin # margin in eq. 11

        self.optim_encoder = optim.Adam(self.enc.parameters(), lr=lr, betas=(0.9, 0.999))
        self.optim_decoder = optim.Adam(self.dec.parameters(), lr=lr, betas=(0.9, 0.999))
        
        # Print layers
        x = torch.randn(2, 3, imgsz, imgsz)
        mu_, log_sigma2_ = self.enc(x)
        z = utils.sample(mu_, log_sigma2_)
        out = self.dec(z)
        
        # print
        print('IntroVAE x:', list(x.shape), 'mu_:', list(mu_.shape), 'z:', list(z.shape), 'out:', list(out.shape))
        
    # Sampling function (using the reparameterisation trick)
    def sample(self, mu, log_sigma2):
        if self.training:
            eps = torch.randn(mu.shape, device=mu.device)
            return mu + torch.exp(log_sigma2 / 2) * eps
        else:
            return mu


    def set_alph_beta_gamma(self, alpha, beta, gamma):
        """
        this func is for pre-training, to set alpha=0 to transfer to vilina vae.
        :param alpha: for adversarial loss
        :param beta: for reconstruction loss
        :param gamma: for variational loss
        :return:
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        """
        The notation used here all come from Algorithm 1, page 6 of official paper.
        can refer to Figure7 in page 15 as well.
        :param x: [b, 3, 1024, 1024]
        :return:
        """

        mu, log_sigma2 = self.enc(x)
        z = self.sample(mu, log_sigma2)
        return self.dec(z), mu, log_sigma2
