import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class Traj_MLP_CVAE(nn.Module):  # T*4 => T*8 => ResBlock -> z => ResBlock 
    def __init__(self, nz, feature_dim, T, residual=False, load_path=None):
        super(Traj_MLP_CVAE, self).__init__()
        self.T = T
        self.feature_dim = feature_dim
        self.nz = nz
        self.residual = residual
        self.load_path = load_path

        """MLP"""
        self.enc1 = ResBlock(Fin=2*feature_dim*T, Fout=2*feature_dim*T, n_neurons=2*feature_dim*T)
        self.enc2 = ResBlock(Fin=2*feature_dim*T, Fout=2*feature_dim*T, n_neurons=2*feature_dim*T)
        self.enc_mu = nn.Linear(2*feature_dim*T, nz)
        self.enc_var = nn.Linear(2*feature_dim*T, nz)

        self.dec1 = ResBlock(Fin=nz+feature_dim*T, Fout=2*feature_dim*T, n_neurons=2*feature_dim*T)
        self.dec2 = ResBlock(Fin=2*feature_dim*T + feature_dim*T, Fout=feature_dim*T, n_neurons=feature_dim*T)

        if self.load_path is not None:
            self._load_model()

    def encode(self, x, y):
        """ x: [bs, T*feature_dim] """
        bs = x.shape[0]
        x = torch.cat([x, y], dim=-1)
        h = self.enc1(x)
        h = self.enc2(h)
        z_mu = self.enc_mu(h)
        z_logvar = self.enc_var(h)

        return z_mu, z_logvar

    def decode(self, z, y):
        """z: [bs, nz]; y: [bs, 2*feature_dim]"""
        bs = y.shape[0]
        x = torch.cat([z, y], dim=-1)
        x = self.dec1(x)
        x = torch.cat([x, y], dim=-1)
        x = self.dec2(x)

        if self.residual:
            x = x + y

        return x.reshape(bs, self.feature_dim, -1)

    def reparameterize(self, mu, logvar, eps):
        std = torch.exp(0.5*logvar)
        return mu + eps * std

    def forward(self, x, y):
        bs = x.shape[0]
        # print(x.shape, y.shape)
        mu, logvar = self.encode(x, y)
        eps = torch.randn_like(logvar)
        z = self.reparameterize(mu, logvar, eps) #if self.training else mu
        pred = self.decode(z, y)

        return pred, mu, logvar

    def sample(self, y, z=None):
        if z is None:
            z = torch.randn((y.shape[0], self.nz), device=y.device)
        return self.decode(z, y)

    def _load_model(self):
        print('Loading Traj_CVAE from {} ...'.format(self.load_path))
        assert self.load_path is not None
        model_cp = torch.load(self.load_path)
        self.load_state_dict(model_cp['model_dict'])
