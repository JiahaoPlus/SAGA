import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncBlock(nn.Module):
    def __init__(self, nin, nout, downsample=True, kernel=3):
        super(EncBlock, self).__init__()
        self.downsample = downsample
        padding = kernel // 2

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=1, padding=padding, padding_mode='replicate'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=nout, out_channels=nout, kernel_size=kernel, stride=1, padding=padding, padding_mode='replicate'),
            nn.LeakyReLU(0.2),
        )

        if self.downsample:
            self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pooling = nn.MaxPool2d(kernel_size=(3,3), stride=(2, 1), padding=1)

    def forward(self, input):
        output = self.main(input)
        output = self.pooling(output)
        return output

class DecBlock(nn.Module):
    def __init__(self, nin, nout, upsample=True, kernel=3):
        super(DecBlock, self).__init__()
        self.upsample = upsample

        padding = kernel // 2
        if upsample:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=2, padding=padding)
        else:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=(2, 1), padding=padding)
        self.deconv2 = nn.ConvTranspose2d(in_channels=nout, out_channels=nout, kernel_size=kernel, stride=1, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, input, out_size):
        output = self.deconv1(input, output_size=out_size)
        output = self.leaky_relu(output)
        output = self.leaky_relu(self.deconv2(output))
        return output


class DecBlock_output(nn.Module):
    def __init__(self, nin, nout, upsample=True, kernel=3):
        super(DecBlock_output, self).__init__()
        self.upsample = upsample
        padding = kernel // 2

        if upsample:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=2, padding=padding)
        else:
            self.deconv1 = nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=kernel, stride=(2, 1), padding=padding)
        self.deconv2 = nn.ConvTranspose2d(in_channels=nout, out_channels=nout, kernel_size=kernel, stride=1, padding=padding)
        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, input, out_size):
        output = self.deconv1(input, output_size=out_size)
        output = self.leaky_relu(output)
        output = self.deconv2(output)
        return output


class AE(nn.Module):
    def __init__(self, downsample=True, in_channel=1, kernel=3):
        super(AE, self).__init__()
        self.enc_blc1 = EncBlock(nin=in_channel, nout=32, downsample=downsample, kernel=kernel)
        self.enc_blc2 = EncBlock(nin=32, nout=64, downsample=downsample, kernel=kernel)
        self.enc_blc3 = EncBlock(nin=64, nout=128, downsample=downsample, kernel=kernel)
        self.enc_blc4 = EncBlock(nin=128, nout=256, downsample=downsample, kernel=kernel)
        self.enc_blc5 = EncBlock(nin=256, nout=256, downsample=downsample, kernel=kernel)

        self.dec_blc1 = DecBlock(nin=256, nout=256, upsample=downsample, kernel=kernel)
        self.dec_blc2 = DecBlock(nin=256, nout=128, upsample=downsample, kernel=kernel)
        self.dec_blc3 = DecBlock(nin=128, nout=64, upsample=downsample, kernel=kernel)
        self.dec_blc4 = DecBlock(nin=64, nout=32, upsample=downsample, kernel=kernel)
        self.dec_blc5 = DecBlock_output(nin=32, nout=1, upsample=downsample, kernel=kernel)

    def forward(self, input):
        # input: [bs, c, d, T]
        x_down1 = self.enc_blc1(input) 
        x_down2 = self.enc_blc2(x_down1)
        x_down3 = self.enc_blc3(x_down2) 
        x_down4 = self.enc_blc4(x_down3)
        z = self.enc_blc5(x_down4)

        x_up4 = self.dec_blc1(z, x_down4.size())
        x_up3 = self.dec_blc2(x_up4, x_down3.size())  
        x_up2 = self.dec_blc3(x_up3, x_down2.size()) 
        x_up1 = self.dec_blc4(x_up2, x_down1.size()) 
        output = self.dec_blc5(x_up1, input.size())

        return output, z


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN_Encoder(nn.Module):
    def __init__(self, downsample=True, in_channel=1, kernel=3):
        super(CNN_Encoder, self).__init__()
        self.enc_blc1 = EncBlock(nin=in_channel, nout=32, downsample=downsample, kernel=kernel)
        self.enc_blc2 = EncBlock(nin=32, nout=64, downsample=downsample, kernel=kernel)
        self.enc_blc3 = EncBlock(nin=64, nout=128, downsample=downsample, kernel=kernel)
        self.enc_blc4 = EncBlock(nin=128, nout=256, downsample=downsample, kernel=kernel)
        self.enc_blc5 = EncBlock(nin=256, nout=256, downsample=downsample, kernel=kernel)

    def forward(self, input):
        x_down1 = self.enc_blc1(input) 
        x_down2 = self.enc_blc2(x_down1) 
        x_down3 = self.enc_blc3(x_down2) 
        x_down4 = self.enc_blc4(x_down3) 
        z = self.enc_blc5(x_down4) 
        size_list = [x_down4.size(), x_down3.size(), x_down2.size(), x_down1.size(), input.size()]
        return z, size_list


class CNN_Decoder(nn.Module):
    def __init__(self, downsample=True, kernel=3):
        super(CNN_Decoder, self).__init__()
        self.dec_blc1 = DecBlock(nin=512, nout=256, upsample=downsample, kernel=kernel)
        self.dec_blc2 = DecBlock(nin=256, nout=128, upsample=downsample, kernel=kernel)
        self.dec_blc3 = DecBlock(nin=128, nout=64, upsample=downsample, kernel=kernel)
        self.dec_blc4 = DecBlock(nin=64, nout=32, upsample=downsample, kernel=kernel)
        self.dec_blc5 = DecBlock_output(nin=32, nout=1, upsample=downsample, kernel=kernel)

    def forward(self, z, size_list):
        x_up4 = self.dec_blc1(z, size_list[0]) 
        x_up3 = self.dec_blc2(x_up4, size_list[1]) 
        x_up2 = self.dec_blc3(x_up3, size_list[2]) 
        x_up1 = self.dec_blc4(x_up2, size_list[3])  
        output = self.dec_blc5(x_up1, size_list[4])
        return output


class Motion_CNN_CVAE(nn.Module):
    def __init__(self, nz, downsample=True, in_channel=1, kernel=3, clip_seconds=2):
        super(Motion_CNN_CVAE, self).__init__()
        self.nz = nz  # dim of latent variables
        self.enc_conv_input = CNN_Encoder(downsample, in_channel, kernel)
        self.enc_conv_gt = CNN_Encoder(downsample, in_channel, kernel)
        self.enc_conv_cat = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel, stride=1, padding=kernel//2, padding_mode='replicate'),
            nn.LeakyReLU(0.2),
            Flatten(),
        )
        self.enc_mu = nn.Linear(512*8*clip_seconds, self.nz)
        self.enc_logvar = nn.Linear(512*8*clip_seconds, self.nz)
        self.dec_dense = nn.Linear(self.nz,  256*8*clip_seconds)
        self.dec_conv = CNN_Decoder(downsample, kernel)
        

    def encode(self, x, y):
        # x: [bs, c, d, T] (input)
        # y: [bs, c, d, T] (gt)
        e_x, _ = self.enc_conv_input(x) 
        e_y, _ = self.enc_conv_gt(y) 
        e_xy = torch.cat((e_x, e_y), dim=1) 
        z = self.enc_conv_cat(e_xy)
        z_mu = self.enc_mu(z)
        z_logvar = self.enc_logvar(z)
        return z_mu, z_logvar

    def reparameterize(self, mu, logvar, eps):
        std = torch.exp(0.5*logvar)
        return mu + eps * std

    def decode(self, x, z):
        e_x, size_list = self.enc_conv_input(x)
        d_z_dense = self.dec_dense(z)
        d_z = d_z_dense.view(e_x.size(0), e_x.size(1), e_x.size(2), e_x.size(3))
        d_xz = torch.cat((e_x, d_z), dim=1)
        y_hat = self.dec_conv(d_xz, size_list)
        return y_hat

    def forward(self, input, gt=None, is_train=True, z=None, is_twice=None):
        # input: [bs, c, d, T]
        self.bs = len(input)
        if is_train:
            mu, logvar = self.encode(input, gt)
            eps = torch.randn_like(logvar)
            z = self.reparameterize(mu, logvar, eps) #if self.training else mu
        else:
            if is_twice:
                mu, logvar = self.encode(input, gt)
                z = mu
            else:
                if z is None:
                    z = torch.randn((self.bs, self.nz), device=input.device)
                mu = 0
                logvar = 1

        pred = self.decode(input, z)
        return pred, mu, logvar

    def sample_prior(self, x):
        z = torch.randn((x.shape[1], self.nz), device=x.device)
        return self.decode(x, z)
        