import math, numpy
import torch
import torch.nn as nn
import torch.nn.parallel

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # find convolutional layer
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # randomly initialize weights
    elif classname.find('BatchNorm') != -1:  # find batchnorm
        nn.init.normal_(m.weight.data, 1.0, 0.02)  # randomly initialize weights
        nn.init.constant_(tensor=m.bias.data, val=0)  # set bias to 0

class VAE(nn.Module):

    def __init__(self):

        super(VAE, self).__init__()

        self.linear = nn.Linear(in_features=5, out_features=5)

        self.mean = nn.Linear(in_features=5, out_features=2)
        self.log_var = nn.Linear(in_features=5, out_features=2)

        self.expand = nn.Linear(in_features=2, out_features=5)

        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()

    def encoder(self, x):

        x = self.linear(x)
        x = self.leaky_relu(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        epsilon = torch.randn_like(log_var).to("cpu")

        sd = torch.exp(0.5 * log_var)
        z = mean + sd * epsilon

        return z, mean, log_var

    def decoder(self, x):

        x = self.expand(x)

        return x

    def forward(self, x):

        x, mean, log_var = self.encoder(x)
        x = self.decoder(x)

        return x, mean, log_var