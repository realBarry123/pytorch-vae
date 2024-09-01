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

    def encoder(self, x):

        x = x.linear(in_features=5, out_features=5)
        mean = x.linear(in_features=5, out_features=2)
        log_var = x.linear(in_features=5, out_features=2)
        x = numpy.random.normal(loc=mean, scale=math.exp(log_var/2), size=2)

        return x

    def decoder(self, x):

        x = x.linear(in_features=2, out_features=5)
        x = x.linear(in_features=5, out_features=5)

        return x
