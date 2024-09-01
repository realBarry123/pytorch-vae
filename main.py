import torch
from model import VAE

net = VAE()
print(net.encoder(torch.tensor([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.0, 2.0, 3.0, 4.0, 5.0]
])))