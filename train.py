import random
import torch
import torch.nn as nn
from model import VAE, weights_init

learning_rate = 0.0002
batch_size = 16

def augment(list):
    new_list = []
    for chord in list:
        for _ in range(16):
            transposition = random.random() * 2
            new_list.append([x * transposition for x in chord])
    return new_list

chords = [
    [130.81, 196.0, 261.63, 329.63, 493.88],
    [130.81, 196.0, 261.63, 349.23, 466.16],
]

chords = torch.tensor(augment(chords))
chord_loader = torch.utils.data.DataLoader(chords, batch_size=batch_size, shuffle=True)

def kl_divergence(mean, log_var):
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return kl_loss

mse_loss = nn.MSELoss(reduction='sum')
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = mse_loss(x_hat, x)
    print(reproduction_loss.item())
    KLD = kl_divergence(mean, log_var)
    #print(KLD.item())

    return reproduction_loss + KLD

net = VAE()
weights_init(net)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

def train():

    net.train()

    for i, data in enumerate(chord_loader, start=0):

        optimizer.zero_grad()

        reconstructed, mean, log_var = net(data)

        loss = loss_function(data, reconstructed, mean, log_var)

        loss.backward()

        optimizer.step()

        # print(loss.item())

for i in range(100):
    train()