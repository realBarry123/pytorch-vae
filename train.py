import random
import torch
from model import VAE

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

chords = augment(chords)

net = VAE()
print(torch.tensor(augment(chords)).shape)
print(net(torch.tensor(chords)))