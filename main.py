import torch
import pyautogui
from time import sleep
from model import VAE

net = VAE()
net.load_state_dict(torch.load("Models/net.pkl"))


def vec_to_chord(x, y):

    chord = net.decoder(torch.tensor([float(x), float(y)])).detach()
    new_chord = []
    for note in chord:
        note = note * (1046.48 - 130.81) + 130.81
        new_chord.append(note)
    return new_chord


def test_with_mouse():

    while True:
        mouse_x, mouse_y = pyautogui.position()
        max_x, max_y = pyautogui.size()
        print(mouse_x/max_x, mouse_y/max_y)
        print(vec_to_chord(mouse_x/max_x, mouse_y/max_y))

        sleep(0.2)

def export(model, name):

    onnx_program = torch.onnx.dynamo_export(model, torch.randn(1, 5))
    onnx_program.save(name)
    
export(net, "chord_vae.onnx")