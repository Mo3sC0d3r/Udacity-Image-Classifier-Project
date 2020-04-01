# Imports here
import  numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL
from PIL import Image
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter	
from collections import OrderedDict
import argparse

import utilities

ap = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments
ap.add_argument('data_dir', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
directory = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
device = pa.gpu
epochs = pa.epochs

def Main():
    train_loader, validation_loader, test_loader = utilities.load_data(directory)
    model, optimizer, criterion = utilities.net_setup(structure,dropout,hidden_layer1,lr,device)
    utilities.train_network(model, optimizer, criterion, epochs, 20, train_loader, device)
    utilities.save_checkpoint(path,structure,hidden_layer1,dropout,lr)
    print("**************Training Complete !! Thanks for the patience******************") 
if __name__ =="__main__":
    Main()