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

structures = {"vgg16":25088,
             "densenet121":1024
             }
def load_data(directory):
	#data_dir = 'flowers'
	data_dir = directory
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	# TODO: Define your transforms for the training, validation, and testing sets
	#data_transforms = 
	train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
	                                       transforms.RandomResizedCrop(224),
	                                       transforms.RandomHorizontalFlip(),
	                                       transforms.ToTensor(),
	                                       transforms.Normalize([0.485, 0.456, 0.406], 
	                                                            [0.229, 0.224, 0.225])])

	test_data_transforms = transforms.Compose([transforms.Resize(256),
	                                      transforms.CenterCrop(224),
	                                      transforms.ToTensor(),
	                                      transforms.Normalize([0.485, 0.456, 0.406], 
	                                                           [0.229, 0.224, 0.225])])

	validation_data_transforms = transforms.Compose([transforms.Resize(256),
	                                            transforms.CenterCrop(224),
	                                            transforms.ToTensor(),
	                                            transforms.Normalize([0.485, 0.456, 0.406], 
	                                                                 [0.229, 0.224, 0.225])]) 


	# TODO: Load the datasets with ImageFolder
	#image_datasets = 
	train_image_datasets = datasets.ImageFolder(train_dir, transform = train_data_transforms)
	validation_image_datasets = datasets.ImageFolder(valid_dir, transform = validation_data_transforms)
	test_image_datasets = datasets.ImageFolder(test_dir, transform = test_data_transforms)

	# TODO: Using the image datasets and the trainforms, define the dataloaders
	#dataloaders =
	train_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size = 64, shuffle = True)
	validation_loader = torch.utils.data.DataLoader(validation_image_datasets, batch_size = 32, shuffle = True)
	test_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size = 32, shuffle = True)

	return train_loader, validation_loader, test_loader



# TODO: Build and train your network

def net_setup(structure = 'vgg16', dropout = 0.5, hidden_layer1 = 120, lr = 0.001, device='gpu'):
    if structure == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained = True)
    else:
        print("Invalid Model: Please try vgg16 or densenet121 ")
        
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('dropout',nn.Dropout(dropout)),
        ('inputs', nn.Linear(structures[structure], hidden_layer1)),
        ('relu1', nn.ReLU()),
        ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
        ('relu2',nn.ReLU()),
        ('hidden_layer2',nn.Linear(90,80)),
        ('relu3',nn.ReLU()),
        ('hidden_layer3',nn.Linear(80,102)),
        ('output', nn.LogSoftmax(dim=1))
                      ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )

    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()

    return model, optimizer, criterion

model, optimizer, criterion = net_setup('densenet121')

def train_network(model, criterion, optimizer, epochs = 3, print_every=20, loader=0, device='gpu'):
	steps = 0

	for e in range(epochs):
	    running_loss = 0
	    for ii, (inputs, labels) in enumerate(loader):
	        steps += 1
	        
	        if torch.cuda.is_available() and device == 'gpu':
	        	inputs,labels = inputs.to('cuda'), labels.to('cuda')
	        
	        optimizer.zero_grad()
	        
	        # Forward and backward passes
	        outputs = model.forward(inputs)
	        loss = criterion(outputs, labels)
	        loss.backward()
	        optimizer.step()
	        
	        running_loss += loss.item()
	        
	        if steps % print_every == 0:
	            model.eval()
	            vlost = 0
	            accuracy=0
	   
	            for ii, (inputs2,labels2) in enumerate(validation_loader):
	                optimizer.zero_grad()
	                
	                inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
	                model.to('cuda:0')
	                with torch.no_grad():    
	                    outputs = model.forward(inputs2)
	                    vlost = criterion(outputs,labels2)
	                    ps = torch.exp(outputs).data
	                    equality = (labels2.data == ps.max(1)[1])
	                    accuracy += equality.type_as(torch.FloatTensor()).mean()
	                    
	            vlost = vlost / len(validation_loader)
	            accuracy = accuracy /len(validation_loader)
	  
	            print("Epoch: {}/{}... ".format(e+1, epochs),
	                  "Loss: {:.4f}".format(running_loss/print_every),
	                  "Validation Lost {:.4f}".format(vlost),
	                   "Accuracy: {:.4f}".format(accuracy))

	            running_loss = 0


# TODO: Save the checkpoint 
def save_checkpoint(path='checkpoint.pth',structure ='densenet121', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=12):
	model.class_to_idx = train_image_datasets.class_to_idx
	model.cpu
	torch.save({'structure' :'densenet121',
	            'hidden_layer1':120,
	            'dropout':dropout,
	            'lr':lr,
	            'no_of_epochs':epochs,
	            'state_dict':model.state_dict(),
	            'class_to_idx':model.class_to_idx},
	            path)

def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']

    model,_,_ = nn_setup(structure , dropout,hidden_layer1,lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    proc_img = Image.open(image_path)
   
    prepoceess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # TODO: Process a PIL image for use in a PyTorch model
    pymodel_img = prepoceess_img(proc_img)
    return pymodel_img


def predict(image_path, model=0, topk=5, device='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if torch.cuda.is_available() and device =='gpu':
    	model.to('cuda:0')

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if device == 'gpu':
    	with torch.no_grad():
        	output = model.forward(img_torch.cuda())

    else:
    	with torch.no_grad():
            output=model.forward(img_torch)
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)
