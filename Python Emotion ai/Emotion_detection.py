#import libaries
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
#change path to working folder
os.chdir("C:\\Users\\lucas\\Desktop\\Intelligente-Systemer-Projekt\\Python Emotion ai")
#get working path in variable
cwd = os.getcwd()
#load data from dataset files
train_data = torch.load(cwd+"/train_data.pt")
test_data = torch.load(cwd+"/test_data.pt")

#weightet sampler for significance of data
class_sample_count = [3995,436,4097,7215,4830,3171,4965] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
weights = 1. / torch.Tensor(class_sample_count)
# Accuracy is also determined by the batch for both test and training, so keeps this in mind 
#and take sizeable batch, should still be determined by the . 
batchsize = 1000
#trainloader
train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
#remember to update accuracy with the batch size
test_loader = DataLoader(test_data, batch_size=batchsize)

#Create neural network model using nn.sequential
net = torch.nn.Sequential(
    
#the structure of the network is inspired by networks training on this model on kaggle, written with keras
torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
torch.nn.BatchNorm2d(32),
torch.nn.ReLU(),

torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
torch.nn.BatchNorm2d(64),
torch.nn.ReLU(),

torch.nn.MaxPool2d(kernel_size=2),

torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
torch.nn.BatchNorm2d(128),
torch.nn.ReLU(),

torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
torch.nn.BatchNorm2d(128),
torch.nn.ReLU(),

torch.nn.MaxPool2d(kernel_size=2),


torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
torch.nn.BatchNorm2d(128),
torch.nn.ReLU(),

torch.nn.MaxPool2d(kernel_size=2),

torch.nn.Flatten(),
torch.nn.Linear(in_features=1152, out_features=7)
)
#load pre trained model parameters if relevant
#net.load_state_dict(torch.load(cwd+"/model_parameters.pt"))

#Setup loss function with the relevant weights
loss = torch.nn.CrossEntropyLoss(weight=weights)
#loss = torch.nn.CrossEntropyLoss()
#setup optimizers
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
#Maximum number of times running throught the entire dataset
num_epochs = 30
#setup empty empty lists for storing 
training_loss = []
test_loss = []
test_accuracy = []
training_accuracy = []

trigger_times = 0
#how many times in row the value doesent improve significantly
patience = 5
#variable for see last loss for comparison
the_last_loss = 0


for epoch in range(num_epochs):
    print(f"epoch {epoch} out of {num_epochs}")
    for i, (x,y) in enumerate(train_loader):
        out = net(x)
        l = loss(out, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
        the_current_loss = l
        
        training_loss += [l.detach().numpy()]
        training_accuracy += [torch.sum(torch.argmax(out, dim=1) == y)/batchsize*100]
        a, b = next(iter(test_loader))
        t_out = net(a)
        test_accuracy += [torch.sum(torch.argmax(t_out, dim=1) == b)/batchsize*100]
        
        test_l = loss(t_out, b)
        print(f"test_loss = {test_l}")
        test_loss += [test_l.detach().numpy()]
        
        # Early stopping
        print('The current loss:', the_current_loss)
        print('The current accuracy:', test_accuracy[-1])

        if the_current_loss > the_last_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                #save model parameters
                torch.save(net.state_dict(), cwd+"/model_parameters.pt")
                #save loss test and training
                np.savetxt(cwd+"/test_loss.csv", test_loss, delimiter=",")
                np.savetxt(cwd+"/training_loss.csv", training_loss, delimiter=",")
                np.savetxt(cwd+"/accuracy.csv", test_accuracy, delimiter=",")
                exit()

        else:
            print('trigger times: 0')
            trigger_times = 0

        the_last_loss = the_current_loss
        
        
#save model parameters
torch.save(net.state_dict(), cwd+"/model_parameters.pt")
#save loss and accuracy (test and training) in csv files
np.savetxt(cwd+"/test_loss.csv", test_loss, delimiter=",")
np.savetxt(cwd+"/training_loss.csv", training_loss, delimiter=",")
np.savetxt(cwd+"/test_accuracy.csv", test_accuracy, delimiter=",")
np.savetxt(cwd+"/training_accuracy.csv", training_accuracy, delimiter=",")


#test and common functions 

#test
#x,y = next(iter(test_loader))
#out = net(x)
#plt.imshow(x[0][0], cmap='gray')



#torch.sum(torch.argmax(out, dim=1) == y)/1000

#save model parameters
#torch.save(net.state_dict(), cwd+"/model_parameters.pt")
#load
#net.load_state_dict(torch.load(cwd+"/model_parameters.pt"))