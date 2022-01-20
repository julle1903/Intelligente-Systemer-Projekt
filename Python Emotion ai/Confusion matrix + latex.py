import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
#change working directory to cunrrent to specified directory
os.chdir("C:\\Users\\lucas\\Desktop\\Intelligente-Systemer-Projekt\\Python Emotion ai")
#get path to variable
cwd = os.getcwd()
#load data from dataset files
train_data = torch.load(cwd+"/train_data.pt")
test_data = torch.load(cwd+"/test_data.pt")

#trainloader
train_loader = DataLoader(train_data, batch_size=500)
#remember to update accuracy with the batch size
test_loader = DataLoader(test_data, batch_size=500)

#Create neural network model using nn.sequential same as from the training script
net = torch.nn.Sequential(
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

#load traning parameters
net.load_state_dict(torch.load(cwd+"/model_parameters_real.pt"))
#setup empty data matrixer
confusion_matrix = np.zeros((7,7))
confusion_matrix_percentage = np.zeros((7,7))
solution_total = np.zeros(7)

#loop for running throught batches of the test data
for i, (x,y) in enumerate(train_loader):
    # calculate nn output
    out = net(x)
    #get the max value of the arrays to get the concrete guesses
    all_guess = torch.argmax(out, dim=1)
    #loop through all the datapoints in the batch
    for i in range(len(out)):
        #get the value of the guesses and labels
        guess = all_guess[i].detach()
        solution = y[i].detach()
        #add value to the specified place in the confusionmatrix
        confusion_matrix[solution][guess] += 1;
        #track how many of each labeled emotion there is.
        solution_total[solution] += 1;        
#divide to get percentage confusionmatrix
sum_correct = 0
total = solution_total.sum();
for i in range(7):
    sum_correct += confusion_matrix[i][i]
    confusion_matrix_percentage[i] = confusion_matrix[i]/solution_total[i]*100
#From start to here is written by Lucas Sylvester (s214636)

# generate latex table - This part of the code is writte by Lucas Emken (s214625)
print('\\begin{table}[]')
print('\\hspace{1cm}Expected')
print('\\\\')
print('\\rotatebox{90}{Predicted}')
print('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}')
print('\\hline')

topstr = ""
for i in range(7):
    topstr += ' & ' + str(i+1)
print (topstr, "\\\\")

row = 0
for i in confusion_matrix_percentage:
    row+=1
    print('\\hline')
    elestr = ""
    for j in i:
        elestr += "\\cellcolor{blue!"+ str(j)+ "}"+ str(round(j,2))+ " & "
    print(row, '&', elestr[:-2], "\\\\")
print('\\hline')
print('\\end{tabular}')
print('\\caption{your caption}')
print('\\end{table}')
test_accuracy = sum_correct/total*100
print(f"Test accuracy with {total} entries is:{test_accuracy}%")