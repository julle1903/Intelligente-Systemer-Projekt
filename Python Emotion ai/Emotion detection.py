import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir("C:\\Users\\lucas\\Desktop\\Intelligente-Systemer-Projekt\\Python Emotion ai")
cwd = os.getcwd()
#use download=True if not already downloaded
#train_data = datasets.MNIST(root='data', train=True, transform=ToTensor(),download=True)
#test_data = datasets.MNIST(root='data', train=False, transform=ToTensor(),download=True)
#data

train_data = torch.load(cwd+"/train_data.pt")
test_data = torch.load(cwd+"/test_data.pt")

#trainloader
train_loader = DataLoader(train_data, batch_size=500)
test_loader = DataLoader(test_data, batch_size=5000)

#middle layer channels, antal feature maps
mid_layer=100

#Create neural network model using nn.sequential
net = torch.nn.Sequential(
torch.nn.Conv2d(in_channels=1, out_channels=mid_layer, kernel_size=3),
torch.nn.ReLU(),
torch.nn.MaxPool2d(kernel_size=2),
torch.nn.Conv2d(in_channels=mid_layer, out_channels=50, kernel_size=6),
torch.nn.ReLU(),
torch.nn.MaxPool2d(kernel_size=2),
torch.nn.Flatten(),
torch.nn.Linear(in_features=4050, out_features=7)
)


loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

num_epochs = 30
all_loss = []



for epoch in range(num_epochs):
    for i, (x,y) in enumerate(train_loader):
        out = net(x)
        l = loss(out, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
        all_loss += [l.detach().numpy()]
        if i%10 == 0:
            #print(f"loss = {l}")
            plt.plot(np.array(all_loss))
            plt.show()
            all_loss = []
            
        test_loss = 0.
        #for x, y in test_loader:
         #   out = net(x)
          #  testl = loss(out, y)
           # test_loss += testl
            
            #print(f"test_loss = {testl}")
        #accuracy
        a, b = next(iter(test_loader))
        outa = net(a)
        accuracy = torch.sum(torch.argmax(outa, dim=1) == b)/5000
        print(accuracy*100)
        


#test
#x,y = next(iter(test_loader))
#out = net(x)
#plt.imshow(x[0][0], cmap='gray')



#torch.sum(torch.argmax(out, dim=1) == y)/1000

#save model parameters
#torch.save(net.state_dict(), cwd+"/model_parameters.pt")
#load
#net.load_state_dict(torch.load(cwd+"/model_parameters.pt"))




