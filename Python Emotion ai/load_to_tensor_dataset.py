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

data = pd.read_csv('fer2013.csv')
#data.info()

all_img_array = data.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
all_emotion_array = data.emotion
all_usage = data.Usage

train_pixels = []
train_solution = []

test_pixels = []
test_solution = []

for datapoint in range(len(all_usage)):
    if all_usage[datapoint] == "Training":
        train_pixels.append(all_img_array[datapoint])
        train_solution.append(all_emotion_array[datapoint])
    else:
        test_pixels.append(all_img_array[datapoint])
        test_solution.append(all_emotion_array[datapoint])   
#convert to correct format (number of datapoint, colors, size, size )
train_pixels = torch.FloatTensor(np.moveaxis(np.array(train_pixels),3,1))
test_pixels = torch.FloatTensor(np.moveaxis(np.array(test_pixels),3,1))

test_solution = torch.LongTensor(np.array(test_solution))
train_solution = torch.LongTensor(np.array(train_solution))


train_data = TensorDataset(train_pixels, train_solution);
test_data = TensorDataset(test_pixels, test_solution);

torch.save(train_data, cwd+"/train_data.pt")
torch.save(test_data, cwd+"/test_data.pt")


