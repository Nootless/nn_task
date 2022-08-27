from model import *
import numpy as np
import pandas as pd
import pickle as pick
import torch
from torch import nn
from torchsummary import summary
from os.path import exists
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset,DataLoader
from matplotlib import pyplot as plt
from utils import *

# output function
output_file = 'output.npy'
data_set = 'day_data_test.npy'
test_batch_size = 9468

# predict function
def predict(data_set):
    with open('pikl1.pkl', 'rb') as f:
        # load data
        data_set =  np.load('day_data_test.npy')
        data_set = Sets(data_set)
        testloader = DataLoader(dataset=data_set, batch_size = test_batch_size ,shuffle=False)
        model = pick.load(f)
        
        # predicts and guess
        for batch_idx, (data,target) in enumerate(testloader):
            model.eval()
            guess = model(data)
            
            # clean up prediction and actual
            # ensure array format
            spp = np.reshape(guess.detach().numpy()[:,0], (-1, 1))
            dpp = np.reshape(guess.detach().numpy()[:,1], (-1, 1))
            target_sp = np.reshape(target.detach().numpy()[:,0], (-1, 1))
            target_dp = np.reshape(target.detach().numpy()[:,1], (-1, 1))
            
            # plot data
            plot_2vectors(spp, target_sp, 'sp')
            plot_2vectors(dpp, target_dp, 'dp')
            np.save('guess.npy', guess.detach().numpy())
            
        
if __name__ == '__main__':        
    predict(data_set)


