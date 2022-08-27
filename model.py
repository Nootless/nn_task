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

# LSTM Model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_labels):
        super(LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.num_labels = num_labels
        self.lstm = nn.LSTM(input_size, hidden_size,batch_first=True, num_layers=num_layers, dropout=.35)
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        h_t = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float)
        c_t = torch.zeros(self.num_layers, self.hidden_size, dtype=torch.float)
        # print(x.shape,x.size(1),h_t.size(0))
        output,_ = self.lstm(x, (h_t, c_t))
        # batch,seq,feature
        # 1000 samples, 
        return self.fc(output)

# Data Loader
class Sets(Dataset):
    def __init__(self,data,transforms=None):
        # to scale data
        # only features are scaled, labels are not
        self.sc = StandardScaler()
        self.x = self.sc.fit_transform(np.float32(data[:, 6:])).reshape(-1,1000)
        self.y = np.float32(data[:,-2:].reshape(-1,2))

    def __getitem__(self,index):
      return torch.from_numpy(self.x[index]),torch.from_numpy(self.y[index])

    def __len__(self):
      return len(self.y)

# Parameters
input_size = 1000
hidden_size = 1000
num_layers = 2
num_labels = 2
batch_size = 22000
test_batch_size = 9468
num_epochs = 10
learning_rate = .01
epoch = 1
if __name__ == '__main__':
    # instatiate model, pull from file if exists
    if exists('pikl.pkl'):
        model = pick.load(open('pikl1.pkl','rb'))
    else:
        model = LSTM(input_size,hidden_size,num_layers,num_labels)

    # initialize loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load and adjust test data
    day_data = np.load('day_data_train.npy')
    day_data = pd.DataFrame(day_data,index=np.reshape(day_data[:,-6], (1,-1)).flatten())
    # add row to keep even rows
    day_data = day_data.append(day_data.iloc[-1])
    day_data = Sets(day_data.to_numpy())

    # Test Data initialization and clean up
    test_data = np.load('day_data_test.npy')
    time = test_data[:,-6]
    test_data = pd.DataFrame(test_data,index=np.reshape(test_data[:,-6], (1,-1)).flatten())
    test_data = Sets(test_data.to_numpy())

    # Load data
    trainloader = DataLoader(dataset=day_data,batch_size = batch_size, shuffle=False)
    testloader = DataLoader(dataset=test_data, batch_size = test_batch_size ,shuffle=False)

    for i in range(epoch):
        # load data from data loader in batch 
        for batch_idx, (data, target) in enumerate(trainloader):
            # print(data.shape)
            # print(target.shape)
            # set training and zero gradients
            model.train()
            optimizer.zero_grad()
            
            print('Model Training...')
            output = model(data)

            print('Calculating Losses....')
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            print(f'EPOCH: {i} \\\ MAE LOSS: {loss}')
            
            # Saving array
            np.save('output.npy',output.detach().numpy())
            # test loop
            for batch_idx, (data,target) in enumerate(testloader):
                model.eval()
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output,target)
                print(f'TEST MAE LOSS: {loss}')    
                     
    with open(f'pikl1.pkl','wb') as file:
        pick.dump(model,file)