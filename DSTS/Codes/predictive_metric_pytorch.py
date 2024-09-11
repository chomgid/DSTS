"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

predictive_metrics.py

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].float(), self.labels[idx].float()

    def X(self):
        return self.data.float()

    def y(self):
        return self.labels.float()


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc =  nn.Linear(hidden_dim, 1) # fully connected 
        self.fc_2 = nn.Linear(128, 1) # fully connected last layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, h_n = self.gru(x)
        y = self.fc(out)
        y = self.sigmoid(y)
        return y
 

def predictive_score_metrics(ori_data, generated_data):
    """Report the performance of Post-hoc RNN one-step ahead prediction.
    
    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      
    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    no, seq_len, dim = ori_data.shape

    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    # # MinMax scale data R->(0,1)
    # scaler = MinMaxScaler()
    # scaler.fit(np.vstack((ori_data, generated_data)).reshape(-1,1))
    # ori_data = scaler.transform(ori_data.reshape(-1,1)).reshape(no,seq_len,dim)
    # generated_data = scaler.transform(generated_data.reshape(-1,1)).reshape(no,seq_len,dim)

    # create train and test dataset
    X = torch.tensor(generated_data[:,:-1,:])
    y = torch.tensor(generated_data[:,1:,:])
    train_dataset = CustomDataset(X,y)
    test_dataset = CustomDataset(torch.tensor(ori_data[:,:-1,:]), torch.tensor(ori_data[:,1:,:]))
    
    # Network parameters
    hidden_dim = 10 if dim == 1 else int(dim / 2)
    num_epochs = 500
    batch_size = 400

    # Create Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, loss function, and optimizer
    model = Predictor(input_dim=dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.L1Loss()  
    optimizer = optim.Adam(model.parameters())
    
    # Training using Synthetic dataset
    model.train()
    epoch_progress_bar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in epoch_progress_bar:
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
        
            output = model(x)
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_progress_bar.set_postfix(loss=loss.item())
    
    # Test the trained model on the original data
    model.eval()
    with torch.no_grad():
        y_pred = model(test_dataset.X().to(device))
        
        # Compute the performance in terms of MAE
        MAE_temp = 0
        for i in range(len(test_dataset)):
            MAE_temp += mean_absolute_error(test_dataset.y()[i], y_pred.cpu()[i])
        
        predictive_score = MAE_temp / len(test_dataset)

    # y_test = scaler.inverse_transform(test_dataset.y().reshape(-1,1)).reshape(no,seq_len-1,dim)
    # y_pred = scaler.inverse_transform(y_pred.cpu().reshape(-1,1)).reshape(no,seq_len-1,dim)
    y_test = test_dataset.y().cpu()

    return predictive_score, y_test, y_pred
    