"""Reimplement TimeGAN-pytorch Codebase into full pytorch code.

-----------------------------

discriminative_metrics.py

Note: Use post-hoc RNN to classify original data and synthetic data

Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

# Necessary Packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def custom_accuracy_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same.")
    
    correct_predictions = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    accuracy = correct_predictions / len(y_true)
    
    return accuracy


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
    
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h_out = self.rnn(x)
        h_out = h_out[-1].squeeze(0)
        
        y_logit = self.fc(h_out)
        y = torch.sigmoid(y_logit)
        return y


def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data
    
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        
    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    no, seq_len, dim = ori_data.shape

    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    # # MinMax scale data R->(0,1)
    # scaler = MinMaxScaler()
    # scaler.fit(np.vstack((ori_data, generated_data)).reshape(-1,1))
    # ori_data = scaler.transform(ori_data.reshape(-1,1)).reshape(no,seq_len,dim)
    # generated_data = scaler.transform(generated_data.reshape(-1,1)).reshape(no,seq_len,dim)

    X = np.vstack((ori_data, generated_data))
    y = np.hstack((np.ones(ori_data.shape[0]), np.zeros(generated_data.shape[0])))

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True)
    train_dataset = CustomDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = CustomDataset(torch.tensor(X_test), torch.tensor(y_test))

    # Network parameters
    dim = X.shape[-1]
    hidden_dim = 5 if dim == 1 else int(dim / 2)
    num_epochs = 500
    batch_size = 64

    # Create Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, loss, and optimizer
    model = Discriminator(input_dim=dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
 
    
    # Training step
    model.train()
    epoch_progress_bar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in epoch_progress_bar:

        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output.squeeze(1), label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_progress_bar.set_postfix(loss=loss.item())


    model.eval()
    with torch.no_grad():
        pred = model(test_dataset.X().to(device)).squeeze(1)
        acc = accuracy_score(test_dataset.y(), (pred.cpu()>0.5))

    print(f'Test Accuracy: {acc*100}%')

    discriminative_score = np.abs(0.5 - acc)
    return discriminative_score, acc, acc