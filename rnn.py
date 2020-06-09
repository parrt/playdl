import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.functional as F
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def train_test_split(X, y, test_size:float):
    n = len(X)
    shuffle_idx = np.random.permutation(range(n))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    n_valid = int(n*test_size)
    n_train = n - n_valid
    X_train, X_valid = X[0:n_train].to(device), X[n_train:].to(device)
    y_train, y_valid = y[0:n_train].to(device), y[n_train:].to(device)
    return X_train, X_valid, y_train, y_valid

def plot_history(history, yrange=(0.0, 5.00), figsize=(3.5,3)):
    plt.figure(figsize=figsize)
    plt.ylabel("Sentiment log loss")
    plt.xlabel("Epochs")
    loss = history[:,0]
    valid_loss = history[:,1]
    plt.plot(loss, label='train_loss')
    plt.plot(valid_loss, label='val_loss')
    # plt.xlim(0, 200)
    plt.ylim(*yrange)
    plt.legend(loc='lower right')
    plt.show()


def normal_transform(x, mean=0.0, std=0.01):
    "Convert x to have mean and std"
    return x*std + mean

def randn(n1, n2,
          device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
          dtype=torch.float,
          mean=0.0, std=0.01, requires_grad=False):
    x = torch.randn(n1, n2, device=device, dtype=dtype)
    x = normal_transform(x, mean=mean, std=std)
    x.requires_grad=requires_grad
    return x

def rtrain(model:nn.Module, train_data:TensorDataset, valid_data:TensorDataset,
           epochs=350,
           test_size=0.20,
           learning_rate = 0.002,
           batch_size=32,
           weight_decay=1.e-4,
           loss_fn=nn.MSELoss(),
           metric=nn.MSELoss(),
           print_every=30):
    "Train a regressor"
    history = []
    train_loader = DataLoader(train_data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for ei in range(epochs): # epochs
        for bi, (batch_x, batch_y) in enumerate(train_loader): # mini-batch
#             if len(batch_x)!=batch_size:
#                 print(f"\tBatch {bi:3d} len {len(batch_x)}")
            y_prob = model(batch_x)
            loss = loss_fn(y_prob, batch_y)

            optimizer.zero_grad()
            loss.backward() # autograd computes U.grad and M.grad
            optimizer.step()

        with torch.no_grad():
            loss        = loss_fn(model(train_data.tensors[0]), train_data.tensors[1])
            loss_valid  = loss_fn(model(valid_data.tensors[0]), valid_data.tensors[1])
            y_pred = model(train_data.tensors[0])
            metric_train = metric(torch.round(y_pred).cpu(), train_data.tensors[1].cpu())
            y_pred = model(valid_data.tensors[0])
            metric_valid = metric(torch.round(y_pred).cpu(), valid_data.tensors[1].cpu())

        history.append( (loss, loss_valid) )
        if ei % print_every == 0:
            print(f"Epoch {ei:3d} loss {loss:7.3f}, {loss_valid:7.3f}   {metric.__class__.__name__} {metric_train:4.3f}, {metric_valid:4.3f}")

    history = torch.tensor(history)
    return model, history

class RNN(nn.Module):
    def __init__(self, input_features=4, output_features=1, hidden_size=10):
        super(RNN, self).__init__()
        self.output_features = output_features
        self.hidden_size = hidden_size
        self.W_xh = randn(hidden_size, 1, std=0.01).double()
        self.W_hh = randn(hidden_size, hidden_size, std=0.01).double()
        self.W_hy = randn(output_features, hidden_size, std=0.01).double()
        self.W_xh = nn.Parameter(self.W_xh)
        self.W_hh = nn.Parameter(self.W_hh)
        self.W_hy = nn.Parameter(self.W_hy)

    def forward(self, x):
        print("x", x.shape)
        batch_size = x.shape[0]
        nfeatures = x.shape[1]
        o = torch.zeros((batch_size, 1)).double()
        for i in range(batch_size):
            # Reset hidden state (history) at start of every record
            h = torch.zeros((self.hidden_size, 1)).double()
            for j in range(nfeatures):  # for all input_features
                h = self.W_hh.mm(h) + self.W_xh * x[i, j]
                h = torch.relu(h)  # better than sigmoid for vanishing gradient
            o[i] = self.W_hy.mm(h)
        return o.reshape(batch_size, self.output_features)


dtype = torch.float
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

timesteps = 200
ncycles=5

x = np.linspace(0,ncycles*np.pi,timesteps)
siny = np.sin(2*x)
plt.plot(x,siny)

k = 8
X = []
y = []
for i in range(k,len(siny)):
    X.append(siny[i-k:i])
    y.append(siny[i])
X = torch.tensor(X).to(device)
y = torch.tensor(y).reshape(-1,1).to(device)
print(X.shape, y.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 0.20)

# test model
rnn = RNN(input_features=k).to(device)
y_pred = rnn(torch.tensor(X,device=device)).detach().cpu()
print(y_pred)