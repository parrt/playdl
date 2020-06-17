import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
#from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
np.set_printoptions(precision=2, suppress=True, linewidth=3000, threshold=20000)
from typing import Sequence
from fastai2.text.all import untar_data, URLs
import codecs
import os
import re
import string
from typing import Sequence
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3)

dtype = torch.float64

def randn(n1, n2, dtype=torch.float64, mean=0.0, std=0.01, requires_grad=False):
    x = torch.randn(n1, n2, dtype=dtype)
    x = x*std + mean # Convert x to have mean and std
    x.requires_grad=requires_grad
    return x

def get_text(filename:str):
    """
    Load and return the text of a text file, assuming latin-1 encoding as that
    is what the BBC corpus uses.  Use codecs.open() function not open().
    """
    f = codecs.open(filename, encoding='latin-1', mode='r')
    s = f.read()
    f.close()
    return s

def onehot(ci:int, vocab):
    v = torch.zeros((len(vocab),1), dtype=torch.float64)
    v[ci] = 1
    return v


def softmax(y):
    expy = np.exp(y)
    if len(y.shape)==1: # 1D case can't use axis arg
        return expy / np.sum(expy)
    return expy / np.sum(expy, axis=1).reshape(-1,1)

def cross_entropy(y_prob, y_true):
    """
    y_pred is n x k for n samples and k output classes and y_true is n x 1
    and is often softmax of final layer.
    y_pred values must be probability that output is a specific class.
    Binary case: When we have y_pred close to 1 and y_true is 1,
    loss is -1*log(1)==0. If y_pred close to 0 and y_true is 1, loss is
    -1*log(small value) = big value.
    y_true values must be positive integers in [0,k-1].
    """
    n = y_prob.shape[0]
    # Get value at y_true[j] for each sample with fancy indexing
    p = y_prob[range(n),y_true]
    return np.mean(-np.log(p))


path = untar_data(URLs.HUMAN_NUMBERS)
text = get_text(path / 'train.txt')
#text = text[:45_000]  # TESTING!!!
text = re.sub(r'[ \n]+', ' . ', text) # use '.' as separator token
tokens = text.split(' ')
tokens = tokens[:-1] # last token is blank '' so delete

# get unique vocab but don't sort; keep order so 'one'=1 etc...
v = set('.')
vocab = ['.']
for t in tokens:
    if t not in v:
        vocab.append(t)
        v.add(t)

index = {w:i for i,w in enumerate(vocab)}
tokens = [index[w] for w in tokens]

X = torch.tensor(tokens[0:-1])
y = torch.tensor(tokens[1:])

ntrain = int(len(X)*.50)
X_train, y_train = X[:ntrain], y[:ntrain]
X_valid, y_valid = X[ntrain:], y[ntrain:]

wtoi = {w:i for i, w in enumerate(vocab)}


class RNN(nn.Module):
    def __init__(self, nfeatures, nhidden, nclasses):
        super(RNN, self).__init__()
        self.nfeatures, self.nhidden, self.nclasses = nfeatures, nhidden, nclasses
        self.W = torch.eye(nhidden, nhidden, dtype=torch.float64)
        # self.W = randn(nhidden, nhidden)#, requires_grad=True)
        self.U = randn(nhidden, nfeatures)#, requires_grad=True)
        self.V = randn(nclasses, nhidden)#, requires_grad=True)
        self.W = nn.Parameter(self.W)
        self.U = nn.Parameter(self.U)
        self.V = nn.Parameter(self.V)

    def forward(self, h0, input):
        h = h0
        seq_outputs = torch.empty(len(input), len(vocab))
        for i in range(0, len(input)):
            x = onehot(input[i], vocab)
            h = self.W.mm(h) + self.U.mm(x)
            h = torch.tanh(h)  # squish to (-1,+1); also better than sigmoid for vanishing gradient
            # h = torch.relu(h)
            o = self.V.mm(h)
            seq_outputs[i] = o.reshape(-1)
        return h, seq_outputs


nhidden = 64
nfeatures = len(vocab)
nclasses = len(vocab)
seqlen = 16
n = len(X_train)

model = RNN(nfeatures, nhidden, nclasses)

learning_rate = 0.002
weight_decay = 0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

nepochs = 15
for epoch in range(nepochs + 1):
    losses = []
    h = torch.zeros(nhidden, 1, dtype=torch.float64, requires_grad=True)  # reset hidden state at start of epoch
    for p in range(0, n, seqlen):  # do one epoch
        h, seq_outputs = model(h, X_train[p:p + seqlen])
        h = h.detach()  # truncated BPTT; tell pytorch to forget prev h computations for dx purposes
        loss = F.cross_entropy(seq_outputs, y_train[p:p + seqlen])
        # myloss = cross_entropy(softmax(seq_outputs.detach().numpy()), y_train[p:p + seqlen].numpy())
        # print(loss.item(), myloss) # These are the same
        losses.append(loss.item())
        loss.backward()  # autograd computes U.grad and M.grad
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        optimizer.zero_grad()

    seq_loss = np.mean(losses)
    # print("Seq loss", seq_loss)#, np.sum(losses))

    with torch.no_grad():
        h0 = torch.zeros(nhidden, 1, dtype=torch.float64)
        _, o = model(h0, X_train)
        train_loss = F.cross_entropy(o, y_train)
        y_prob = F.softmax(o, dim=1)
        # my_train_loss = cross_entropy(y_prob, y_train)
        y_pred = np.argmax(y_prob, axis=1)
        metric_train = accuracy_score(y_pred, y_train)
        # print(f"Epoch {epoch:3d} loss {train_loss:7.4f} accuracy {metric_train:4.3f}")

        h0 = torch.zeros(nhidden, 1, dtype=torch.float64)
        _, o = model(h0, X_valid)
        valid_loss = F.cross_entropy(o, y_valid)
        y_prob = F.softmax(o, dim=1)
        y_pred = torch.argmax(y_prob, dim=1)
        metric_valid = accuracy_score(y_pred, y_valid)
        print(f"Epoch: {epoch:3d} seq loss {seq_loss:8.4f}     training loss {train_loss:7.4f} accuracy {metric_train:4.3f}     validation loss {valid_loss:7.4f} accuracy {metric_valid:4.3f}")
