

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
import torchviz

np.set_printoptions(precision=2, suppress=True, linewidth=3000, threshold=20000)
from typing import Sequence

dtype = torch.float
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

np.random.seed(1) # TESTING!


# In[2]:


def normal_transform(x, mean=0.0, std=0.01):
    "Convert x to have mean and std"
    return x*std + mean

def randn(n1, n2,          
          mean=0.0, std=0.01, requires_grad=False,
          device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
          dtype=torch.float64):
    x = torch.randn(n1, n2, device=device, dtype=dtype)
    x = normal_transform(x, mean=mean, std=std)
    x.requires_grad=requires_grad
    return x


# In[3]:


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


# In[4]:


def softmax(y):
    expy = torch.exp(y)
    if len(y.shape)==1: # 1D case can't use axis arg
        return expy / torch.sum(expy)
    return expy / torch.sum(expy, dim=1)#.reshape(-1,1)

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
    return torch.mean(-torch.log(p))


def onehot(c) -> torch.tensor:
    v = torch.zeros((len(vocab),1), dtype=torch.float64)
    v[ctoi[c]] = 1
    return v.to(device)


def forward1(x):
    h = torch.zeros(nhidden, 1, dtype=torch.float64, device=device, requires_grad=False)  # reset hidden state at start of record
    for j in range(len(x)):  # for each char in a name
        x_onehot = onehot(x[j])
        h = W.mm(h) + U.mm(x_onehot)# + b
#             h = torch.tanh(h)  # squish to (-1,+1)
        h = torch.relu(h)
#             print("h",h)
    # h is output of RNN, a fancy CBOW embedding for variable-length sequence in x
    # run through a final layer to map that h to a one-hot encoded predicted class
#         h = dropout(h, p=0.4)
    o = V.mm(h)# + Vb
    o = o.reshape(1,nclasses)
#     print(torch.sum(o[0]).item())
    o = softmax(o)
    return o


def forward(X:Sequence[Sequence]):#, apply_softmax=True):
    "Cut-n-paste from body of training for use with metrics"
#     outputs = torch.empty(len(X), nclasses, dtype=torch.float64).to(device)
    outputs = []
    for i in range(0, len(X)): # for each input record
        o = forward1(X[i])
        outputs.append( o.reshape(-1) )
    outputs = torch.stack(outputs)
    # outputs.reshape(len(X), nclasses)
    return outputs


def dropout(a:torch.tensor,   # activation/output of a layer
            p=0.0             # probability an activation is zeroed
           ) -> torch.tensor:
    usample = torch.empty_like(a).uniform_(0, 1) # get random value for each activation
    mask = (usample>p).int()                     # get mask as those with value greater than p
    a = a * mask                                 # kill masked activations
    a /= 1-p                                     # scale during training by 1/(1-p) to avoid scaling by p at test time
                                                 # after dropping p activations, (1-p) are left untouched, on average
    return a


# ## Load
# 
# Let's download [training](https://raw.githubusercontent.com/hunkim/PyTorchZeroToAll/master/data/names_train.csv.gz) and [testing](https://raw.githubusercontent.com/hunkim/PyTorchZeroToAll/master/data/names_test.csv.gz) data for last names.   This data set is a bunch of last names and the nationality or language. 

# In[5]:


df_train = pd.read_csv("notebooks/data/names_train.csv", header=None)
df_train.columns = ['name','language']
df_test = pd.read_csv("notebooks/data/names_test.csv", header=None)
df_test.columns = ['name','language']


# In[6]:


df_train.shape, df_test.shape


# In[7]:


df_train.head(2)


# In[8]:


# TESTING SUBSAMPLE
df_train = df_train.sample(n=2000)
df_test = df_test.sample(n=2000)


# ## Clean

# In[9]:


badname = df_train['name']=='To The First Page' # wth?
df_train[badname].head(2)


# In[10]:


# probably destroying useful info but much smaller vocab
df_train['name'] = df_train['name'].str.lower()
df_test['name'] = df_test['name'].str.lower()


# ## Get vocab

# In[11]:


def vocab(strings):
    letters = [list(l) for l in strings]
    vocab = set([c for cl in letters for c in cl])
    vocab = sorted(list(vocab))
    ctoi = {c:i for i, c in enumerate(vocab)}
    return vocab, ctoi


# In[12]:


vocab, ctoi = vocab(df_train['name'])


# ## Split names into variable-length lists

# In[13]:


X_train = [list(name) for name in df_train['name']]


# In[14]:


df_train = df_train[df_train['name']!='To The First Page']
badname = df_test['name']=='To The First Page'
df_test = df_test[df_test['name']!='To The First Page']


# ## Split names into variable-length lists

# In[15]:


X, y = df_train['name'], df_train['language']
X = [list(name) for name in X]
X[0:2]


# In[16]:


X_test, y_test = df_test['name'], df_test['language']
X_test = [list(name) for name in X_test]
X_test[0:2]


# ## Split out validation set

# In[17]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20)


y_train = y_train.astype('category').cat.as_ordered()
y_cats = y_train.cat.categories

y_train = torch.tensor(np.array(y_train.cat.codes.values), dtype=torch.long)


y_valid = torch.tensor(np.array(pd.Categorical(y_valid, categories=y_cats, ordered=True).codes),
                       dtype=torch.long)
y_test = pd.Categorical(y_test, categories=y_cats, ordered=True).codes




# ## Model
# 
# Just some matrices. First, set up hyper parameters:

# In[63]:


nhidden = 100
nfeatures = len(vocab)
nclasses = len(y_cats)
n = len(X_train)
print(f"{n:,d} training records, {nfeatures} features (chars), {nclasses} target languages, state is {nhidden}-vector")

torch.manual_seed(0) # SET SEED FOR TESTING
W = torch.eye(nhidden, nhidden,   dtype=torch.float64, device=device, requires_grad=True)
U = randn(nhidden,     nfeatures, dtype=torch.float64, device=device, requires_grad=True) # embed one-hot char vec
V = randn(nclasses,    nhidden,   dtype=torch.float64, device=device, requires_grad=True)  # take RNN output (h) and predict target
#b = randn(nhidden,    1,         dtype=torch.float64, device=device, requires_grad=True)  # bias
#Vb = randn(nclasses,  1,         dtype=torch.float64, device=device, requires_grad=True)  # bias for final, classifier layer

learning_rate = 0.001
weight_decay = 0.0#0001

optimizer = torch.optim.Adam([W,U,V], lr=learning_rate, weight_decay=weight_decay)

history = []
epochs = 15
for epoch in range(1, epochs + 1):
#     print(f"EPOCH {epoch}")
    epoch_training_loss = 0.0
    epoch_training_accur = 0.0
    for i in range(0, n): # an epoch trains all input records
        x = X_train[i]
        # o = forward1(x)
#         print(i,x)
        h = torch.zeros(nhidden, 1, dtype=torch.float64, device=device, requires_grad=False)  # reset hidden state at start of record
        for j in range(len(x)):  # for each char in a name
            x_onehot = onehot(x[j])
            h = W.mm(h) + U.mm(x_onehot)# + b
            h = torch.relu(h)
#             print("h",h)
        # h is output of RNN, a fancy CBOW embedding for variable-length sequence in x
        # run through a final layer to map that h to a one-hot encoded predicted class
#         h = dropout(h, p=0.4)
        o = V.mm(h)# + Vb
        o = o.reshape(1,nclasses)
        o = softmax(o)
#             print("softmax",o,"y_train[i]",y_train[i])

        loss = cross_entropy(o, y_train[i])
        optimizer.zero_grad()
        loss.backward() # autograd computes U.grad, M.grad, ...
        optimizer.step()

        with torch.no_grad():
            epoch_training_loss += loss.item()
            correct = torch.argmax(o[0]).item()==y_train[i]
            epoch_training_accur += correct
    #         print("\tword loss", torch.mean(torch.tensor(losses)).item())

    epoch_training_loss /= n
    epoch_training_accur /= n
#     print(f"Epoch {epoch:3d} training loss {epoch_training_loss:7.4f} accur {epoch_training_accur:7.4f}")

    with torch.no_grad():
        o = forward(X_train)#, apply_softmax=False)
        train_loss = cross_entropy(o, y_train)
        correct = torch.argmax(o, dim=1).detach().cpu()==y_train
        train_accur = torch.sum(correct) / float(len(X_train))

        o = forward(X_valid)
        valid_loss = cross_entropy(o, y_valid)
        correct = torch.argmax(o, dim=1).detach().cpu()==y_valid
        valid_accur = torch.sum(correct) / float(len(X_valid))

        # valid_loss = 0.0
        # valid_accur = 0.0
        # for i,x in enumerate(X_valid):
        #     o = forward([x])
        #     o = o.reshape(1,-1)
        #     o2 = forward1(x)
        #     o2 = o2.reshape(1,-1)
        #     # print("same?", torch.allclose(o,o2))
        #     print("o =",o[0,:5])
        #     print("o2=",o2[0,:5])
        #     loss = 0#F.cross_entropy(o, torch.tensor([y_valid[i]],dtype=torch.long)).item()
        #     valid_loss += loss
        #     y_pred = torch.argmax(o[0]).item()  # don't need softmax for prediction really
        #     correct = y_pred==y_valid[i]
        #     valid_accur += correct
        # valid_loss /= len(X_valid)
        # valid_accur /= len(X_valid)

        history.append((train_loss, valid_loss))
        print(f"Epoch: {epoch:3d} accum loss {epoch_training_loss:7.4f} accur {epoch_training_accur:4.3f} | train loss {train_loss:7.4f} accur {train_accur:4.3f} | valid loss {valid_loss:7.4f} accur {valid_accur:4.3f}")

history = torch.tensor(history)
plot_history(history, yrange=(0,7))



