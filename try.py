# rnn clean implementation for dataset 1.6M tweets, two label

import torch
from torchtext import data
from torchtext import datasets
import torchtext.vocab as vocab
import random
import torch.nn as nn
import torch.optim as optim
import time
import spacy
from sklearn.metrics import f1_score
import numpy as np 


rnn = nn.LSTM(100, 100, num_layers=3, 
                   bidirectional=True, dropout=0.5)


#embedded = [sent len, batch size, emb dim]
embedded = torch.randn((80, 16, 100))

output, (hidden, cell) = rnn(embedded)

print(output.shape)

print(hidden.shape)