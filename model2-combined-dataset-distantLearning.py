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
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 output_dim, n_layers, bidirectional, dropout, padding_idx, n_filters, filter_sizes):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=bidirectional, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                                              kernel_size = (fs, hidden_dim * 2)) 
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        
    def forward(self, text):
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
                
        # #hidden = [batch size, hid dim * num directions]

        # #text = [sent len, batch size]
        
        # text = text.permute(1, 0)
                
        # #text = [batch size, sent len]
        
        # embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        # embedded = embedded.unsqueeze(1)
                
        output = self.dropout(output)
        
        output = output.permute(1, 0, 2)
        
        output = output.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(output)).squeeze(3) for conv in self.convs]
            
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)
            

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum()/torch.FloatTensor([y.shape[0]])


def f1_scores(preds, y):
    f1_macro = f1_score(preds, y, average='macro')

    f1_weighted = f1_score(preds, y, average='weighted')

    return f1_macro, f1_weighted


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = categorical_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    predictions_arr = []
    labels_arr = []

    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)

            # f1_score code
            predictions_arr += predictions.argmax(dim=1).tolist()
            labels_arr += batch.label.tolist()
            # f1_score end
            
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    f1_macro, f1_weighted = f1_scores(predictions_arr, labels_arr) 
    return epoch_loss / len(iterator), epoch_acc / len(iterator), f1_macro, f1_weighted



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def filter(token):
    if token[0] == '@':
        return '<at_@>'
    if token[:4] == 'http':
        return '<http>'
    return token.lower()


def predict_sentiment(sentence):
    nlp = spacy.load('en')
    tokenized = [filter(tok.text) for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim=1) 
    sentiment = LABEL.vocab.itos[max_preds.item()]

    probs = nn.functional.softmax(preds, dim=1)[0].tolist()
    return sentence, sentiment, probs


SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field()
LABEL = data.LabelField()

fields = {'post': ('text', TEXT), 'sentiment': ('label', LABEL)}

train_path = 'combined_train_set.json'
test_non_target_path = 'non_target_test.json'
test_target_path = 'target_test.json'

train_data, test_non_target_data = data.TabularDataset.splits(
                            path = 'my_data_v4',
                            train = train_path,
                            test= test_non_target_path,
                            format = 'json',
                            fields = fields
)
_, test_target_data = data.TabularDataset.splits(
                            path = 'my_data_v4',
                            train = train_path,
                            test = test_target_path,
                            format = 'json',
                            fields = fields
)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

custom_embeddings = vocab.Vectors(name = '.vector_cache/1-6M-my-train-embedding-200d-fixed.txt',
                                  cache = '.vector_cache/',
                                  unk_init = torch.Tensor.normal_)
TEXT.build_vocab(train_data, max_size=25000, vectors=custom_embeddings)
LABEL.build_vocab(train_data)

BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_non_target_iterator, test_target_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_non_target_data, test_target_data), sort_key=lambda x: len(x.text),
    batch_size=BATCH_SIZE,
    device=device)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 200
HIDDEN_DIM = 200 # !!! change -> 256
OUTPUT_DIM = 3 # !!! change -> 3
N_LAYERS = 1 # !!! change -> 2
BIDIRECTIONAL = True
DROPOUT = 0.5
N_FILTERS = 150
FILTER_SIZES = [3,4,5]

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, TEXT.vocab.stoi['<pad>'],
        N_FILTERS, FILTER_SIZES)

pretrained_embeddings = TEXT.vocab.vectors
print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[TEXT.vocab.stoi['<unk>']] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[TEXT.vocab.stoi['<pad>']] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.requires_grad = False

print('The model has %d trainable parameters' % count_parameters(model))

optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad == True], weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 20
best_valid_loss = float('inf')

model_save_dir = 'model-rnn-cnn.pt'

print('start')
print(vars(train_data.examples[0]))

print("Unique tokens in TEXT vocabulary: %d" % len(TEXT.vocab))
print("Unique tokens in LABEL vocabulary: %d" % len(LABEL.vocab))

print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.freqs.most_common(20))
print(LABEL.vocab.stoi)

print("train_data %d, valid_data %d, test_non_target_data %d, test_target_data %d" % 
    (len(train_data), len(valid_data), len(test_non_target_data), len(test_target_data)))


for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc, f1_macro, f1_weighted = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_save_dir)

    if epoch == N_EPOCHS / 2:
        print('Unfreeze embeddings')
        model.embedding.weight.requires_grad = True
        optimizer = optim.Adam(model.parameters())

    
    print('Epoch: %d | Epoch Time: %dm %ds' % (epoch+1, epoch_mins, epoch_secs))
    print('\tTrain Loss: %.3f | Train Acc: %.2f%%' % (train_loss, train_acc*100))
    print('\t Val. Loss: %.3f |  Val. Acc: %.2f%% | F1_macro: %.3f | F1_weighted: %.3f' % (valid_loss, valid_acc*100, f1_macro, f1_weighted))

import os
if os.path.exists(model_save_dir):
    model.load_state_dict(torch.load(model_save_dir))

test_loss, test_acc, f1_macro, f1_weighted = evaluate(model, test_non_target_iterator, criterion)
print('NON_TARGET size(%d): Test Loss: %.3f | Test Acc: %.2f%% | F1_macro: %.3f | F1_weighted: %.3f' % 
    (len(test_non_target_data), test_loss, test_acc*100, f1_macro, f1_weighted))

test_loss, test_acc, f1_macro, f1_weighted = evaluate(model, test_target_iterator, criterion)
print('TARGET size(%d): Test Loss: %.3f | Test Acc: %.2f%% | F1_macro: %.3f | F1_weighted: %.3f' % 
    (len(test_target_data), test_loss, test_acc*100, f1_macro, f1_weighted))

print(predict_sentiment("This film is terrible"))
print(predict_sentiment("This film is great"))
print(predict_sentiment("my_target_wrapper Obama my_target_wrapper is great, but Trump is awful"))
print(predict_sentiment("my_target_wrapper Obama my_target_wrapper is good, but Trump is bad"))

print(predict_sentiment("Obama is great, but my_target_wrapper Trump my_target_wrapper is awful"))
print(predict_sentiment("Obama is good, but my_target_wrapper Trump my_target_wrapper is bad"))
