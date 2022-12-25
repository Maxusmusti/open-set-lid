import kaldiio
import sys, os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from pickle import Unpickler
from tdnn import TDNN
from collections import defaultdict
import sys

random.seed(1)

if len(sys.argv) != 2:
    print("Usage: python3 test_tdnn.py <saved-model-path>")
    print("e.g.  python3 test_tdnn.py ./saved-models/tdnn-final-submission.pickle")
    exit()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

in_set = ['ENG', 'GER', 'ICE', 'FRE', 'SPA', 'ARA', 'RUS', 'BEN', 'KAS', 'GRE', 'CAT', 'KOR', 'TUR', 'TAM', 'TEL', 'CHI', 'TIB', 'JAV', 'EWE', 'HAU', 'LIN', 'YOR', 'HUN', 'HAW', 'MAO', 'ITA', 'URD', 'SWE', 'PUS', 'GEO', 'HIN', 'THA']
out_of_set = ['DUT', 'HEB', 'UKR', 'BUL', 'PER', 'ALB', 'UIG', 'MAL', 'BUR', 'IBA', 'ASA', 'AKU', 'ARM', 'HRV', 'FIN', 'JPN', 'NOR', 'NEP', 'RUM']

data = []
for i,lang in enumerate(in_set + out_of_set, 0):
    print(lang, "(In-set)" if lang in in_set else "(Out-of-set)")
    filepath = './feature-subset/' + lang + '/raw_mfcc_pitch_' + lang + '.1.ark'

    for key, numpy_array in kaldiio.load_ark(filepath):
        inputs = torch.from_numpy(np.expand_dims(numpy_array, axis=0))
        labels = torch.from_numpy(np.array([i if lang in in_set else -1]))
        data.append((inputs, labels))

print("\nGenerating test set")
random.shuffle(data)
data_concatenated = dict()
for iter,i in enumerate(data): 
    label = i[1].numpy()[0]
    mfcc = np.squeeze(i[0].numpy(), axis=0)
    if label in data_concatenated:
        data_concatenated[label].append(mfcc)
    else:
        data_concatenated[label] = [mfcc]

for i in data_concatenated:
    data_concatenated[i] = np.vstack(data_concatenated[i])

del data

def chunkify_tensor(tensor, size=400):
    return torch.split(tensor, size, dim=1)[:-1] # except last one bc that isn't the right size

'''
Train (tdnn):       95% of in-set
Train (lda/plda):   80% of out-of-set
Test:               5% of in-set + 20% out-of-set
'''

train1, train2, test = [], [], []
for i in data_concatenated:
    label = torch.from_numpy(np.array([i]))
    mfcc = torch.from_numpy(np.expand_dims(data_concatenated[i], axis=0))
    chunks = chunkify_tensor(mfcc)
    if i >= 0:
        cutoff = int(len(chunks) * 0.95)
        for chunk in chunks[:cutoff]:
            train1.append((chunk.to(device), label.to(device)))
        for chunk in chunks[cutoff:]:
            test.append((chunk.to(device), label.to(device)))
    else:
        cutoff = int(len(chunks) * 0.8)
        for chunk in chunks[:cutoff]:
            train2.append((chunk.to(device), label.to(device)))
        for chunk in chunks[cutoff:]:
            test.append((chunk.to(device), label.to(device)))

del data_concatenated

for i in test:
    assert(test[0][0].size(dim=0) == i[0].size(dim=0))
    assert(test[0][0].size(dim=1) == i[0].size(dim=1))
    assert(test[0][0].size(dim=2) == i[0].size(dim=2))


class Net(nn.Module):
    def __init__(self, in_size, num_classes):
        super().__init__()
        
        self.layer1 = TDNN(input_dim=in_size, output_dim=256, context_size=3)
        self.layer2 = TDNN(input_dim=256, output_dim=256, context_size=3, dilation=1)
        self.layer3 = TDNN(input_dim=256, output_dim=256, context_size=3, dilation=1)
        self.layer4 = TDNN(input_dim=256, output_dim=256, context_size=1)
        self.layer5 = TDNN(input_dim=256, output_dim=256, context_size=1)
        self.final_layer = TDNN(input_dim=256, output_dim=num_classes, context_size=1)
        
    def forward(self, x):
        forward_pass = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.ReLU(),
            self.layer3,
            nn.ReLU(),
            self.layer4,
            nn.ReLU(),
            self.layer5,
            nn.ReLU(),
            self.final_layer)
        
        return forward_pass(x)


LOAD_PATH = sys.argv[1]
print('Loading the model')

infile = open(LOAD_PATH, "rb")
net = Unpickler(infile).load()
infile.close()

print('Finished loading')
print()


#only test on the in-set ones
test = [i for i in test if i[1].to('cpu').numpy()[0] >= 0]
random.shuffle(test)


print("Testing TDNN on the in-set samples of test set...")
correct, total = 0, 0

for i, data in enumerate(test,0):
    if i%2000 == 0 and total > 0:
        print("iter:", i, 'of', len(test))

    inputs, labels = data[0].to(device), data[1].to(device)
    
    outputs = net(inputs)
    outputs = torch.mean(outputs, 1)
    outputs = F.softmax(outputs, dim=1)
    predicted = outputs.argmax(1, keepdim=True)

    total += 1
    if predicted == labels:
        correct += 1

print("\nOverall acc on in-set test data:", correct/total)

