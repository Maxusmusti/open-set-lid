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
import pickle
from pickle import Unpickler
from tdnn import TDNN
from collections import defaultdict
import sklearn.discriminant_analysis as lda
import plda
from collections import defaultdict
import glob
from torchsummary import summary
from sklearn.decomposition import PCA

random.seed(1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)



in_set = ['ENG', 'GER', 'ICE', 'FRE', 'SPA', 'ARA', 'RUS', 'BEN', 'KAS', 'GRE', 'CAT', 'KOR', 'TUR', 'TAM', 'TEL', 'CHI', 'TIB', 'JAV', 'EWE', 'HAU', 'LIN', 'YOR', 'HUN', 'HAW', 'MAO', 'ITA', 'URD', 'SWE', 'PUS', 'GEO', 'HIN', 'THA']
out_of_set = ['DUT', 'HEB', 'UKR', 'BUL', 'PER', 'ALB', 'UIG', 'MAL', 'BUR', 'IBA', 'ASA', 'AKU', 'ARM', 'HRV', 'FIN', 'JPN', 'NOR', 'NEP', 'RUM']

print("\n===================================LOADING MFCC+PITCH FROM ARK==================================\n")

data = []
for i,lang in enumerate(in_set + out_of_set, 0):
    print(lang, "(In-set)" if lang in in_set else "(Out-of-set)")
    filepath = './feature-subset/' + lang + '/raw_mfcc_pitch_' + lang + '.1.ark'

    for key, numpy_array in kaldiio.load_ark(filepath):
        inputs = torch.from_numpy(np.expand_dims(numpy_array, axis=0))
        labels = torch.from_numpy(np.array([i if lang in in_set else (-i+len(in_set)-1) ]))
        data.append((inputs, labels))

print("\n===================================SPLITTING DATA INTO 3 SETS: train1, train2, test==================================\n")

'''
Train1 (tdnn):          95% of in-set
Train2 (lda/plda):      80% of out-of-set
Test:                   5% of in-set + 20% out-of-set
'''

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

random.shuffle(train1)
random.shuffle(train2)
random.shuffle(test)

for i in test:
    assert(test[0][0].size(dim=0) == i[0].size(dim=0))
    assert(test[0][0].size(dim=1) == i[0].size(dim=1))
    assert(test[0][0].size(dim=2) == i[0].size(dim=2))

random.shuffle(test)
test = test[:3000] # subset for demo purposes

print("\n===================================PREPARING TDNN MODELS==================================\n")

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

LOAD_PATH = './saved-models/tdnn-final-submission'

print('Loading model: first copy for softmax + threshold')
infile = open(LOAD_PATH + ".pickle", "rb")
net = Unpickler(infile).load()
infile.close()

print('Loading model: second copy for LDA + PLDA')
infile2 = open(LOAD_PATH + ".pickle", "rb")
net2 = Unpickler(infile2).load()
net2.final_layer = nn.Identity()
infile2.close()

print("\n===================================TEST TEST TEST==================================\n")
'''
Measure 4 things:
- IS detection (threshold): out of all IS samples, how many were labeled as IS
- OOS detection (threshold): out of all OOS samples, how many were labeled as OOS
- IS ID (softmax): out of all IS samples, how many were labeled as IS and given the correct language ID
- OOS ID (plda): out of correclty detected OOS, How many are correctly labeled by plda
'''

lda_load_dir = './saved-lda/'
plda_load_dir = './saved-plda/'

def print_accuracies(y_correct, y_pred):
    assert(len(y_correct) == len(y_pred))
    accuracies = {
        'IS_DET': {'correct': 0, 'total': 0},
        'OOS_DET': {'correct': 0, 'total': 0},
        'IS_ID': {'correct': 0, 'total': 0},
        'OOS_ID': {'correct': 0, 'total': 0}
    }

    for i in range(len(y_correct)):

        if y_correct[i] >= 0:

            accuracies['IS_DET']['total'] += 1

            if y_pred[i] >= 0:
                accuracies['IS_DET']['correct'] += 1
                accuracies['IS_ID']['total'] += 1

                if y_pred[i] == y_correct[i]:
                    accuracies['IS_ID']['correct'] += 1

        else:

            accuracies['OOS_DET']['total'] += 1

            if y_pred[i] < 0:
                accuracies['OOS_DET']['correct'] += 1
                accuracies['OOS_ID']['total'] += 1

                if y_pred[i] == y_correct[i]:
                    accuracies['OOS_ID']['correct'] += 1

    for key in sorted(accuracies.keys()):
        try:
            print('\t', key, ':', accuracies[key]['correct']/accuracies[key]['total'])
        except:
            pass
    return accuracies

def get_mode(lst):
    counter = defaultdict(int)
    for i in lst:
        counter[i] += 1
    maxes = [k for k, v in counter.items() if v == max(counter.values())]    
    return random.choice(maxes)

def get_mode2(lst):
    astranspose = np.transpose(np.array(lst)).tolist()
    return [get_mode(i) for i in astranspose]

def get_lda_plda_layers(lda_load_directory, plda_load_directory):

    lda_layers = []
    for lda_layer in sorted(glob.glob(lda_load_directory + '*.pk')):
        lda_layers.append(pickle.load(open(lda_layer, 'rb')))
    print(len(lda_layers), "lda layers loaded")
    
    plda_layers = []
    for plda_layer in sorted(glob.glob(plda_load_directory + '*.pk')):
        plda_layers.append(pickle.load(open(plda_layer, 'rb')))
    print(len(plda_layers), "plda layers loaded")

    print()
    return lda_layers, plda_layers

do_testing = True
if do_testing:
    lda_layers, plda_layers = get_lda_plda_layers(lda_load_dir, plda_load_dir)

    net = net.to(device)
    net2 = net2.to(device)

    random.shuffle(test)

    all_paths = []
    with torch.no_grad():

        save_predicted1, save_conf, save_flattened, save_predicted2, save_labels = [], [], [], [], []

        for i, data in enumerate(test,0):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(inputs)
            outputs = torch.mean(outputs, 1)
            outputs = F.softmax(outputs, dim=1)
            predicted1 = outputs.argmax(1, keepdim=True)[0]
            conf = outputs.amax(1, keepdim=True).item()
            
            output_net2 = net2(inputs)
            flattened = np.squeeze(torch.flatten(output_net2)).detach().cpu().numpy()

            save_predicted1.append(predicted1)
            save_conf.append(conf)
            save_flattened.append(flattened)
            save_labels.append(labels.to('cpu').numpy()[0])

            if i % 1000 == 0 and i != 0:
                print("iter", i, 'of', len(test))
                ensemble_predictions = []
                for layer_i in range(len(lda_layers)):
                    lda_output = lda_layers[layer_i].transform(save_flattened)
                    plda_output, _ = plda_layers[layer_i].predict(lda_output)
                    ensemble_predictions.append(plda_output)
                predicted2 = get_mode2(ensemble_predictions)
                save_predicted2 = predicted2

                for i in range(len(save_predicted1)):
                    all_paths.append((save_predicted1[i], save_conf[i], save_predicted2[i], save_labels[i]))
                save_predicted1, save_conf, save_flattened, save_predicted2, save_labels = [], [], [], [], []
        
        # dupe code as above
        ensemble_predictions = []
        for layer_i in range(len(lda_layers)):
            lda_output = lda_layers[layer_i].transform(save_flattened)
            plda_output, _ = plda_layers[layer_i].predict(lda_output)
            ensemble_predictions.append(plda_output)
        predicted2 = get_mode2(ensemble_predictions)
        save_predicted2 = predicted2

        for i in range(len(save_predicted1)):
            all_paths.append((save_predicted1[i], save_conf[i], save_predicted2[i], save_labels[i]))
        save_predicted1, save_conf, save_flattened, save_predicted2, save_labels = [], [], [], [], []
    print()

    assert(len(all_paths) == len(test))
    accuracies_plot = []
    y_correct = [i[3] for i in all_paths]
    for threshold in [i/100 for i in range(0,105,5)]: # 0, 0.05, 0.1, 0.15, ... , 0.95, 1.0

        y_pred = []
        for i in all_paths:
            predicted = i[0]
            if i[1] < threshold:
                predicted = i[2]
            y_pred.append(predicted)
            
        print(f'Threshold: {threshold}')
        accuracies = print_accuracies(y_correct, y_pred)
        accuracies_plot.append((threshold, accuracies))
        print()

