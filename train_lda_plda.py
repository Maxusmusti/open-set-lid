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
from torchsummary import summary
from sklearn.decomposition import PCA
import sys

random.seed(1)

if len(sys.argv) != 2:
    print("Usage: python3 train_lda_plda.py <saved_tdnn_outputs_dir> \ne.g. python3 train_lda_plda.py ./saved-tdnn-outputs")
    exit()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)



print("\n===================================TRAINING & SAVING LDA/PLDA==================================\n")

tdnn_labels_load_dir = sys.argv[1] + '/tdnn_y_final.txt'
tdnn_outputs_load_dir = sys.argv[1] + '/tdnn_X_final.txt'

lda_save_dir = './saved-lda/'
plda_save_dir = './saved-plda/'

y = np.loadtxt(tdnn_labels_load_dir)
load_size = 4000
lda_dim = 18

i = 0
while i < y.shape[0]:
    print("\titer:", i, "of", len(y))

    print("\t\tLoading TDNN outputs...")
    ychunk = np.loadtxt(tdnn_labels_load_dir, skiprows=i, max_rows=load_size)
    Xchunk = np.loadtxt(tdnn_outputs_load_dir, skiprows=i, max_rows=load_size)
    i += load_size
    print(np.unique(ychunk).shape) 
    print("\t\tFitting + Transforming LDA...")
    dim_red = lda.LinearDiscriminantAnalysis(n_components=lda_dim)
    dim_red_X = dim_red.fit_transform(Xchunk, ychunk)
    dim_red_y = np.array(ychunk)
        
    print("\t\tSaving LDA...")
    pickle.dump(dim_red, open(lda_save_dir + 'lda_temp_' + f'{(int(i/load_size)):08}' + '.pk', 'wb'))

    print("\t\tStarting PLDA fitting...")
    plda_classifier = plda.Classifier()
    plda_classifier.fit_model(dim_red_X, dim_red_y)
    
    print("\t\tSaving PLDA layer...")
    pickle.dump(plda_classifier, open(plda_save_dir + 'plda_temp_' + f'{(int(i/load_size)):08}' + '.pk', 'wb'))        
    


    
