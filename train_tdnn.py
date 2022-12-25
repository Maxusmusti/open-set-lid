import sys, os
import random
import numpy as np
import pandas as pd
import kaldiio
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from tdnn import TDNN
import matplotlib.pyplot as plt
from pickle import Pickler, Unpickler


random.seed(1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

in_set = ['ENG', 'GER', 'ICE', 'FRE', 'SPA', 'ARA', 'RUS', 'BEN', 'KAS', 'GRE', 'CAT', 'KOR', 'TUR', 'TAM', 'TEL', 'CHI', 'TIB', 'JAV', 'EWE', 'HAU', 'LIN', 'YOR', 'HUN', 'HAW', 'MAO', 'ITA', 'URD', 'SWE', 'PUS', 'GEO', 'HIN', 'THA']
out_of_set = ['DUT', 'HEB', 'UKR', 'BUL', 'PER', 'ALB', 'UIG', 'MAL', 'BUR', 'IBA', 'ASA', 'AKU', 'ARM', 'HRV', 'FIN', 'JPN', 'NOR', 'NEP', 'RUM']

langs = in_set + out_of_set

num_in_set = 32

in_set = langs[:num_in_set]
out_of_set = langs[num_in_set:]

root_dir = "./feature-subset/"

isolated_speakers = {
    "ENG": [27, 28, 29, 30],
    "SPA": [30]
}


assert(len(in_set) + len(out_of_set) == 51)
assert(len(set(in_set).intersection(set(out_of_set))) == 0)
for lang in os.listdir(root_dir):
    assert(lang in in_set or lang in out_of_set)


clip_size = 3
hours_per_lang = 4
num_epochs = 15


class LanguageDataset(Dataset):
    def __init__(self, chunks):
        self.chunks=chunks
        
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, index):
        el = self.chunks[index]
        sample = el[0].squeeze()
        label = el[1][0]
        
        return sample, label


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
    

def create_train_test_data(switch_point=0.8, batch_size=512):
    print("\n-----Getting Train/Test Data-----\n")

    train, test, oos_test = [], [], []

    max_sample_length = clip_size * 100
    num_chunks_per_file = (hours_per_lang * 3600) // (clip_size * 30)
    switch_point = 0.8

    for i,lang in enumerate(in_set + out_of_set, 0):
        print(lang, "(In-set)" if lang in in_set else "(Out-of-set)")

        chunks = []
        for f_idx in range(1, 2):
            if lang in isolated_speakers and f_idx in isolated_speakers[lang]:
                continue

            filepath = root_dir + lang + '/raw_mfcc_pitch_' + lang + '.' + str(f_idx) + '.ark'
            file_chunks = []

            for key, numpy_array in kaldiio.load_ark(filepath):
                curr_len = len(numpy_array)

                if curr_len >= max_sample_length:
                    file_chunks +=  np.split(numpy_array, np.arange(max_sample_length, curr_len, max_sample_length))[:-1]
                else:
                    padded_chunk = np.pad(numpy_array, ((max_sample_length - curr_len, 0), (0, 0)), "constant")
                    file_chunks += [padded_chunk]

                if len(file_chunks) >= num_chunks_per_file:
                    chunks += file_chunks
                    break

        random.shuffle(chunks)
        chunks= np.array(chunks)

        for j, chunk in enumerate(chunks):
            inputs = torch.from_numpy(chunk)
            inputs.to(device)
            labels = torch.from_numpy(np.array([i if lang in in_set else -1]))
            labels.to(device)

            if j + 1 <= switch_point * len(chunks):
                if lang in in_set:
                    train.append((inputs,labels))
            else:
                if lang in in_set:
                    test.append((inputs, labels))
                else:
                    oos_test.append((inputs, labels))

    print()
    print("\n-----Finished Data Splitting-----\n")
    print("Creating Train and Test loaders...")
    
    batch_size = 512

    train_set = LanguageDataset(train)
    val_set = LanguageDataset(test)
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_set, shuffle=True, batch_size=batch_size)
    
    
    return train_loader, val_loader, test + oos_test


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train_model(model, train_loader, optimizer, criterion, device=None):
    train_loss = 0.0
    train_acc = 0.0

    model.train()

    for x, y in train_loader:
      if device is not None:
        x = x.to(device)
        y = y.to(device)

      optimizer.zero_grad()

      y_pred = model(x)
      y_pred = torch.mean(y_pred, 1)
      loss = criterion(y_pred, y)
      acc = calculate_accuracy(y_pred, y)

      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      train_acc += acc.item()

    return train_loss / len(train_loader), train_acc / len(train_loader)


def evaluate_model(model, val_loader, criterion, device=None):
    val_loss = 0.0
    val_acc = 0.0

    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
          if device is not None:
            x = x.to(device)
            y = y.to(device)

          y_pred = model(x)
          y_pred = torch.mean(y_pred, 1)
          loss = criterion(y_pred, y)
          acc = calculate_accuracy(y_pred, y)

          val_loss += loss.item()
          val_acc += acc.item()

    return val_loss / len(val_loader), val_acc / len(val_loader)


def train_network(net, train_loader, val_loader):
    # Initialize the TDNN, loss, and optimizer
    net.to(device)
    criterion = nn.CrossEntropyLoss() # a common loss function for multi-class classification problems
    optimizer = optim.AdamW(net.parameters(), lr=0.001) # a common optimizer for multi-class classification problems

    SAVE_PATH = "./saved-models/exp-tdnn-256-" + str(hours_per_lang) + "h-" + str(clip_size) + "s-" + str(num_epochs) + "epochs"

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_loss = float('inf')
    best_epoch = 0

    print('Started Training')


    for epoch in range(num_epochs):  # number of epochs
        train_loss, train_accuracy = train_model(net, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate_model(net, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_loss < best_loss:
          best_loss = val_loss
          best_epoch = epoch

          with open(SAVE_PATH + ".pickle", "wb") as outfile:
            pickle.dump(net, outfile, protocol=4)


        print("Epoch: " + str(epoch) + ", Train Loss: " + str(train_loss) + ", Train Accuracy: " + str(train_accuracy) +             ", Val Loss: " + str(val_loss) + ", Val Accuracy: " + str(val_accuracy))


    torch.save(net.state_dict(), SAVE_PATH + ".pth") # Save the model

    print("Best Val Loss: " + str(best_loss) + " at Epoch: " + str(best_epoch))

    print('Finished Training')

    return train_losses, train_accuracies, val_losses, val_accuracies


def save_plots(train_losses, train_accuracies, val_losses, val_accuracies):
    model_name = "TDNN trained on " + str(hours_per_lang) + " hours per language in " + str(clip_size) + "s clips for " + str(num_epochs) + " epochs"
    
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Valid")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(model_name)

    res_path = os.path.join("./", "results", str(hours_per_lang) + "h", str(clip_size) + "s", str(num_epochs) + "epochs")
    os.makedirs(res_path, exist_ok=True)
    plt.savefig(os.path.join(res_path, "loss_plot"))
    
    plt.clf()

    plt.plot(train_accuracies, label="Train")
    plt.plot(val_accuracies, label="Valid")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (Super)")
    plt.legend()
    plt.title(model_name)

    plt.savefig(os.path.join(res_path, "accuracy_plot"))
    
    plt.close()


def test_with_oos(model, test_set):
    res = {}

    for thresh in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        correct = 0
        for x, y in test_set:
          # Send to GPU if available
          if device is not None:
            x = x.to(device)
            y = y.to(device)

          x = x.unsqueeze(0)
          y_pred = model(x)
          y_pred = torch.mean(y_pred, 1)
          y_pred = F.softmax(y_pred, dim=1)

          conf = y_pred.amax(1, keepdim=True).item()
          if conf > thresh:
              top_pred = y_pred.argmax(1, keepdim=True)
          else:
              top_pred = -1

          if top_pred == y:
            correct += 1

        acc = correct / len(test_set)
        print(thresh, acc)

        res[str(thresh)] = str(acc)

    return res    


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: training_script.py <num hours per language> <size of each audio clip in s> <num epochs> <train_split> \n Eg: python3 train_tdnn.py 5 4 15 0.8")
        exit(1)

    hours_per_lang = int(sys.argv[1])
    clip_size = int(sys.argv[2])
    num_epochs = int(sys.argv[3])
    train_split = float(sys.argv[4])

    train_loader, val_loader, test_set = create_train_test_data(switch_point=0.8, batch_size=512)
    
    net = Net(16, len(in_set)).to(device)
    train_losses, train_accuracies, val_losses, val_accuracies = train_network(net, train_loader, val_loader)
    #save_plots(train_losses, train_accuracies, val_losses, val_accuracies)
    #results_with_oos = test_with_oos(net, test_set)
