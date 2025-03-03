#!/usr/bin/env python
import os
import string
import random
import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# From HW2
from torch.utils.data import TensorDataset, DataLoader
#import dataloader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd

from torch.nn.utils.rnn import pad_sequence

from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()

global_char_dict = ['a', 'b', 'c',  'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/']
# Actual model
class MyModel(nn.Module):

    def __init__(self):
        """
        Initialize the Feed-Foward model.
        Inputs:
        - sentence_dim: The dimension of the sentence embeddings
        - word_dim: The dimension of the final word embedding
        - hidden_dim: The dimension of the hidden layer
        - output_dim: The dimension of the output (e.g. number of ASCII characters)
        """

        super(MyModel, self).__init__()
        self.char_to_int = None
        self.n_vocab = -1
        self.best_model = None

        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = None #nn.Linear(256, 76)

    def forward(self, x) -> torch.Tensor:
        """
        Perform a forward pass through the network.
        Inputs:
        - context: torch.Tensor: The sentence embeddings of the context
        - curr_word: torch.Tensor: The embeddings of the characters of the current word
        Returns:
        - torch.Tensor: The logits for each of the answers
        """
        # print(self.n_vocab)
        x, _ = self.lstm(x)
        # take only the last output -- this allows sequences of variable length?
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

    def load_training_data(self):
        _train_data = pd.read_csv('data_new/train_cutoff_sentences_multi.csv', encoding='utf-8')
        print(_train_data.shape)
        chars = set()
        char_dict = {}
        for s in _train_data["sentence"]:
            for c in s.lower():
                if c not in char_dict:
                    char_dict[c] = 1
                else:
                    char_dict[c] += 1

            #chars.update(s.lower())

        for a in global_char_dict:
            if a not in char_dict:
                    char_dict[a] = 1000

        self.char_to_int = ["<unk>"]
        for c in char_dict.keys():
            if (c in global_char_dict or char_dict[c] >= 10) and ad.is_latin(u"" + c): # and c != "†":
                self.char_to_int.append(c)


        self.n_vocab = len(self.char_to_int)
        print(self.char_to_int)
        print("Total Vocab: ", self.n_vocab)
        self.linear = nn.Linear(256, self.n_vocab)

        dataX = []
        dataY = []
        for i, row in _train_data.iterrows():
            currX = torch.tensor([self.char_to_int.index(char) if char in self.char_to_int else 0 for char in row["sentence"].lower() ])
            currX = currX.to("cuda" if torch.cuda.is_available() else "cpu")
            dataX.append(currX)
            label = row["label"].lower()
            dataY.append(self.char_to_int.index(label) if label in self.char_to_int else 0)

        X = pad_sequence(dataX, batch_first=True, padding_value = 0, padding_side='left')
        X = X[:, -100:].reshape((len(dataX), 100, 1))
        X = X / float(self.n_vocab)
        y = torch.tensor(dataY)
        X = X.to("cuda" if torch.cuda.is_available() else "cpu")
        y = y.to("cuda" if torch.cuda.is_available() else "cpu")

        loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=128)
        return loader

    @classmethod
    def load_test_data(cls, fname):
        test_data = []
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.replace("’","'")
                inp = line[:-1]  # the last character is a newline
                test_data.append(inp)
        loader = data.DataLoader(test_data, shuffle=False, batch_size=1)

        return loader

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.to(device)
        # Hyperparameters for training
        n_epochs = 300
        batch_size = 512
        eval_batch_size = 2048
        lr = 1e-3

        print("Beginning Training with Parameters:\n---Learning Rate: {l}"
              "\n---Batch Size: {b}\n---Eval Batch Size: {e}\n---Epochs: {E}\n"
              .format(l=lr, b=batch_size, e=eval_batch_size, E=n_epochs))

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr)
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

        best_model = None
        best_loss = np.inf
        total_time = 0.0
        for epoch in range(n_epochs):
            start_time = time.time()
            self.train()
            for X_batch, y_batch in data:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                y_pred = self(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Validation
            self.eval()
            loss = 0
            with torch.no_grad():
                for X_batch, y_batch in data:
                    y_pred = self(X_batch.to(device))
                    loss += loss_fn(y_pred, y_batch.to(device))
                if loss < best_loss:
                    best_loss = loss
                    self.best_model = self.state_dict()
                end_time = time.time()
                total_time += end_time - start_time
                percent_done = round((epoch + 1)/n_epochs * 100, 1)
                average_time = round(total_time / (epoch + 1), 3)
                sys.stdout.write('\r[{0}{1} {2}%] Epoch: {3} Average Epoch Time: {4}'.format
                                 ('#' * (int(epoch / 10)),'-' * (int(n_epochs / 10) - int(epoch / 10)),
                                  percent_done, epoch + 1, average_time))

        self.save(work_dir)
        print()

    def run_pred(self, data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        # your code here
        preds = []
        total_time = 0.0
        total_len = len(data)
        progress = 0
        for prompt in data:
            progress += 1
            start_time = time.time()
            pattern = [self.char_to_int.index(c) if c in self.char_to_int else 0 for c in prompt[0].lower()]
            pattern = torch.as_tensor(pattern)
            pattern = pattern.to(device)
            if pattern.shape[0] > 100:
                pattern = pattern[-100:]
            else:
                padding = (100 - pattern.shape[0], 0)
                pattern = F.pad(pattern, padding, mode='constant', value=0)

            self.eval()
            with torch.no_grad():
                # format input array of int into PyTorch tensor
                x = np.reshape(pattern.cpu(), (1, len(pattern), 1)) / float(self.n_vocab)
                x = torch.tensor(x, dtype=torch.float32, device=device)
                # generate logits as output from the model
                prediction = model(x)
                # convert logits into one character
                indices = torch.topk(prediction, 4).indices.cpu().numpy()
                values = []
                i = 0
                while len(values) < 3:
                    ind = indices[0][i]
                    if ind != 0:
                        values.append(self.char_to_int[ind])
                    i += 1
                #values = [self.char_to_int[i] for i in indices[0]]
                preds.append("".join(values))

                end_time = time.time()
                total_time += end_time - start_time
                percent_done = round(total_len/progress * 100, 1)
                average_time = round(total_time / progress, 3)
                sys.stdout.write('\r[{0}{1} {2}%] Prediction: {3} Average Prediction Time: {4}'.format
                                 ('#' * (int(progress / 10)),'-' * (int(total_len / 10) - int(progress / 10)),
                                  percent_done, progress + 1, average_time))
        print()
        return preds

    def save(self, work_dir):
        # Save the model
        torch.save([self.best_model, self.char_to_int], f'{work_dir}/lstm.model.checkpoint')

    @classmethod
    def load(cls, work_dir):
        # Load the model
        nn_model = MyModel()
        best_model, char_to_int = torch.load(f'{work_dir}/lstm.model.checkpoint', map_location=torch.device('cpu'))
        nn_model.linear = nn.Linear(256, len(char_to_int))
        nn_model.load_state_dict(best_model)
        nn_model.char_to_int = char_to_int
        return nn_model



if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel().to("cuda" if torch.cuda.is_available() else "cpu")
        print('Loading training data')
        train_data = model.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir).to("cuda" if torch.cuda.is_available() else "cpu")
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
