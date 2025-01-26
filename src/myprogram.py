#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# From HW2
import json
import re
from typing import List, Tuple, Dict, Union
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import Adam

import dataloader

char_to_idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 
                'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 
                'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 
                'z': 25, '.': 26, ',': 27, '!': 28, '?': 29, ' ': 30, '<unk>': 31}

# Actual model
class MyModel(nn.Module):
    
    def __init__(self, sentence_dim: int, word_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the Feed-Foward model.
        Inputs:
        - sentence_dim: The dimension of the sentence embeddings
        - word_dim: The dimension of the final word embedding
        - hidden_dim: The dimension of the hidden layer
        - output_dim: The dimension of the output (e.g. number of ASCII characters)
        """

        super(MyModel, self).__init__()

        # Define the architecture of the model
        self.fc1 = torch.nn.Linear(sentence_dim + word_dim, hidden_dim)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, context: torch.Tensor, curr_word: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.
        Inputs:
        - context: The sentence embeddings of the context
        - question: The sentence embeddings of the question
        - answerA: The sentence embeddings of answer A
        - answerB: The sentence embeddings of answer B
        - answerC: The sentence embeddings of answer C
        Returns:
        - torch.Tensor: The logits for each of the answers
        """

        input = torch.cat((context, curr_word), dim=1) # concatenate sentence and word embeddings
        logit = self.fc2(self.act1(self.fc1(input))) # calculate logit
        return logit

    @classmethod
    def load_training_data(cls):
        # Make a model
        st_model = SentenceTransformer("all-mpnet-base-v2")
        # Load training data as a list - TODO: Sophie
        sentences = None
        embeddings = dataloader.get_st_embeddings(sentences, st_model) 
        # Load characters as a list - TODO: Sophie
        chars = None

        return dataloader.get_dataloader(embeddings, chars)

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # Hyperparameters for training
        lr: float = 1e-3,
        batch_size: int = 32,
        eval_batch_size: int = 128,
        n_epochs: int = 10

        train_loader = dataloader.get_dataloader(data, batch_size=batch_size, shuffle=True) # TODO- get dataloader
        
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(n_epochs): # Iterate over the epochs

            model.train() # Set the model to training mode
            for batch in train_loader: # Iterate over the batches of the training data
                optimizer.zero_grad()  # This is done to zero-out any existing gradients stored from previous steps

                # Get the embeddings for context, question, answerA, answerB, answerC
                context_batch = batch["context"]
                word_batch = batch["word"]

                # Get the characters
                char_batch = batch["label"]

                # Calculate model logits and loss
                batch_logits = model(context_batch, word_batch)
                batch_loss = loss_fn(batch_logits, char_batch)

                # Perform a backward pass and update the weights
                batch_loss.backward()

                # Perform a step of optimization
                optimizer.step()

        self.save(work_dir)

    def run_pred(self, data):
        # your code here
        preds = []
        for inp in data:
            # TODO: split into context and words (should be torch.Tensor)
            context, words = None
            logits = model(context, words)
            (_, top_indices) = torch.topk(logits, k=3)
            # TODO: turn top indices into list of three characters
            top_chars = None
            preds.append(''.join(list(top_chars)))

            # # this model just predicts a random character each time
            # top_guesses = [random.choice(all_chars) for _ in range(3)]
            # preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # Save the model
        torch.save(self.state_dict(), f'{work_dir}/model.checkpoint')

        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # TODO: figure out actual dimensions

        # Load the model
        nn_model = MyModel(
            input_dim=50, # Change this to 768 if you want to train with sentence transformer embeddings
            hidden_dim=100, # You can change this to any number of hidden units you want
            num_classes=1 # You can change this to the number of classes for the multiclass case
        )
        nn_model.load_state_dict(torch.load(f'{work_dir}/model.checkpoint'))
        return nn_model

        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


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
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
