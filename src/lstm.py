import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd

class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output -- this allows sequences of variable length?
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x

best_model, char_to_int = torch.load("/Users/sophi/Downloads/UW/CSE_447/cse447-project/work/45_epochs_single-char.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())

model = CharModel()
model.load_state_dict(best_model)

dev_sentences = []
with open("/Users/sophi/Downloads/UW/CSE_447/cse447-project/data_new/dev_input.txt") as file:
    for line in file:
        dev_sentences.append(line.replace("\n", ""))

dev_labels = []
with open("/Users/sophi/Downloads/UW/CSE_447/cse447-project/data_new/dev_labels.txt") as file:
    for line in file:
        dev_labels.append(line.replace("\n", ""))

dev_data = list(zip(dev_sentences, dev_labels))
# print(dev_data)

# test if sequence length can be different
seq_length = 100
preds = []

num_correct = 0
for i in range(len(dev_data)):
    prompt, true = dev_data[i]
    y_true = char_to_int[true] if true in char_to_int else 0
    
    pattern = [char_to_int[c] if c in char_to_int else 0 for c in prompt.lower()]
    pattern = torch.as_tensor(pattern)
    if pattern.shape[0] > 100:
        pattern = pattern[-100:]
    else:
        padding = (100 - pattern.shape[0], 0)
        pattern = F.pad(pattern, padding, mode='constant', value=0)
        
    #print(pattern)
    model.eval()
    #print('Prompt: "%s"' % prompt)
    with torch.no_grad():
        # format input array of int into PyTorch tensor
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(x)
        # convert logits into one character
        indices = torch.topk(prediction, 3).indices.numpy()
        #print(indices[0])
        values = [int_to_char[i] for i in indices[0]]
        preds.append("".join(values))
        if y_true in indices:
            num_correct += 1
        # append the new character into the prompt for the next iteration
print(f"Accuracy: {num_correct/len(dev_data)}")

with open("lstm_data/dev_pred.txt", 'wt') as f:
    for p in preds:
        f.write('{}\n'.format(p))

    
