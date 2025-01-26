import pandas as pd
import torch
import numpy as np

data_dir = "../data"

train_df = pd.read_csv(
    f"{data_dir}/sst_train_binary.csv"
)

dev_df = pd.read_csv(
    f"{data_dir}/sst_dev_binary.csv"
)

def cutoff_sentence(data):
    sentences = data["sentence"].values
    new_sens = []
    labels = []
    for s in sentences:
        sen_len = len(s)
        cutoff = np.random.randint(1, sen_len)
        new_sens.append(s[0:cutoff])
        labels.append(s[cutoff].lower())

    data.drop('sentence', axis=1, inplace=True)
    data.drop('label', axis=1, inplace=True)
    data["sentence"] = new_sens
    data["label"] = labels

    

cutoff_sentence(train_df)
train_df.to_csv(f'{data_dir}/train_cutoff_sentences.csv', index=False)
cutoff_sentence(dev_df)
dev_df.to_csv(f'{data_dir}/dev_cutoff_sentences.csv', index=False)