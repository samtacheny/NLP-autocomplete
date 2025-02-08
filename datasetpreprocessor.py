import pandas as pd
import numpy as np
from datasets import load_dataset

dataset = load_dataset("wikipedia", "20220301.en")
data_dir = "../data"

train_df = pd.read_csv(
    f"{data_dir}/sst_train_binary.csv"
)

dev_df = pd.read_csv(
    f"{data_dir}/sst_dev_binary.csv"
)


def cutoff_sentence(data):
    new_sens = []
    labels = []
    for article in dataset:
        text = article["text"]
        sentences = text.split(". ")
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