import pandas as pd
import torch
import numpy as np

data_dir = "../data"

def load_data():
    train_df = pd.read_csv(
        f"{data_dir}/sst_train_binary.csv"
    )
    dev_df = pd.read_csv(
        f"{data_dir}/sst_dev_binary.csv"
    )
    return train_df, dev_df

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

    
def create_data(train_df, dev_df):
    cutoff_sentence(train_df)
    train_df.to_csv(f'{data_dir}/train_cutoff_sentences.csv', index=False)
    cutoff_sentence(dev_df)
    dev_df.to_csv(f'{data_dir}/dev_cutoff_sentences.csv', index=False)

# Turns CSV file into two txt files for prediction analysis
def dev_to_test_format():
        dev_data = pd.read_csv('data/dev_cutoff_sentences.csv', encoding='utf-8')
        sentences = dev_data['sentence'].tolist()  # Load training data as a list
        chars = dev_data['label'].tolist()  # Load characters (answers) as a list
        with open("dev_input.txt", mode='w', encoding='utf-8') as input:
             input.write("\n".join(sentences) + "\n")
        with open("dev_labels.txt", mode='w', encoding='utf-8') as labels:
             labels.write("\n".join(chars) + "\n")

def main():
    # train_df, dev_df = load_data()
    # create_data(train_df, dev_df)
    dev_to_test_format()

if __name__ == '__main__':
    main()