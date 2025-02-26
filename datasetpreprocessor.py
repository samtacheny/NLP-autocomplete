import pandas as pd
import numpy as np
from datasets import load_dataset
import re

# 80/20 train/dev split
sentence_per_article = 10 # Number of train sentences taken per article
train_articles = 1000 # Number of articles used for training
num_train_sentences = sentence_per_article * train_articles
dev_articles = 0.2 * train_articles # Number of articles used for validation
num_dev_sentences = sentence_per_article * dev_articles

# dataset = load_dataset("wikipedia", "20220301.en")
data_dir = "data_new"

# Top 15 numbers of wiki articles
# English, Cebuano, German, French, Swedish, Dutch, Spanish, Italian, Polish, Vietnamese

# Romance, Romance, Germnanic, Germanic, Austronesian, Slavic, Austroasiatic
# Spanish, French, English, German, Cebuano, Polish, Vietnemese, Swedish
def load_data():
    languages = ['es', 'fr', 'en', 'de', 'ceb', 'pl', 'sv']
    datasets = []
    for lang in languages:
        datasets.append(load_dataset("wikimedia/wikipedia", f"20231101.{lang}", trust_remote_code=True))
        # datasets.append(load_dataset("wikipedia", "20220301.simple", trust_remote_code=True))
    return datasets

def cutoff_sentence(dataset):
    dataset = dataset.shuffle()
    train_valid = dataset['train'].train_test_split(test_size=0.2)
    train_data = train_valid['train']
    dev_data = train_valid['test']

    train_sentences = []
    train_labels = []
    sentence_counter = 0
    for article in train_data:
        if sentence_counter > num_train_sentences:
             break
        text = article["text"]
        # Removes references
        reference_index = text.rfind("References")
        text = text[:reference_index]
        # Split on sentences
        sentences = text.split(".") # Include other forms of punctuation?
        article_counter = 0
        for s in sentences:
            # Remove text with lots of numbers
            num_numbers = len(re.findall('[0-9]', s))
            if num_numbers > 4:
                 continue
            s = re.sub('[0-9]+', '', s) # Remove numbers
            s = re.sub('\n+', '', str(s)) # Remove newlines
            s = re.sub("' '+", ' ', s) # Standardize spaces
            s = s.strip() # Strip leading whitespace
            # print(s)
            sen_len = len(s)
            if sen_len > 10 and s[0].isupper(): # Keep only longish sentences
                cutoff = np.random.randint(5, sen_len)
                train_sentences.append(s[0:cutoff])
                train_labels.append(s[cutoff].lower())
                # Stop after threshold
                sentence_counter += 1
                article_counter += 1
            if article_counter > sentence_per_article or sentence_counter > num_train_sentences:
                break

    dev_sentences = []
    dev_labels = []
    sentence_counter = 0
    for article in dev_data:
        if sentence_counter > num_dev_sentences:
             break
        text = article["text"]
        # Removes references
        reference_index = text.rfind("References")
        text = text[:reference_index]
        # Split on sentences
        sentences = text.split(".") # Include other forms of punctuation?
        article_counter = 0
        for s in sentences:
            # Remove text with lots of numbers
            num_numbers = len(re.findall('[0-9]', s))
            if num_numbers > 4:
                 continue
            s = re.sub('[0-9]+', '', s) # Remove numbers
            s = re.sub('\n+', '', str(s)) # Remove newlines
            s = re.sub("' '+", ' ', s) # Standardize spaces
            s = s.strip() # Strip leading whitespace
            # print(s)
            sen_len = len(s)
            if sen_len > 10 and s[0].isupper(): # Keep only longish sentences
                cutoff = np.random.randint(5, sen_len)
                dev_sentences.append(s[0:cutoff])
                dev_labels.append(s[cutoff].lower())
                # Stop after threshold
                sentence_counter += 1
                article_counter += 1
            if article_counter > sentence_per_article or sentence_counter > num_train_sentences:
                break

    return train_sentences, train_labels, dev_sentences, dev_labels

def create_train(sentences, labels):
    train_df = pd.DataFrame(data={'sentence': sentences, 'label': labels})
    train_df.to_csv(f'{data_dir}/train_cutoff_sentences_multi.csv', index=False)

def create_dev(sentences, labels):
    with open(f'{data_dir}/dev_input_multi.txt', mode='w', encoding='utf-8') as input:
             input.write("\n".join(sentences) + "\n")
    with open(f'{data_dir}/dev_labels_multi.txt', mode='w', encoding='utf-8') as output:
             output.write("\n".join(labels) + "\n")

def main():
    print('Loading Data')
    datasets = load_data()
    print('Making Sentences')
    # Concatenate languages
    train_sentences = np.array([])
    train_labels = np.array([])
    dev_sentences = np.array([])
    dev_labels = np.array([])
    i = 1
    for dataset in datasets:
        print(f'Language {i}')
        t_sent, t_lab, d_sent, d_lab = cutoff_sentence(dataset)
        train_sentences = np.append(train_sentences, t_sent)
        train_labels = np.append(train_labels, t_lab)
        dev_sentences = np.append(dev_sentences, d_sent)
        dev_labels = np.append(dev_labels, d_lab)
        i = i + 1
    # Shuffle lists
    p_train = np.random.permutation(len(train_sentences))
    train_sentences = train_sentences[p_train]
    train_labels = train_labels[p_train]
    p_dev = np.random.permutation(len(dev_sentences))
    dev_sentences = dev_sentences[p_dev]
    dev_labels = dev_labels[p_dev]

    print('Writing training data')
    create_train(train_sentences, train_labels)
    print('Writing dev data')
    create_dev(dev_sentences, dev_labels)

if __name__ == '__main__':
    main()