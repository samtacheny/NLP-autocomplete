import pandas as pd
import numpy as np
from datasets import load_dataset
import re

data_dir = "data_new"
data_suffix = 'multi'

def load_data(languages):
    datasets = []
    for lang in languages:
        datasets.append(load_dataset("wikimedia/wikipedia", f"20231101.{lang}", trust_remote_code=True))
        # datasets.append(load_dataset("wikipedia", "20220301.simple", trust_remote_code=True))
    return datasets

def cutoff_sentence(dataset, symbols, reference, num_train_sentences, num_dev_sentences, sentence_per_article):
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
        for word in reference:
            reference_index = text.rfind(word) 
            text = text[:reference_index]
        # Split on sentences
        sentences = text.split(".")
        article_counter = 0
        for s in sentences:
            s = str(s)
            # Ignore sentences with lots of numbers (throws off the flow once removed)
            num_numbers = len(re.findall('[0-9]', s))
            if num_numbers > 4:
                 continue
            s = re.sub('[0-9]+', ' ', str(s)) # Remove numbers
            # Replace quotations
            s = re.sub('[‘’]', '\'', s) # Replace quotation marks
            s = re.sub("[“”„]", '"', s)
            for sym in symbols: # Remove symbols
                 s = s.replace(sym, '')
            s = re.sub('\n+', '', s) # Remove newlines
            s = re.sub("[\u200b\xa0 ]+", ' ', s) # Standardize spaces
            s = s.strip() # Strip leading whitespace
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
        for word in reference:
            reference_index = text.rfind(word) 
            text = text[:reference_index]
        # Split on sentences
        sentences = text.split(".") # Include other forms of punctuation?
        article_counter = 0
        for s in sentences:
            s = str(s)
            # Ignore sentences with lots of numbers (throws off the flow once removed)
            num_numbers = len(re.findall('[0-9]', s))
            if num_numbers > 4:
                 continue
            s = re.sub('[0-9]+', ' ', s) # Remove numbers
            # Replace quotations
            s = re.sub('[‘’]', '\'', s) # Replace quotation marks
            s = re.sub("[“”„]", '"', s)
            for sym in symbols: # Remove symbols
                 s = s.replace(sym, '')
            s = re.sub('\n+', '', s) # Remove newlines
            s = re.sub("[\u200b\xa0 ]+", ' ', s) # Standardize spaces
            s = s.strip() # Strip leading whitespace
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
    train_df.to_csv(f'{data_dir}/train_cutoff_sentences_{data_suffix}.csv', index=False)

def create_dev(sentences, labels):
    with open(f'{data_dir}/dev_input_{data_suffix}.txt', mode='w', encoding='utf-8') as input:
             input.write("\n".join(sentences) + "\n")
    with open(f'{data_dir}/dev_labels_{data_suffix}.txt', mode='w', encoding='utf-8') as output:
             output.write("\n".join(labels) + "\n")

sentence_per_article = 10 # Number of train sentences taken per article
languages = ['es', 'fr', 'en', 'de', 'ceb', 'pl', 'sv'] # Spanish, French, English, German, Cebuano, Polish, Swedish
train_length = [1500, 1500, 1500, 1500, 500, 500, 500] * sentence_per_article
references = [['Referencias'], ['Notes et références'], ['References'], ['Literatur'], [], ['Przypisy'], ['Noter', 'Källor', 'Referenser']]
symbols = ['«', '»', '°', '(', ')', '-', ';', '–', '!', '|', '/', '+', '[', ']', '*', '—', '=', '#', '$', '.']

def main():
    print('Loading Data')
    datasets = load_data(languages)
    print('Making Sentences')
    # Concatenate languages
    train_sentences = np.array([])
    train_labels = np.array([])
    dev_sentences = np.array([])
    dev_labels = np.array([])
    for i in range(len(datasets)):
        print(f'Language {i + 1}')
        dataset = datasets[i] # Dataset
        reference = references[i] # List of words to cut out references
        num_train_sentences = train_length[i] # Number of sentences for training
        num_dev_sentences = 0.2 * num_train_sentences # Number of sentences for validation
        t_sent, t_lab, d_sent, d_lab = cutoff_sentence(dataset, symbols, reference, 
                                                       num_train_sentences, num_dev_sentences, sentence_per_article)
        train_sentences = np.append(train_sentences, t_sent)
        train_labels = np.append(train_labels, t_lab)
        dev_sentences = np.append(dev_sentences, d_sent)
        dev_labels = np.append(dev_labels, d_lab)
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