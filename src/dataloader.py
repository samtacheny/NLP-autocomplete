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

# Get sentence transformer embeddings
def get_st_embeddings(
    sentences: List[str],
    st_model: SentenceTransformer,
    batch_size: int = 32
):
    """
    Compute the sentence embedding using the Sentence Transformer model.

    Inputs:
    - sentence: The input sentence
    - st_model: SentenceTransformer model
    - batch_size: Encode in batches to avoid memory issues in case multiple sentences are passed

    Returns:
    torch.Tensor: The sentence embedding of shape [d,] (when only 1 sentence) or [n, d] where n is the number of sentences and d is the embedding dimension
    """
    sentence_embeddings = None

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i : i + batch_size]
        batch_embeddings = st_model.encode(batch_sentences, convert_to_tensor=True)
        if sentence_embeddings is None:
            sentence_embeddings = batch_embeddings
        else:
            sentence_embeddings = torch.cat(
                [sentence_embeddings, batch_embeddings], dim=0
            )

    return sentence_embeddings

# Creates datasets
class EmbeddedDataset(torch.utils.data.Dataset):

    def __init__(self, embeddings: List[Dict[str, torch.Tensor]], chars: List[str]):
        self.embeddings = embeddings
        self.chars = chars

        # Must be contiguous from 0 to num classes (e.g. num characters we care about)
        self.char2char_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 
                             'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 
                             'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 
                             'z': 25, '.': 26, ',': 27, '!': 28, '?': 29, ' ': 30, '<unk>': 31}

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        sample = self.embeddings[idx]
        return {
            "context": sample["context"],
            "word": sample["word"],
            # encodes character label as the ASCII value
            "char": self.char2char_id[self.chars[idx]] if self.chars[idx] in self.char2char_id.keys() else self.char2char_id['<unk>']
        }


def get_dataloader(
    embeddings: List[Dict[str, torch.Tensor]],
    labels: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
):
    dataset = EmbeddedDataset(embeddings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)