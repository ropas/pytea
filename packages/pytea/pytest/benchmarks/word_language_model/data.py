import os
from io import open
import torch
import random


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

        # <FIXED LINE FOR EXPERIMENT>
        self.length = random.randint(100, 1000)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        # <FIXED LINE FOR EXPERIMENT>
        # return len(self.idx2word)

        return self.length


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, "train.txt"))
        self.valid = self.tokenize(os.path.join(path, "valid.txt"))
        self.test = self.tokenize(os.path.join(path, "test.txt"))

    def tokenize(self, path):
        # <FIXED LINE FOR EXPERIMENT>: set constant data

        # """Tokenizes a text file."""
        # assert os.path.exists(path)
        # # Add words to the dictionary
        # with open(path, 'r', encoding="utf8") as f:
        #     for line in f:
        #         words = line.split() + ['<eos>']
        #         for word in words:
        #             self.dictionary.add_word(word)

        # # Tokenize file content
        # with open(path, 'r', encoding="utf8") as f:
        #     idss = []
        #     for line in f:
        #         words = line.split() + ['<eos>']
        #         ids = []
        #         for word in words:
        #             ids.append(self.dictionary.word2idx[word])
        #         idss.append(torch.tensor(ids).type(torch.int64))
        #     ids = torch.cat(idss)

        # return ids

        length = 1024
        return torch.rand(length).type(torch.int64)
