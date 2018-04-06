import os
import torch
import numpy as np
#from gensim.models.keyedvectors import KeyedVectors
#from gensim.models.word2vec import Word2Vec
#from gensim.models import FastText

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.pretrain_vec = [] # should match index order of words in dict.

    def add_word(self, word, vec=None):
        if vec is None:
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
        else:
            self.pretrain_vec.append(vec)
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, language):
        self.dictionary = Dictionary()
        if language is not None:
            self.pretrained = self.add_pretrained(os.path.join(path, 'wiki.' + language + '.vec'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def add_pretrained(self, path):
        assert os.path.exists(path)

        # Add words with pretrained vectors to the dictionary
        # might be weird because no eos was added?
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split()
                if len(words) == 2: #first line
                    continue
                word = words[0]
                vec = words[1:]
                if len(vec) != 300:
                    continue #this skips the space embedding
                #vec = np.array(list(map(float, vec)))
                vec = list(map(float,vec))
                tokens += 1
                
                self.dictionary.add_word(word, vec)
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
