import random

import numpy as np
import torch
from torch.autograd import Variable


def create_dataset(src_data, spacy_src, trg_data, spacy_trg):
    min_len = min(len(src_data), len(trg_data))
    data_set = []
    for i in range(min_len):
        src = ['<sos>'] + [tok.text for tok in spacy_src.tokenizer(src_data[i])][::-1] + ['<eos>']
        trg = ['<sos>'] + [tok.text for tok in spacy_trg.tokenizer(trg_data[i])] + ['<eos>']
        if '\n' in src:
            src.remove('\n')
        if '\n' in trg:
            trg.remove('\n')
        data_set.append({'src': src, 'trg': trg})

    return data_set


def build_vocab(data_set):
    src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    trg_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    src_allChar_list = []
    trg_allChar_list = []
    for item in data_set:
        for char in item['src']:
            if char not in src_allChar_list:
                src_allChar_list.append(char)
                continue
            if (char in src_allChar_list) and (char not in src_vocab.keys()):
                src_vocab[char] = len(src_vocab)

        for char in item['trg']:
            if char not in trg_allChar_list:
                trg_allChar_list.append(char)
                continue
            if (char in trg_allChar_list) and (char not in trg_vocab.keys()):
                trg_vocab[char] = len(trg_vocab)
    return src_vocab, trg_vocab


class BuildBatch:
    def __init__(self, data, batch_size, src_vocab, trg_vocab, shuffle=True):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.data = data

        self.size = len(data)
        self.indices = list(range(self.size))
        if shuffle:
            random.shuffle(self.indices)

        self.start = 0
        self.end = self.start + self.batch_size

        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __iter__(self):
        return self

    def __len__(self):
        return self.steps

    def __next__(self):
        while self.start <= self.size:
            if self.end >= self.size:
                self.end = self.size
            batch_indices = np.asarray(self.indices[self.start:self.end])
            self.start += self.batch_size
            self.end = self.start + self.batch_size

            src_batch_list = []
            src_len = []

            trg_batch_list = []
            trg_len = []

            for indice in batch_indices:
                src = self.data[indice]['src']
                src_len.append(len(src))
                trg = self.data[indice]['trg']
                trg_len.append(len(trg))

                src_list = []
                trg_list = []
                # 利用vocab对字符编码
                for item in src:
                    src_list.append(self.src_vocab.get(item, 3))  # 3 代表 <unk>
                for item in trg:
                    trg_list.append(self.trg_vocab.get(item, 3))

                src_batch_list.append(src_list)
                trg_batch_list.append(trg_list)
            src_max_len = max(src_len)
            trg_max_len = max(trg_len)

            src_data = np.zeros((src_max_len, self.batch_size))
            trg_data = np.zeros((trg_max_len, self.batch_size))

            for i in range(len(src_batch_list)):
                for j in range(len(src_batch_list[i])):
                    src_data[j, i] = int(src_batch_list[i][j])

            src_data = torch.LongTensor(src_data)

            for i in range(len(trg_batch_list)):
                for j in range(len(trg_batch_list[i])):
                    trg_data[j, i] = int(trg_batch_list[i][j])

            trg_data = torch.LongTensor(trg_data)

            return {'src': src_data, 'trg': trg_data}
