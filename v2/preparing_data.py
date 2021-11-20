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


def get_batch_data(data, src_vocab, trg_vocab):
    src_batch_list = []
    trg_batch_list = []

    src_len = []
    trg_len = []

    for item in data:
        item_src = item['src']
        item_trg = item['trg']

        src_len.append(len(item_src))
        trg_len.append(len(item_trg))

        src_list = []
        trg_list = []

        for char in item_src:
            src_list.append(src_vocab.get(char, 3))  # 3 代表 <unk>
        for char in item_trg:
            trg_list.append(trg_vocab.get(char, 3))

        src_batch_list.append(src_list)
        trg_batch_list.append(trg_list)

    src_max_len = max(src_len)
    trg_max_len = max(trg_len)

    src_batch_data = np.zeros((src_max_len, len(src_batch_list)))
    trg_batch_data = np.zeros((trg_max_len, len(trg_batch_list)))

    for i in range(len(src_batch_list)):
        for j in range(len(src_batch_list[i])):
            src_batch_data[j, i] = int(src_batch_list[i][j])

    src_batch_data = torch.LongTensor(src_batch_data)

    for i in range(len(trg_batch_list)):
        for j in range(len(trg_batch_list[i])):
            trg_batch_data[j, i] = int(trg_batch_list[i][j])

    trg_batch_data = torch.LongTensor(trg_batch_data)
    return src_batch_data, trg_batch_data
