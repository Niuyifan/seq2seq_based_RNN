import math
import random
import time

import numpy as np
import spacy
import torch
from torch import nn, optim

from encoder_decoder import Encoder, Decoder
from preparing_data import create_dataset, build_vocab, BuildBatch
from seq2seq import Seq2Seq
from train_model import train_seq2seq, epoch_time



def init_weight(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


if __name__ == '__main__':
    SEED = 1234
    BATCH_SIZE = 4

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')

    en_train_data_file = open(r"E:\python_files\nlp\sequence to sequence\data_set\Multi30k dataset\task1\raw\train.en")
    de_train_data_file = open(r"E:\python_files\nlp\sequence to sequence\data_set\Multi30k dataset\task1\raw\train.de",
                              encoding='utf-8')
    en_test_data_file = open(
        r"E:\python_files\nlp\sequence to sequence\data_set\Multi30k dataset\task1\raw\test_2018_flickr.en")
    de_test_data_file = open(
        r"E:\python_files\nlp\sequence to sequence\data_set\Multi30k dataset\task1\raw\test_2018_flickr.de",
        encoding='utf-8')

    en_train_data = en_train_data_file.readlines()
    de_train_data = de_train_data_file.readlines()
    en_test_data = en_test_data_file.readlines()
    de_test_data = de_test_data_file.readlines()

    train_data = create_dataset(de_train_data, spacy_de, en_train_data, spacy_en)
    test_data = create_dataset(de_test_data, spacy_de, en_test_data, spacy_en)
    de_vocab, en_vocab = build_vocab(train_data)

    # train_batch_data = BuildBatch(train_data[:10], BATCH_SIZE, de_vocab, en_vocab, shuffle=False)

    INPUT_DIM = len(de_vocab)
    OUTPUT_DIM = len(en_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    N_EPOCHS = 10
    CLIP = 1

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec)

    model.apply(init_weight)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略<pad>标签

    print(model)

    for epoch in range(N_EPOCHS):
        train_batch_data = BuildBatch(train_data, BATCH_SIZE, de_vocab, en_vocab, shuffle=False)

        start_time = time.time()

        train_loss = train_seq2seq(model, train_batch_data, optimizer, criterion, CLIP)
        # valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
