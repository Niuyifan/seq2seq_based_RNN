import random

import torch

from v2.preparing_data import get_batch_data


def train_batch(model, data_set, src_vocab, trg_vocab, batch_size, optimizer, criterion, clip, shuffle=False):
    model.train()
    epoch_loss = 0
    epoch_num = 0

    indices = list(range(len(data_set)))
    if shuffle:
        random.shuffle(indices)

    start = 0
    count = 0

    while start < len(data_set):
        count += 1
        end = start + batch_size if start + batch_size < len(indices) else len(indices)

        src, trg = get_batch_data(data_set, indices, start, end, src_vocab, trg_vocab)

        start = end

        optimizer.zero_grad()

        output = model(src, trg)
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)

        trg = trg[1:].view(-1)
        # trg : [(trg len - 1) * batch size]
        # output : [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        print('第'+str(count)+'个batch的loss:',loss)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        epoch_num += 1

    return epoch_loss / epoch_num


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
