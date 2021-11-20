import torch
from torch.autograd import Variable


def train_seq2seq(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i in range(len(iterator)):
        batch = next(iterator)
        src = batch['src']
        trg = batch['trg']

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
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / (len(iterator))


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
