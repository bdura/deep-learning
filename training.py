import time
import collections
import os
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

import models


# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word


# Processes the raw data from text files
def ptb_valid_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")

    word_to_id, id_2_word = _build_vocab(train_path)

    valid_data = _file_to_word_ids(valid_path, word_to_id)

    return valid_data, word_to_id, id_2_word


# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."

    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# # LOAD DATA
# print('Loading data from ' + args.data)
# raw_data = ptb_raw_data(data_path=args.data)
# train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
# vocab_size = len(word_to_id)
# print('  vocabulary size: {}'.format(vocab_size))


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    
    This prevents Pytorch from trying to backpropagate into previous input 
    sequences when we use the final hidden states from one mini-batch as the 
    initial hidden states for the next mini-batch.
    
    Using the final hidden states in this way makes sense when the elements of 
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


def run_epoch(model, data, optimizer=None, is_train=False, lr=1.0,
              device=None, debug=False, return_grads=False, max_steps=-1):
    """
    One epoch of training/validation (depending on flag is_train).
    """

    loss_fn = nn.CrossEntropyLoss()

    model.keep_hiddens = return_grads

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_train or return_grads:
        model.train()
    else:
        model.eval()
    epoch_size = ((len(data) // model.batch_size) - 1) // model.seq_len
    start_time = time.time()
    if not isinstance(model, models.FullTransformer):
        hidden = model.init_hidden()
        hidden = hidden.to(device)

    costs = 0.0
    iters = 0
    losses = []

    per_t_losses = []

    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        if isinstance(model, models.FullTransformer):
            batch = Batch(torch.from_numpy(x).long().to(device))
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1, 0)
        else:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)
        tt = torch.squeeze(targets.view(-1, model.batch_size * model.seq_len))

        # LOSS COMPUTATION
        # This line currently averages across all the sequences in a mini-batch 
        # and all time-steps of the sequences.
        # For problem 5.1, you will (instead) need to compute the average loss
        # at each time-step separately.

        t = torch.from_numpy(y.astype(np.int64)).transpose(0, 1)

        per_t_loss = [
            loss_fn(o, t[i])
            for i, o in enumerate(outputs)
        ]
        last_loss = per_t_loss[-1]

        per_t_loss = np.array([l.item() for l in per_t_loss])

        per_t_losses.append(per_t_loss)

        loss = loss_fn(outputs.contiguous().view(-1, model.vocab_size).to(device), tt)
        costs += loss.data.item() * model.seq_len
        losses.append(costs)
        iters += model.seq_len

        if debug:
            print(step, loss)
        if is_train:  # Only update parameters if training 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            if isinstance(optimizer, torch.optim.Adam):
                optimizer.step()
            else:
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.add_(-lr, p.grad.data)
            if step % (epoch_size // 10) == 10:
                print('step: ' + str(step) + '\t'
                      'loss (sum over all examples seen this epoch):' + str(costs) + '\t'
                      'speed (wps):' + str(iters * model.batch_size / (time.time() - start_time)))

        if step + 1 == max_steps:
            break

    if not isinstance(model, models.FullTransformer) and return_grads:
        last_loss.backward()
        gradients = [torch.cat(tuple(h.grad for h in hid)) for hid in model.hiddens]

        return np.exp(costs / iters), losses, np.stack(per_t_losses), gradients

    return np.exp(costs / iters), losses, np.stack(per_t_losses)


def get_distribution(data, batch_size, seq_len):

    first_tokens = []

    for x, y in ptb_iterator(data, batch_size, seq_len):
        inputs = x.astype(np.int64).transpose(0, 1)
        first_tokens.append(inputs[0])

    first_tokens = np.concatenate(first_tokens)

    return first_tokens


def generate_sequence(model, hidden, i2w, tokens, sequence_length=35):

    init = np.random.choice(tokens, size=model.batch_size)

    sequences = []

    model.eval()

    with torch.no_grad():

        init = torch.tensor(init, dtype=torch.long)

        for tokens in model.generate(init, hidden, sequence_length):

            tokens = np.array(tokens)
            words = [i2w[token] for token in tokens]

            sequences.append(' '.join(words))

    return sequences


# MAIN LOOP
def main_loop(num_epochs, optimizer, model, train_data, valid_data,
              save_best, lr_schedule, lr_decay_base, m_flat_lr, lr, save_dir):
    train_ppls = []
    train_losses = []
    val_ppls = []
    val_losses = []
    times = []

    for epoch in range(num_epochs):
        t0 = time.time()
        print('\nEPOCH ' + str(epoch) + ' ------------------')
        if lr_schedule:
            lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
            lr *= lr_decay  # decay lr if it is time

        # RUN MODEL ON TRAINING DATA
        train_ppl, train_loss = run_epoch(model, train_data, optimizer, True, lr)

        # RUN MODEL ON VALIDATION DATA
        val_ppl, val_loss = run_epoch(model, valid_data)

        # SAVE MODEL IF IT'S THE BEST SO FAR
        if val_ppl < best_val_so_far:
            best_val_so_far = val_ppl
            if save_best:
                print("Saving model parameters to best_params.pt")
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_params.pt'))
            # NOTE ==============================================
            # You will need to load these parameters into the same model
            # for a couple Problems: so that you can compute the gradient
            # of the loss w.r.t. hidden state as required in Problem 5.2
            # and to sample from the the model as required in Problem 5.3
            # We are not asking you to run on the test data, but if you
            # want to look at test performance you would load the saved
            # model and run on the test data with batch_size=1

        # LOC RESULTS
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        train_losses.extend(train_loss)
        val_losses.extend(val_loss)
        times.append(time.time() - t0)
        log_str = 'epoch: ' + str(epoch) + '\t' \
                  + 'train ppl: ' + str(train_ppl) + '\t' \
                  + 'val ppl: ' + str(val_ppl) + '\t' \
                  + 'best val: ' + str(best_val_so_far) + '\t' \
                  + 'time (s) spent in epoch: ' + str(times[-1])
        print(log_str)
        with open(os.path.join(save_dir, 'log.txt'), 'a') as f_:
            f_.write(log_str + '\n')

    # SAVE LEARNING CURVES
    lc_path = os.path.join(save_dir, 'learning_curves.npy')
    print('\nDONE\n\nSaving learning curves to ' + lc_path)
    np.save(lc_path, {'train_ppls': train_ppls,
                      'val_ppls': val_ppls,
                      'train_losses': train_losses,
                      'val_losses': val_losses,
                      'times': times})


if __name__ == '__main__':

    gru = models.GRU(batch_size=20, seq_len=35, hidden_size=1500, num_layers=2,
                     vocab_size=10000, dp_keep_prob=.35, emb_size=200)

    valid_data, word_to_id, id_2_word = ptb_valid_data(data_path='data/')

    tokens = get_distribution(valid_data, 20, 35)

    generate_sequence(gru, word_to_id, id_2_word, tokens, sequence_length=35)
