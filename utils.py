import matplotlib.pyplot as plt
import numpy as np

import torch

import os

import collections


def get_curves(folder):
    """
    Loads the learning curves from the experiment folder

    Returns:
        (dict): The dictionary containing the curves.
    """
    return np.load(folder + 'learning_curves.npy').item()


def get_times(folder):
    """
    The time per epoch is not saved... But it is logged.

    Args:
        folder (str): The experiment folder.

    Returns:
        The time spent on each epoch.
    """

    with open(folder + 'log.txt', 'r') as f:
        lines = f.readlines()

    lines = [float(line.split('\t')[-1].split(' ')[-1]) for line in lines]

    return np.array(lines)


def plot(train, valid, name, save=True, log_scale=False):
    plt.figure()

    if isinstance(train, list):
        for t, v, n in zip(train, valid, name):
            plt.plot(t, label='{} train'.format(n))
            plt.plot(v, label='{} validation'.format(n))
    else:
        plt.plot(train, label='train')
        plt.plot(valid, label='validation')

    plt.legend()

    plt.title(name)

    plt.ylabel('Perplexity (log scale)' if log_scale else 'Perplexity')
    plt.xlabel('Epoch')

    plt.yscale('log' if log_scale else 'linear')

    if log_scale:
        filename = 'output/{}-log-ppl.pdf'.format(name)
    else:
        filename = 'output/{}-ppl.pdf'.format(name)

    if save:
        plt.savefig(filename, bbox_inches='tight')


def plot_time(train, valid, times, name, save=True, log_scale=False):
    plt.figure()

    if isinstance(train, list):
        hours = [time / 3600 for time in times]
        for t, v, h, n in zip(train, valid, hours, name):
            plt.plot(h, t, label='{} train'.format(n))
            plt.plot(h, v, label='{} validation'.format(n))
    else:
        hours = times / 3600

        plt.plot(hours, train, label='train')
        plt.plot(hours, valid, label='validation')

    plt.legend()

    plt.title(name)

    plt.ylabel('Perplexity (log scale)' if log_scale else 'Perplexity')
    plt.xlabel('Wall time (h)')

    plt.yscale('log' if log_scale else 'linear')

    if log_scale:
        filename = 'output/{}-log-ppl.pdf'.format(name)
    else:
        filename = 'output/{}-ppl.pdf'.format(name)

    if save:
        plt.savefig(filename, bbox_inches='tight')


def plot_ppl(name, folder, save=True, log_scale=False, time=False):

    curves = get_curves(folder)

    times = get_times(folder)
    times = np.cumsum(times)

    if time:
        plot_time(curves['train_ppls'], curves['val_ppls'], times, name + '-time', save, log_scale)
    else:
        plot(curves['train_ppls'], curves['val_ppls'], name, save, log_scale)


def plot_ppls(experiments, save=True, log_scale=False, time=False):

    folders = [e['folder'] for e in experiments]
    names = [e['name'] for e in experiments]

    curves = [get_curves(folder) for folder in folders]

    times = [np.cumsum(get_times(folder)) for folder in folders]

    if time:
        plot_time(
            [curve['train_ppls'] for curve in curves],
            [curve['val_ppls'] for curve in curves],
            times,
            names,
            save,
            log_scale
        )
    else:
        plot(
            [curve['train_ppls'] for curve in curves],
            [curve['val_ppls'] for curve in curves],
            names,
            save,
            log_scale
        )


def load_weights(folder):
    model = torch.load(folder + 'best_params.pt', map_location='cpu')
    return model
