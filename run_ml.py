import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from nnfs.datasets import spiral_data, vertical_data

import data_loader_two_by_two as dat
import nn_framework.framework as nnf

train_set, eval_set = dat.get_data_set()

sample = next(train_set())
n_pix = sample.shape[0] * sample.shape[1]
n_nodes = [n_pix, n_pix]

autoencoder = nnf.ANN(model=None)
autoencoder.train(train_set)
autoencoder.evaluate(eval_set)
