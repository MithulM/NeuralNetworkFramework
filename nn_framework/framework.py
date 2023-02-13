import numpy as np


class ANN:
    def __init__(self, model=None):
        self.layers = model
        self.n_train = int(1e6)
        self.n_eval = int(1e6)

    def train(self, training_set):
        for i in range(self.n_train):
            x = next(training_set()).ravel()
            print(x)

    def evaluate(self, evaluation_set):
        for i in range(self.n_eval):
            x = next(evaluation_set()).ravel()
            print(x)
