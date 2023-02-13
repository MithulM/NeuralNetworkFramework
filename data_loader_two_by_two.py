import numpy as np
def get_data_set():
    examples = [
        np.array([[0, 0], [0, 0]]),
        np.array([[0, 0], [0, 1]]),
        np.array([[0, 0], [1, 0]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 1], [0, 1]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 1], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[1, 0], [0, 1]]),
        np.array([[1, 0], [1, 0]]),
        np.array([[1, 0], [1, 1]]),
        np.array([[1, 1], [0, 0]]),
        np.array([[1, 1], [0, 1]]),
        np.array([[1, 1], [1, 0]]),
        np.array([[1, 1], [1, 1]])
    ]

    def training_set():
        while True:
            yield np.random.choice(examples)
    def evaluation_set():
        while True:
            yield np.random.choice(examples)

    return training_set, evaluation_set