import matplotlib.pyplot as plt
import nnfs
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from nnfs.datasets import spiral_data, vertical_data

import data_loader_two_by_two as dat


class ActivationFunction(ABC):
    @staticmethod
    @abstractmethod
    def forward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

    @staticmethod
    @abstractmethod
    def backwards(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass


class LinearActivation(ActivationFunction, ABC):
    @staticmethod
    def forward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return sample

    @staticmethod
    def backward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.ones_like(sample)


class SigmoidActivation(ActivationFunction, ABC):
    @staticmethod
    def forward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1 / (1 + np.exp(-sample))

    @staticmethod
    def backward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return sample * (1 - sample)


class ReLuActivation(ActivationFunction, ABC):
    @staticmethod
    def forward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.maximum(sample, 0)

    @staticmethod
    def backward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.where(sample > 0, 1, 0)


class SoftmaxActivation(ActivationFunction, ABC):
    @staticmethod
    def forward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        exp_sample = np.exp(sample - np.max(sample, axis=1, keepdims=True))
        return exp_sample / np.sum(exp_sample, axis=1, keepdims=True)

    @staticmethod
    def backward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return sample * (1 - sample)


class TanhActivation(ActivationFunction, ABC):
    @staticmethod
    def forward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.tanh(sample)

    @staticmethod
    def backward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1 - sample ** 2


class LeakyReLuActivation(ActivationFunction, ABC):
    @staticmethod
    def forward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.where(sample > 0, sample, sample * 0.01)

    @staticmethod
    def backward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.where(sample > 0, 1, 0.01)


class StepActivation(ActivationFunction, ABC):
    @staticmethod
    def forward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.where(sample >= 0, 1, 0)

    @staticmethod
    def backward(sample: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.zeros_like(sample)


class LossFunctions:
    @staticmethod
    def mean_squared_error(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def cross_entropy(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.mean(-np.log(np.clip(y_pred[range(y_true.shape[0]), y_true], 1e-7, 1 - 1e-7)))


class Layer:
    def __init__(self, inputs):
        if len(inputs) < 1:
            raise ValueError("Need at least one data")
        if len(inputs[0]) < 1:
            raise ValueError("Need at least one neuron")
        self.output = inputs

    def getoutput(self):
        return self.output

    def getsize(self):
        return len(self.output[0])

    def getbatchsize(self):
        return len(self.output)


class HiddenLayer(Layer):
    def __init__(self, prevLayer, numOfNeurons: int, weightss=None, biass=None, actfunc=LinearActivation):
        self.activationfunc = actfunc
        self.input = prevLayer
        self.batchsize = self.input.getbatchsize()
        super().__init__([[0] * numOfNeurons for _ in range(self.batchsize)])
        self.layerShape = numOfNeurons
        self.weights = weightss or 0.1 * np.random.randn(self.input.getsize(), self.layerShape)
        self.biases = biass or np.zeros((1, self.layerShape))
        if weightss and biass:
            if len(weightss) != len(biass) != numOfNeurons:
                raise ValueError("Number of weights or biases doesn't match number of neurons.")
            for weigh in weightss:
                if len(weigh) != self.input.getsize():
                    raise ValueError("At least one of the neuron's weights doesn't match the given input layer")

    def tweek_node_bias(self, neuron: int, db: int):
        if neuron < 0 or neuron >= self.layerShape:
            raise ValueError(f"neuron {neuron} is outOfBounds. There are {self.layerShape} neurons.")
        self.biases[neuron] += db

    def tweek_node_weights(self, neuron: int, input: int, dw: int):
        if neuron < 0 or neuron >= self.layerShape:
            raise ValueError(f"neuron {neuron} is outOfBounds. There are {self.layerShape} neurons.")
        if neuron < 0 or neuron >= self.layerShape:
            raise ValueError(
                f"neuron {neuron} is outOfBounds. There are {self.input.getsize()} neurons in the previous layer.")
        self.weights[neuron][input] += dw

    def forward(self):
        self.input.forward()
        pre_func = np.dot(np.array(self.input.getoutput()), np.array(self.weights)) + self.biases
        self.output = self.activationfunc.forward(pre_func)

    def loss(self, target: npt.NDArray[np.float64], lossfunc=LossFunctions.cross_entropy):
        return lossfunc(target, self.output)

    def backword(self, target: npt.NDArray[np.float64], lossfunc=LossFunctions.cross_entropy):
        loss = self.loss(target, lossfunc)


class InputLayer(Layer):
    def __init__(self, inputs):
        super().__init__(inputs)

    def forward(self):
        return self.output


if __name__ == "__main__":
    nnfs.init()

    # Init
    training_set, evaluation_set = dat.get_data_set()
    X, y = vertical_data(100, 3)
    spiral_X, spiral_y = spiral_data(100, 3)
    plt.scatter(spiral_X[:, 0], spiral_X[:, 1], c=y, s=40, cmap='brg')
    plt.show()

    spiral_X_layer = InputLayer(spiral_X)
    spiral_layer1 = HiddenLayer(spiral_X_layer, 3, actfunc=ReLuActivation)
    spiral_layer2 = HiddenLayer(spiral_layer1, 3, actfunc=SoftmaxActivation)

    # Run
    sample = next(training_set)
    n_pixels = sample.shape[0] * sample.shape[1]
    n_nodes = [n_pixels, n_pixels]

    spiral_layer2.forward()
    print(spiral_layer2.getoutput())

    print(spiral_layer2.loss(spiral_y, LossFunctions.cross_entropy))
