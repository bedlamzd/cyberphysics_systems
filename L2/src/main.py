import csv
from typing import List

import numpy as np
# import matplotlib.pyplot as plt
from scipy.special import expit


class NetworkMNIST:
    def __init__(
            self,
            input_nodes: int,
            hidden_nodes: int,
            output_nodes: int,
            activation_func=expit,
            alpha: float = .5
            ):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.alpha = alpha
        self.activation_func = activation_func
        self._init()

    def _init(self):
        self.in_hid_weights = np.random.uniform(-.5, .5, (self.hidden_nodes, self.input_nodes))
        self.hid_out_weights = np.random.uniform(-.5, .5, (self.output_nodes, self.hidden_nodes))

    def calculate(self, input: np.ndarray):
        hid_in = self.in_hid_weights @ input
        hid_out = self.activation_func(hid_in)
        out_in = self.hid_out_weights @ hid_out
        out_out = self.activation_func(out_in)
        return out_out, hid_out

    def update_weights(self, input_image: np.ndarray, label: np.ndarray):
        out, hid_out = self.calculate(input_image)
        out_epsilon = label - out # FIXME label I guess should be vector
        hid_epsilon = self.hid_out_weights @ out_epsilon # FIXME dimensions are wrong
        self.hid_out_weights += self.alpha * (out_epsilon * out * (1 - out)) @ hid_out
        self.in_hid_weights += self.alpha * (hid_epsilon * hid_out * (1 - hid_out)) @ input_image

    def train(self, training_data: List):
        for input_image, label in training_data:
            self.update_weights(input_image, label)

    def validate(self, validation_data: List):
        results = []
        for validation_image, label in validation_data:
            results.append(self.validate_image(validation_image, label))
        results = np.array(results, dtype=int)
        print(f'Accuracy {np.mean(results) * 100}%')

    def validate_image(self, validation_image: np.ndarray, label: np.ndarray):
        return label == np.argmax(self.calculate(validation_image))

def data_loader(fname):
    with open(fname, 'r') as file:
        data = list(csv.reader(file))
        data = np.array(data, dtype=int)
    return data

if __name__ == '__main__':
    train_data_path = '../data/mnist_train.csv'
    test_data_path = '../data/mnist_test.csv'

    raw_train_data = data_loader(train_data_path)
    raw_test_data = data_loader(test_data_path)

    train_data = [(row[1:]/255, row[0]) for row in raw_train_data]
    test_data = [(row[1:]/255, row[0]) for row in raw_test_data]

    in_nodes = train_data[0][0].size
    hid_nodes = 100
    out_nodes = 10
    alpha = 0.5

    nn = NetworkMNIST(in_nodes, hid_nodes, out_nodes, alpha=alpha)
    nn.train(train_data)
    nn.validate(test_data)
