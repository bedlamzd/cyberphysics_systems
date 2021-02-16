import csv
from typing import List

import numpy as np
import matplotlib.pyplot as plt
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
        self._init_weights()

    def _init_weights(self):
        self.in_hid_weights = np.random.uniform(-.5, .5, (self.hidden_nodes, self.input_nodes))
        self.hid_out_weights = np.random.uniform(-.5, .5, (self.output_nodes, self.hidden_nodes))

    def calculate(self, input: np.ndarray):
        hid_in = self.in_hid_weights @ input
        hid_out = self.activation_func(hid_in)
        out_in = self.hid_out_weights @ hid_out
        out_out = self.activation_func(out_in)
        return out_out, hid_out

    def train(self, training_data: List):
        for input_image, label in training_data:
            self.update_weights(input_image, label)

    def update_weights(self, input_image: np.ndarray, label: np.ndarray):
        out, hid_out = self.calculate(input_image)
        out_epsilon = label - out
        hid_epsilon = self.hid_out_weights.T @ out_epsilon
        self.hid_out_weights += self.alpha * (out_epsilon * out * (1 - out)) @ hid_out.T
        self.in_hid_weights += self.alpha * (hid_epsilon * hid_out * (1 - hid_out)) @ input_image.T

    def validate(self, validation_data: List):
        results = []
        for validation_image, label in validation_data:
            results.append(self.validate_image(validation_image, label))
        results = np.array(results, dtype=int)
        print(f'Accuracy {np.mean(results) * 100}%')

    def validate_image(self, validation_image: np.ndarray, label: np.ndarray):
        return np.argmax(label) == np.argmax(self.calculate(validation_image)[0])


def data_loader(fname):
    with open(fname, 'r') as file:
        data = list(csv.reader(file))
        data = np.array(data, dtype=int)
    return data


def data_preprocessor(raw_data: np.ndarray):
    processed_data = []
    for row in raw_data:
        image = np.array(row[1:] / 255, ndmin=2).T
        label_vector = np.insert(np.zeros(9), row[0], 1).reshape((10, 1))
        processed_data.append((image, label_vector))
    return processed_data


def get_params(*, default: bool = False):
    DEFAULT_INPUT_NODES = 784
    DEFAULT_HIDDEN_NODES = 100
    DEFAULT_OUTPUT_NODES = 10
    DEFAULT_LEARNING_RATE = 0.5

    if default:
        in_nodes = DEFAULT_INPUT_NODES
        hid_nodes = DEFAULT_HIDDEN_NODES
        out_nodes = DEFAULT_OUTPUT_NODES
        alpha = DEFAULT_LEARNING_RATE
    else:
        in_nodes = int(input('Enter number of nodes in input layer: ') or DEFAULT_INPUT_NODES)
        hid_nodes = int(input('Enter number of nodes in hidden layer: ') or DEFAULT_HIDDEN_NODES)
        out_nodes = int(input('Enter number of output nodes: ') or DEFAULT_OUTPUT_NODES)
        alpha = float(input('Enter learning rate: ') or DEFAULT_LEARNING_RATE)

    return in_nodes, hid_nodes, out_nodes, alpha


def display_and_save_image(image, label):
    f = plt.figure()
    plt.imshow(image.reshape((28, 28)), cmap='gray')
    f.suptitle(f'Image of digit "{np.argmax(label)}"')
    f.savefig(img_path + 'random_image.png')


if __name__ == '__main__':
    np.random.seed(42)  # to get consistent results

    VARIANT = 6

    img_path = '../img/'

    train_data_path = '../data/mnist_train.csv'
    test_data_path = '../data/mnist_test.csv'

    # load and prepare data. define parameters
    raw_train_data = data_loader(train_data_path)
    raw_test_data = data_loader(test_data_path)

    train_data = data_preprocessor(raw_train_data)  # [ (image, label_vector), ... ]
    test_data = data_preprocessor(raw_test_data)  # [ (image, label_vector), ... ]

    in_nodes, hid_nodes, out_nodes, alpha = get_params(default=True)

    # initialize net and feed data to it
    nn = NetworkMNIST(in_nodes, hid_nodes, out_nodes, alpha=alpha)
    nn.train(train_data)
    nn.validate(test_data)

    # a couple more training iterations
    for _ in range(3):
        print(f'Additional training iteration #{_ + 1}')
        nn.train(train_data)
        nn.validate(test_data)

    # display image from set
    image, label = test_data[VARIANT]
    display_and_save_image(image, label)
