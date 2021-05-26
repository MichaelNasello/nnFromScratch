"""
File defines some common Neural Network loss functions.
"""

import numpy as np


class MSE:
    """
    Defines the Mean Squared Error loss function.
    """

    def __init__(self):
        self.downstream_grad = None

    def __call__(self, y, y_hat):
        self.input = y
        self.y_hat = y_hat

        return np.mean(np.square(y - y_hat))

    def backwards(self):
        self.downstream_grad = 2 * (self.input - self.y_hat)


class CrossEntropy:
    """
    Defines the Cross Entropy loss function.
    """

    def __init__(self):
        self.downstream_grad = None

    def __call__(self, y, y_hat):
        self.input = y
        self.y_hat = y_hat

        return self.neg_log_likelihood(self.softmax(y), y_hat)

    @staticmethod
    def softmax(y):
        e = np.exp(y)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def neg_log_likelihood(y_soft, y_hat):
        neg_log_likelihood = -y_hat * np.log(y_soft)
        return np.mean(np.sum(neg_log_likelihood, axis=1))

    def backwards(self):
        self.downstream_grad = self.softmax(self.input) - self.y_hat
        return self.downstream_grad
