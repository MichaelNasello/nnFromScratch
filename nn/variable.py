"""
File defines the Variable class.
"""

import numpy as np


class Variable:
    """
    Builds a model parameter that can be learned via Backpropagation.
    """

    def __init__(self, array, trainable=True):
        """
        Receives a pre-initialized array.
        """

        self.value = array
        self.trainable = trainable
        self.shape = self.value.shape

        # Stored for Backpropagation
        self.grad = np.zeros_like(self.value)

        # Stored for Momentum
        self.velocity = np.zeros_like(self.value)

        # Stored for the Adam Optimizer (if used)
        self.m = np.zeros_like(self.value)
        self.v = np.zeros_like(self.value)
