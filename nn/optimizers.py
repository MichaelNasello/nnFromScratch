"""
File defines some common optimizers.
"""

import numpy as np


class SGD:
    """
    Defines the Stochastic Gradient Descent optimizer.
    """

    @staticmethod
    def step(model, l_r, wd, momentum):
        """
        Updates model parameters with their gradients.
        """

        for param in model.parameters:
            if param.trainable:

                # Update velocity of gradient
                param.velocity = momentum * param.velocity - l_r * param.grad - l_r * wd * param.grad
                param.value += param.velocity

    @staticmethod
    def zero_grad(model):
        """
        Sets all trainable parameter gradients to zero.
        """

        for param in model.parameters:
            if param.trainable:
                param.grad = 0


class Adam:
    """
    Defines the Adam optimizer for faster convergence compared to SGD.
    """

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, model, l_r, wd, t):
        """
        Updates model parameters with their gradients.
        """

        for param in model.parameters:
            if param.trainable:

                # Momentum with beta1
                param.m = self.beta1 * param.m + (1 - self.beta1) * param.grad

                # RMS with beta2
                param.v = self.beta2 * param.v + (1 - self.beta2) * np.square(param.grad)

                # Bias correction
                m_debiased = param.m / (1 - self.beta1**t)
                v_debiased = param.v / (1 - self.beta2**t)

                # Param update (moving av of grad and weight decay)
                param.value = param.value - l_r * (m_debiased / np.sqrt(v_debiased + self.eps)) #- 2 * l_r * wd * param.value

    @staticmethod
    def zero_grad(model):
        """
        Sets all trainable parameter gradients to zero.
        """

        for param in model.parameters:
            if param.trainable:
                param.grad = 0
