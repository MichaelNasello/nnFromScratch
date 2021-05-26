"""
File defines some common Neural Network layers.

The upstream gradient for layer n is the derivative of the loss with respect to the output of layer n+1.
The downstream gradient for  Layer n is the derivative of the loss with respect to layer n.

The upstream gradient is grabbed from the layer n+1 and is needed to perform the chain rule.
The downstream gradient is passed to the layer n-1 and is needed for that layer to perform chain rule.
"""

import numpy as np

from nn.variable import Variable


class Dense:
    """
    Defines a Dense layer.
    """

    def __init__(self, input_shape, units):
        """
        Assumes input_shape as [batch, activations].
        """

        self.input_shape = input_shape
        self.units = units

        # Stored for gradient calculation
        self.input = None
        self.upstream_grad = None
        self.downstream_grad = None

        # Glorot Uniform Initialization
        sd = np.sqrt(6 / (input_shape[1] + units))
        w_init = np.random.uniform(-sd, sd, (self.input_shape[1], units))
        b_init = np.random.uniform(-sd, sd, (1, units))

        # Defining layer parameters
        self.w = Variable(w_init)
        self.b = Variable(b_init)

    def __call__(self, x):
        self.input = x
        return self.input @ self.w.value + self.b.value

    def backwards(self, upstream_grad):
        """
        Downstream gradient: d_loss/d_out_n_minus_one = upstream_grad x d_out_n/d_out_n_minus_one
        Weight gradient: d_loss/d_W = d_out_n/dW x upstream_grad
        Bias gradient: d_loss/d_b = d_out_n/db x upstream_grad
        """

        self.downstream_grad = upstream_grad @ np.transpose(self.w.value)

        w_grad = np.transpose(self.input) @ upstream_grad
        b_grad = upstream_grad

        # We take an average gradient over the batch
        self.w.grad = np.sum(w_grad, axis=0) / w_grad.shape[0]
        self.b.grad = np.sum(b_grad, axis=0) / b_grad.shape[0]

        return self.downstream_grad


class Convolution2D:
    """
    Defines a Convolutional layer.
    """

    def __init__(self, input_shape, out_filters, kernel_size=3, stride=1, padding="valid"):
        """
        Incoming tensors are expected to be of shape [batch, x, y, filter].
        """

        self.input_shape = input_shape
        self.in_filters = self.input_shape[3]
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Set padding value
        if self.padding == "valid":
            self.p = 0
        elif self.padding == "same":
            # TODO: same padding
            raise NotImplementedError
        elif type(self.padding) == int:
            self.p = self.padding

        # Stored for gradient calculation
        self.input = None
        self.input_padded = None
        self.upstream_grad = None
        self.downstream_grad = None

        # Glorot Uniform Initialization
        sd = np.sqrt(6 / (np.square(self.kernel_size) * (self.in_filters + self.out_filters)))
        w_init = np.random.uniform(
            -sd,
            sd,
            size=(
                self.in_filters,
                self.kernel_size,
                self.kernel_size,
                self.out_filters
            )
        )
        b_init = np.zeros(shape=self.out_filters)

        self.w = Variable(w_init)
        self.b = Variable(b_init)

    def __call__(self, x):
        self.input = x
        self.input_padded = np.pad(x, ((0,), (self.p,), (self.p,), (0,)), "constant")

        # Output has shape [batch, n_w, n_h, out_filters]
        n_w = (x.shape[1] + 2 * self.p - self.kernel_size) // self.stride + 1
        n_h = (x.shape[2] + 2 * self.p - self.kernel_size) // self.stride + 1
        output = np.empty(shape=(x.shape[0], n_w, n_h, self.out_filters))

        for batch in range(self.input.shape[0]):
            for in_filter in range(self.in_filters):
                for out_filter in range(self.out_filters):
                    for w in range(n_w):
                        for h in range(n_h):
                            w_range = (self.stride * w, self.stride * w + self.kernel_size)
                            h_range = (self.stride * h, self.stride * h + self.kernel_size)

                            inner_x = self.input_padded[batch, w_range[0]:w_range[1], h_range[0]:h_range[1], in_filter]
                            output[batch, w, h, out_filter] = self.inner_op(
                                inner_x,
                                self.w.value[in_filter, :, :, out_filter],
                                self.b.value[out_filter]
                            )

        return output

    @staticmethod
    def inner_op(inner_x, inner_w, inner_b):
        return np.sum(np.multiply(inner_x, inner_w) + inner_b)

    def backwards(self):
        return




class Dropout:
    """
    Defines a Dropout layer.
    """

    def __init__(self, drop_rate):
        self.downstream_grad = None
        self.drop_rate = drop_rate

        self.drop_idx = None

    def __call__(self, x, training):
        if training:
            self.drop_idx = np.random.choice([0, 1], size=x.shape, p=[self.drop_rate, 1 - self.drop_rate])
            return np.multiply(self.drop_idx, x)
        else:
            return x

    def backwards(self, upstream_grad):
        return np.multiply(self.drop_idx, upstream_grad)


class BatchNormalization:
    """
    Defines a Batch Normalization layer.
    """

    def __init__(self):
        return

    def __call__(self, x):
        return

    def backwards(self):
        return


class ReLU:
    """
    Defines a Relu activation layer.
    """

    def __init__(self):
        self.downstream_grad = None

    def __call__(self, x):
        self.input = x
        return np.maximum(0, self.input) - 0.5

    def backwards(self, upstream_grad):
        self.downstream_grad = upstream_grad * (self.input > 0)
        return self.downstream_grad


class LeakyRelu:
    """
    Defines a Leaky Relu activation layer.
    """

    def __init__(self):
        return

    def __call__(self, x):
        return

    def backwards(self):
        return


class Tanh:
    """
    Defines a Tanh activation layer.
    """

    def __init__(self):
        return

    def __call__(self, x):
        return

    def backwards(self):
        return


class Sigmoid:
    """
    Defines a Sigmoid activation layer.
    """

    def __call__(self, x):
        return

    def backwards(self):
        return
