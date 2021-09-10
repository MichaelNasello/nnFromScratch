"""
File defines the Model class.
"""

import json
import numpy as np
import pathlib

import nn.layers
import nn.optimizers


class Model:
    """
    The Model class encapsulates all model details.
    """

    def __init__(self, input_shape, layers, loss_f, optimizer):
        self.input_shape = input_shape
        self.layers = layers
        self.loss_f = loss_f
        self.optimizer = optimizer

        if type(self.optimizer) == nn.optimizers.SGD:
            self.opt_type = "sgd"
        elif type(self.optimizer) == nn.optimizers.Adam:
            self.opt_type = "adam"

        self.training = False

        # Build list of parameters and model summary
        model_desc = []
        self.parameters = []
        for layer in self.layers:
            layer_desc = [type(layer).__name__]
            if hasattr(layer, "w"):
                self.parameters.append(layer.w)
                layer_desc.append(["Weight", list(layer.w.shape)])
            if hasattr(layer, "b"):
                self.parameters.append(layer.b)
                layer_desc.append(["Bias", list(layer.b.shape)])

            model_desc.append(layer_desc)

        self.model_summary = {
            "Input": [1, self.input_shape[1]],
            "Layers": model_desc,
        }

    def forwards(self, x):
        """
        Forward pass through the model.
        """

        for layer in self.layers:
            if type(layer) in [nn.layers.Dropout]:
                x = layer(x, training=self.training)
            else:
                x = layer(x)

        return x

    def backwards(self):
        """
        Backward pass through the model. Parameter gradients computed.
        """

        downstream_grad = self.loss_f.backwards()
        for r_layer in reversed(self.layers):
            downstream_grad = r_layer.backwards(downstream_grad)

    @staticmethod
    def get_batch(x, y, i, batch_size):
        """
        Returns batch_size sample of dataset.
        """
        x_b = x[i * batch_size: i * batch_size + batch_size]
        y_b = y[i * batch_size: i * batch_size + batch_size]
        return x_b, y_b

    def reset_param_states(self):
        """
        Resets all param states.
        """

        for param in self.parameters:
            param.grad = np.zeros_like(param.value)
            param.velocity = np.zeros_like(param.value)
            param.m = np.zeros_like(param.value)
            param.v = np.zeros_like(param.value)

    def sum_of_weights(self):
        """
        Computes sum of weights, needed for the addition to loss computation for weight decay.
        """

        weight_sum = 0
        for param in self.parameters:
            weight_sum += np.sum(np.square(param.value))

        return weight_sum

    def fit_one_cycle(self, x, y, n_epochs, lr_schedule, mom_schedule, batch_size, validation_set):
        """
        Trains model for n_epochs on (x, y). Evaluate on validation_set.

        Learning rates and momentum are scheduled using the 1cycle policy.
        """
        print("Beginning train...\n")

        num_batches = x.shape[0] // batch_size
        num_val_batches = len(validation_set[0]) // batch_size

        # Reset param states when starting a training session
        self.reset_param_states()

        # Compute scheduler slopes
        lr_schedule.set_slope(num_batches)
        if self.opt_type == "sgd":
            mom_schedule.set_slope(num_batches)

        for epoch in range(n_epochs):

            losses = []
            val_losses = []
            val_accuracies = []

            for batch in range(num_batches):
                x_b, y_b = self.get_batch(x, y, batch, batch_size)

                # Get model output and loss
                self.training = True
                pred = self.forwards(x_b)
                loss = self.loss_f(pred, y_b)
                losses.append(loss)

                # Backwards pass and parameter updates
                self.backwards()
                lr = lr_schedule(batch)
                if self.opt_type == "sgd":
                    mom = mom_schedule(batch)
                    self.optimizer.step(self, lr, mom)
                elif self.opt_type == "adam":
                    self.optimizer.step(self, lr, batch + 1)
                self.optimizer.zero_grad(self)

            # Computing validation loss, accuracy
            for val_batch in range(num_val_batches):
                x_bv, y_bv = self.get_batch(validation_set[0], validation_set[1], val_batch, batch_size)

                # Get model output and loss
                self.training = False
                pred_v = self.forwards(x_bv)
                loss_v = self.loss_f(pred_v, y_bv)

                # Get performance results
                mean_acc = np.mean(np.argmax(pred_v, axis=1) == np.argmax(y_bv, axis=1))
                val_accuracies.append(mean_acc)
                val_losses.append(loss_v)

            print(
                f"Epoch: {epoch + 1:2}/{n_epochs} -> Loss: {np.mean(losses):.3f} Val Loss: {np.mean(val_losses):.3f}, "
                f"Val Acc: {np.mean(val_accuracies):.3f}"
            )

        print("Done training.")

    def predict(self, x):
        """
        Generates predictions.
        """

        self.training = False
        for layer in self.layers:
            if type(layer) in [nn.layers.Dropout]:
                x = layer(x, training=self.training)
            else:
                x = layer(x)

        return x

    def save_model_weights(self, path):
        """
        Saves a model.
        """

        if not path.exists():
            pathlib.Path.mkdir(path)
            pathlib.Path.mkdir(path / "params")

        # Dump model description
        with open(path / "model.json", "w") as f:
            json.dump(self.model_summary, f)

        # Dump params
        for i, param in enumerate(self.parameters):
            np.save(path / "params" / f"param{i}.npy", param.value)

    def load_model_weights(self, path):
        """
        Loads a model.
        """

        if not path.exists():
            raise Exception("Invalid model path.")
        if not (path / "params").exists():
            raise Exception("Param values are missing.")
        if not (path / "model.json").exists():
            raise Exception("Model description file missing.")

        # Load json
        with open(path / "model.json") as f:
            load_dict = json.load(f)
        if load_dict != self.model_summary:
            raise Exception("Models are not matching, cannot load parameters.")

        for i, param in enumerate(self.parameters):
            param.value = np.load(path / "params" / f"param{i}.npy")
