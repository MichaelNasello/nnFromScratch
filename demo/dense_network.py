from data.dataset import *
from nn.layers import *
from nn.losses import *
from nn.model import Model
from nn.optimizers import *
from nn.scheduler import HyperParamScheduler


if __name__ == "__main__":

    data_path = pathlib.Path.home() / "Desktop" / "MNIST" / "numpy"
    x = np.load(str(data_path / "x.npy"))
    y = np.load(str(data_path / "y.npy"))

    x = x.astype(np.int)
    y = y.astype(np.int)

    # One hot encode y
    y = np.eye(10)[y]

    dataset = Dataset(x, y, split=(0.7, 0.2, 0.1), normalize=True)

    # Reshaping input
    num_train_samples = dataset.x_train.shape[0]
    num_val_samples = dataset.x_val.shape[0]
    num_test_samples = dataset.x_test.shape[0]
    dataset.x_train = np.reshape(dataset.x_train, (num_train_samples, -1))
    dataset.x_val = np.reshape(dataset.x_val, (num_val_samples, -1))
    dataset.x_test = np.reshape(dataset.x_test, (num_test_samples, -1))

    # Initializing layers
    batch_size = 32
    dense1 = Dense((batch_size, dataset.x_train.shape[1]), 200)
    relu1 = ReLU()
    dense2 = Dense((batch_size, 200), 100)
    relu2 = ReLU()
    dense3 = Dense((batch_size, 100), 50)
    relu3 = ReLU()
    dense4 = Dense((batch_size, 50), 10)

    loss_f = CrossEntropy()
    optimizer = Adam()

    # Initializing model
    model = Model(
        input_shape=(batch_size, dataset.x_train.shape[1]),
        layers=[dense1, relu1, dense2, relu2, dense3, relu3, dense4],
        loss_f=loss_f,
        optimizer=optimizer
    )

    # Create hyper-param schedulers
    lr_schedule = HyperParamScheduler(3e-4, 3e-3, True)
    mom_schedule = HyperParamScheduler(0.9, 0.99, False)

    # Train
    model.fit_one_cycle(
        dataset.x_train,
        dataset.y_train,
        50,
        lr_schedule,
        mom_schedule,
        batch_size,
        [dataset.x_val, dataset.y_val]
    )

    # Save model
    save_path = pathlib.Path.home() / "Desktop" / "MNIST_model"
    model.save_model_weights(save_path)

    # Load model and test
    model.load_model_weights(save_path)
    pred = np.argmax(model.predict(dataset.x_test[0]))
    act = np.argmax(dataset.y_test[0])

    print(f"\nPrediction: {pred}, Actual: {act}")
