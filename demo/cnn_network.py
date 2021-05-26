from data.dataset import *
from nn.layers import *
from nn.losses import *
from nn.model import Model
from nn.optimizers import *
from nn.scheduler import HyperParamScheduler


if __name__ == "__main__":

    data_path = pathlib.Path.home() / "Desktop" / "MNIST" / "numpy"
    x = np.load(data_path / "x.npy")
    y = np.load(data_path / "y.npy")

    x = x.astype(np.int)
    y = y.astype(np.int)

    # Add filter axis to x
    x = x[:, :, :, np.newaxis]

    # One hot encode y
    y = np.eye(10)[y]

    dataset = Dataset(x, y, split=(0.7, 0.2, 0.1), normalize=True)

    num_train_samples = dataset.x_train.shape[0]
    num_val_samples = dataset.x_val.shape[0]
    num_test_samples = dataset.x_test.shape[0]

    # Initializing layers
    batch_size = 32
    conv1 = Convolution2D(
        input_shape=(batch_size, dataset.x_train.shape[1], dataset.x_train.shape[2], dataset.x_train.shape[3]),
        out_filters=10,
        padding=1
    )
    conv2 = Convolution2D(
        input_shape=(batch_size, dataset.x_train.shape[1], dataset.x_train.shape[2], dataset.x_train.shape[3]),
        out_filters=15,
        padding=1
    )

    loss_f = CrossEntropy()
    optimizer = Adam()

    # Initializing model
    model = Model(
        input_shape=(batch_size, dataset.x_train.shape[1]),
        layers=[conv1, conv2],
        loss_f=loss_f,
        optimizer=optimizer
    )

    out = model.forwards(dataset.x_train[0:32])
    print(out.shape)


    """
    # Create hyper-param schedulers
    lr_schedule = HyperParamScheduler(3e-4, 3e-3, True)
    mom_schedule = HyperParamScheduler(0.8, 0.9, False)

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

    print(f"\nPrediction: {pred}, Actual: {act}")"""