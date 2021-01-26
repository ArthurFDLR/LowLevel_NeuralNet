import LowLevel_NeuralNet as llnn
import numpy as np

if __name__ == "__main__":
    # Define layers composing the model
    layers = [
        llnn.Conv2d("c1", 1, 8, 3),
        llnn.ReLU(),
        llnn.Conv2d("c2", 8, 8, 3),
        llnn.ReLU(),
        llnn.MaxPool2d(2, 2),
        llnn.ReLU(),
        llnn.Conv2d("c3", 8, 16, 3),
        llnn.ReLU(),
        llnn.Conv2d("c4", 16, 16, 3),
        llnn.ReLU(),
        llnn.MaxPool2d(2, 2),
        llnn.Flatten(),
        llnn.Linear("f", 16 * 4 * 4, 10),
    ]

    # Build the model
    x = llnn.Input2d("images", 28, 28, 1)
    for layer in layers:
        x = layer(x)

    # Import model parameters
    x.resolve(np.load("tests/data/mnist_params.npz"))
    # Compile the model
    CNN_model = x.compile(llnn.Builder())

    # Evaluation of the pre-trained model
    test_size = 1000
    mnist_test = np.load("tests/data/mnist_test.npz")
    images_test = mnist_test["images"][:test_size]
    labels_test = mnist_test["labels"][:test_size]
    labels_predicted = CNN_model(images=images_test).argmax(axis=1)
    correctly_predicted = np.count_nonzero(np.equal(labels_test, labels_predicted))

    print(
        "Model accuracy: {}% ({}/{})".format(
            100.0 * correctly_predicted / test_size, correctly_predicted, test_size
        )
    )
