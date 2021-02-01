import numpy as np
import LowLevel_NeuralNet as llnn
import numpy_reference
import time

if __name__ == "__main__":

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

    x = llnn.Input2d("images", 28, 28, 1)
    for layer in layers:
        x = layer(x)
    
    x.resolve(np.load("tests/data/mnist_params.npz"))
    mnist_test = np.load("tests/data/mnist_test.npz")
    images = mnist_test["images"][:10000]

    ref_time = time.time()
    ref_label = x.compile(numpy_reference.Builder())(images=images).argmax(axis=1)
    ref_time = time.time() - ref_time

    llnn_time = time.time()
    llnn_label = x.compile(llnn.Builder())(images=images).argmax(axis=1)
    llnn_time = time.time() - llnn_time
    
    assert np.allclose(ref_label, llnn_label)

    print(f"\nNumpy reference: {ref_time} seconds\nllnn: {llnn_time} seconds ({100.*(ref_time-llnn_time)/ref_time:.2f}% faster)")