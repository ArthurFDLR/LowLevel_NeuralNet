import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model
import numpy as np
import LowLevel_NeuralNet as llnn
import numpy_reference
import time
import csv

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

    with open('speed_results.csv', mode='w') as speed_file:
        speed_writer = csv.writer(speed_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        speed_writer.writerow(['', 'Numpy', 'TensorFlow', 'llnn'])

        for test_size in range(1, 65):
            print(test_size)
            images = mnist_test["images"][:test_size]
            labels = mnist_test["labels"][:test_size]

            np_time_sum = 0
            llnn_time_sum = 0
            tf_time_sum = 0
            N=10
            for i in range(N): # Mean computation time
                np_time = time.time()
                np_label = x.compile(numpy_reference.Builder())(images=images)
                np_time_sum += time.time() - np_time

                llnn_time = time.time()
                llnn_label = x.compile(llnn.Builder())(images=images)
                llnn_time_sum = time.time() - llnn_time

                tf_model = load_model("tests/data/mnist_params.h5")
                tf_time = time.time()
                tf_label = tf_model.predict(images)
                tf_time_sum = time.time() - tf_time

            np_time_sum /= N
            llnn_time_sum /= N
            tf_time_sum /= N

            speed_writer.writerow([test_size, np_time_sum, tf_time_sum, llnn_time_sum])

    print("Test results saved in ./speed_results.csv")