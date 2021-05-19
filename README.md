<h1 align = "center"> LowLevel-NeuralNet: A Neural-Network inference accelerator library</h1>

[![GitHub Workflow Status][workflow-shield]][workflow-url]
[![MIT License][license-shield]][license-url]
![Python - Version][python-shield]
[![Linting][lint-shield]][lint-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<p align="center">
    <img src="./.github/markdown/speed_test.png" alt="Control panel page" width="80%" style="border-radius: 5px;">
</p>

The LowLevel-NeuralNet library aims at dissociating the implementation of mathematical operations and computation of the result. This separation allows us to use the simple Python syntax to write equations while having access to the C++ language's high computational efficiency. Both characteristics are critical in Neural Networks' world due to the massive computational tasks at hand and the complexity of the equations used. This structure applies to most state-of-the-art libraries like TensorFlow, PyTorch, and many others.

This library is more of an academic exercise than a ready-to-use library as other tools are way more efficient and offer more features. However, intense attention has been given to all functions' test by comparing their results with the equivalent implementation using Numpy.

## Installation

As stated above, this library should mainly be used for experimentation. The installation process aims at creating a development environment.

1. Ensure that you have Python 3.6 or above currently running:
   
    `python --version`

2. Git clone this repository:
   
   `git clone https://github.com/ArthurFDLR/LowLevel_NeuralNet`

3. Create a new virtual environment:
   
   `python -m venv venv`

4. Install development dependencies:
   
   `make dev-env`

5. Build the C++ side of the library:
   
   `make build-lib`

6. You should now be able to run the test suite:

    `make test`

_Note:_ Remember that you will have to rebuild the library after any modification!

## How to use

Here is a simple example of a Convolutional Neural Network pre-trained on the MNIST dataset - see [./example.py](./example.py) for the full code.

```python
# Define layers of the model
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
mnist_test = np.load("tests/data/mnist_test.npz")
images = mnist_test["images"][:1000]
labels_predicted = CNN_model(images=images).argmax(axis=1)
```

[license-shield]: https://img.shields.io/github/license/ArthurFDLR/LowLevel_NeuralNet?style=for-the-badge
[license-url]: https://github.com/ArthurFDLR/LowLevel_NeuralNet/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/arthurfdlr/
[workflow-shield]: https://img.shields.io/github/workflow/status/ArthurFDLR/LowLevel_NeuralNet/LowLevel_NeuralNetwork?style=for-the-badge&label=build%20%26%20test
[workflow-url]: https://github.com/ArthurFDLR/LowLevel_NeuralNet/actions
[python-shield]: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue?style=for-the-badge
[lint-shield]: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
[lint-url]: https://github.com/psf/black