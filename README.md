# LowLevel-NeuralNet: A Neural-Network inference accelerator library

[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ArthurFDLR/LowLevel_NeuralNet/LowLevel_NeuralNetwork?label=build%20%26%20test)](https://github.com/ArthurFDLR/LowLevel_NeuralNet/actions)
![Python - Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
[![GitHub](https://img.shields.io/github/license/ArthurFDLR/LowLevel_NeuralNet)](https://github.com/ArthurFDLR/LowLevel_NeuralNet/blob/master/LICENSE.txt)
[![Linting](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
