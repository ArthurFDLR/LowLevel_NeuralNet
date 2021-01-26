import pytest
import numpy as np
import os

import LowLevel_NeuralNet as llnn
import numpy_reference

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)


def random_kwargs(kwargs):
    return {
        k: np.random.random(shape) if shape != None else np.random.random()
        for k, shape in kwargs.items()
    }


def is_same(p, n, **kwargs):
    e0 = p.compile(numpy_reference.Builder())
    e1 = p.compile(llnn.Builder())
    nkwargs = [random_kwargs(kwargs) for i in range(n)]
    return all([np.allclose(e0(**nkwargs[i]), e1(**nkwargs[i])) for i in range(n)])


## Scalar operations


def test_scalar_input():
    x = llnn.Input("x")
    assert is_same(x, 1, x=None)


def test_scalar_const():
    c = llnn.Const(np.random.random())
    assert is_same(c, 1)


def test_scalar_sum_mix():
    x = llnn.Input("x")
    c = llnn.Const(np.random.random())
    y = x + c
    assert is_same(y, 1, x=None)


def test_scalar_mul_mix():
    x = llnn.Input("x")
    c = llnn.Const(np.random.random())
    y = x * c
    assert is_same(y, 1, x=None)


def test_scalar_sum_input():
    x = llnn.Input("x")
    y = llnn.Input("y")
    z = x + y
    assert is_same(z, 1, x=None, y=None)


def test_scalar_mul_input():
    x = llnn.Input("x")
    y = llnn.Input("y")
    z = x * y
    assert is_same(z, 1, x=None, y=None)


def test_scalar_sum_mul_input():
    x = llnn.Input("x")
    y = llnn.Input("y")
    z = llnn.Input("z")
    r = x + y * z
    assert is_same(r, 1, x=None, y=None, z=None)


def test_scalar_sub_mul_input():
    x = llnn.Input("x")
    y = llnn.Input("y")
    z = llnn.Input("z")
    r = (x - y) * z
    assert is_same(r, 1, x=None, y=None, z=None)


def test_scalar_sub_mul_sum_mix():
    x = llnn.Input("x")
    y = llnn.Input("y")
    c = llnn.Const(np.random.random())
    z = x * y - x * c - y * c + c * c
    assert is_same(z, 10, x=None, y=None)


## Vectorial operations


def test_vect_input():
    x = llnn.Input("x")
    assert is_same(x, 1, x=(9,)) and is_same(x, 1, x=(9, 9))


def test_vect_const():
    c1 = llnn.Const(np.random.random((10,)))
    c2 = llnn.Const(np.random.random((10, 10)))
    assert is_same(c1, 1) and is_same(c2, 1)


def test_vect_sum_input():
    x = llnn.Input("x")
    y = llnn.Input("y")
    z = x + y
    assert all(
        [
            is_same(z, 1, x=(11,), y=(11,)),
            is_same(z, 1, x=(11, 12), y=(11, 12)),
            is_same(z, 1, x=(11, 12, 13), y=(11, 12, 13)),
            is_same(z, 1, x=(11, 12, 13, 14), y=(11, 12, 13, 14)),
        ]
    )


def test_vect_sub_input():
    x = llnn.Input("x")
    y = llnn.Input("y")
    z = x - y
    assert all(
        [
            is_same(z, 1, x=(11,), y=(11,)),
            is_same(z, 1, x=(11, 12), y=(11, 12)),
            is_same(z, 1, x=(11, 12, 13), y=(11, 12, 13)),
            is_same(z, 1, x=(11, 12, 13, 14), y=(11, 12, 13, 14)),
        ]
    )


def test_vect_mul_input():
    x = llnn.Input("x")
    y = llnn.Input("y")
    z = x * y
    assert is_same(z, 1, x=(11, 12), y=(12, 13))


def test_vect_mul_input_null():
    x = llnn.Input("x")
    y = llnn.Input("y")
    z = x * y
    assert is_same(z, 1, x=None, y=(12, 13)) and is_same(z, 1, x=(11, 12), y=None)


def test_vect_sum_mul_input():
    x = llnn.Input("x")
    y = llnn.Input("y")
    z = llnn.Input("z")
    r = x + y * z
    assert is_same(r, 1, x=(11, 13), y=(11, 12), z=(12, 13))


def test_vect_sub_mul_input():
    x = llnn.Input("x")
    y = llnn.Input("y")
    z = llnn.Input("z")
    r = (x - y) * z
    assert is_same(r, 1, x=(11, 12), y=(11, 12), z=(12, 13))


def test_vect_mulmul_input():
    x = llnn.Input("x")
    y = llnn.Input("y")
    z = llnn.Input("z")
    r = x * y * z
    assert is_same(r, 1, x=(11, 12), y=(12, 13), z=(13, 14))


def test_vect_sub_mul_sum_mix():
    x = llnn.Input("x")
    y = llnn.Input("y")
    c1 = llnn.Const(np.random.random((11, 12)))
    c2 = llnn.Const(np.random.random((12, 13)))
    z = x * y - x * c2 - c1 * y + c1 * c2
    assert is_same(z, 1, x=(11, 12), y=(12, 13))


## Neural Networks operations


def test_relu():
    relu = llnn.ReLU()
    x = relu(llnn.Input("x"))
    assert is_same(x, 1, x=(10, 11, 12, 13))


def test_flatten():
    flatten = llnn.Flatten()
    x = flatten(llnn.Input("x"))
    assert is_same(x, 1, x=(10, 11, 12, 13))


def test_input2d():
    x = llnn.Input2d("images", 10, 11, 3)
    assert is_same(x, 1, images=(50, 10, 11, 3))


def test_linear():
    f = llnn.Linear("f", 100, 10)
    x = f(llnn.Input("x"))
    x.resolve(
        {"f.weight": np.random.random((10, 100)), "f.bias": np.random.random((10,))}
    )
    assert is_same(x, 1, x=(50, 100))


def test_maxpool():
    pool = llnn.MaxPool2d(3, 3)
    x = pool(llnn.Input2d("x", 12, 15, 3))
    assert is_same(x, 1, x=(10, 12, 15, 3))


def test_convolution():
    c = llnn.Conv2d("c", 3, 16, 5)
    x = c(llnn.Input2d("x", 15, 20, 3))
    x.resolve(
        {"c.weight": np.random.random((16, 3, 5, 5)), "c.bias": np.random.random((16,))}
    )
    assert is_same(x, 1, x=(10, 15, 20, 3))


def test_ANN():
    relu = llnn.ReLU()
    flatten = llnn.Flatten()
    f1 = llnn.Linear("f1", 28 * 28, 100)
    f2 = llnn.Linear("f2", 100, 10)
    x = llnn.Input2d("images", 28, 28, 1)
    x = flatten(x)
    x = f2(relu(f1(x)))
    x.resolve(np.load("tests/data/msimple_params.npz"))
    mnist_test = np.load("tests/data/mnist_test.npz")
    images = mnist_test["images"][:20]

    infer0 = x.compile(numpy_reference.Builder())
    infer1 = x.compile(llnn.Builder())
    logit0 = infer0(images=images)
    logit1 = infer1(images=images)
    assert np.allclose(logit0, logit1)


def test_ANN_large():
    relu = llnn.ReLU()
    flatten = llnn.Flatten()
    f1 = llnn.Linear("f1", 28 * 28, 100)
    f2 = llnn.Linear("f2", 100, 10)
    x = llnn.Input2d("images", 28, 28, 1)
    x = flatten(x)
    x = f2(relu(f1(x)))
    x.resolve(np.load("tests/data/msimple_params.npz"))
    mnist_test = np.load("tests/data/mnist_test.npz")
    images = mnist_test["images"][:1000]

    infer0 = x.compile(numpy_reference.Builder())
    infer1 = x.compile(llnn.Builder())
    label0 = infer0(images=images).argmax(axis=1)
    label1 = infer1(images=images).argmax(axis=1)
    assert np.allclose(label0, label1)


def test_CNN():
    pool = llnn.MaxPool2d(2, 2)
    relu = llnn.ReLU()
    flatten = llnn.Flatten()

    x = llnn.Input2d("images", 28, 28, 1)
    c1 = llnn.Conv2d("c1", 1, 8, 3)  # 28->26
    c2 = llnn.Conv2d("c2", 8, 8, 3)  # 26->24
    x = pool(relu(c2(relu(c1(x)))))  # 24->12
    c3 = llnn.Conv2d("c3", 8, 16, 3)  # 12->10
    c4 = llnn.Conv2d("c4", 16, 16, 3)  # 10->8
    x = pool(relu(c4(relu(c3(x)))))  # 8->4
    f = llnn.Linear("f", 16 * 4 * 4, 10)
    x = f(flatten(x))

    x.resolve(np.load("tests/data/mnist_params.npz"))
    mnist_test = np.load("tests/data/mnist_test.npz")
    images = mnist_test["images"][:20]

    infer0 = x.compile(numpy_reference.Builder())
    infer1 = x.compile(llnn.Builder())
    logit0 = infer0(images=images)
    logit1 = infer1(images=images)
    assert np.allclose(logit0, logit1)


def test_CNN_large():
    pool = llnn.MaxPool2d(2, 2)
    relu = llnn.ReLU()
    flatten = llnn.Flatten()

    x = llnn.Input2d("images", 28, 28, 1)
    c1 = llnn.Conv2d("c1", 1, 8, 3)  # 28->26
    c2 = llnn.Conv2d("c2", 8, 8, 3)  # 26->24
    x = pool(relu(c2(relu(c1(x)))))  # 24->12
    c3 = llnn.Conv2d("c3", 8, 16, 3)  # 12->10
    c4 = llnn.Conv2d("c4", 16, 16, 3)  # 10->8
    x = pool(relu(c4(relu(c3(x)))))  # 8->4
    f = llnn.Linear("f", 16 * 4 * 4, 10)
    x = f(flatten(x))

    x.resolve(np.load("tests/data/mnist_params.npz"))
    mnist_test = np.load("tests/data/mnist_test.npz")
    # The number of images should be increased on a local machine but
    # keep a reasonable number for Github automatic testing.
    images = mnist_test["images"][:100]

    infer0 = x.compile(numpy_reference.Builder())
    infer1 = x.compile(llnn.Builder())
    label0 = infer0(images=images).argmax(axis=1)
    label1 = infer1(images=images).argmax(axis=1)
    assert np.allclose(label0, label1)
