import numpy as np
import time


def Input(expr, op, args, **kwargs):
    if op.name in kwargs:
        c = kwargs[op.name]
        if isinstance(c, (int, float)):
            return float(c)
        elif hasattr(c, "shape"):
            return c.astype(float)
        else:
            raise Exception("%s: Input must be float or int or ndarray: %s" % (expr, c))
    else:
        raise Exception("%s: missing input" % expr)


def Input2d(expr, op, args, **kwargs):
    if not op.name in kwargs:
        raise Exception("%s: missing input" % expr)
    imgs = kwargs[op.name]
    if not hasattr(imgs, "shape"):
        raise Exception("%s: Input must be ndarray: %s" % (expr, imgs))
    if any(
        [
            len(imgs.shape) != 4,
            imgs.shape[1] != op.parameters["height"],
            imgs.shape[2] != op.parameters["width"],
            imgs.shape[3] != op.parameters["in_channels"],
        ]
    ):
        raise Exception("%s: Invalid input size: %s" % (expr, imgs.shape))
    # NHWC => NCHW
    return imgs.astype(float).transpose(0, 3, 1, 2)


def Const(expr, op, args, **kwargs):
    return op.parameters["value"]


def Neg(expr, op, args, **kwargs):
    return -args[0]


def Add(expr, op, args, **kwargs):
    a = args[0]
    b = args[1]
    if not hasattr(a, "shape") and not hasattr(b, "shape"):
        return a + b
    elif hasattr(a, "shape") and hasattr(b, "shape"):
        if a.shape != b.shape:
            raise Exception("%s: size mismatch: %s+%s" % (expr, a.shape, b.shape))
        return a + b
    else:
        raise Exception("%s: cannot mix scalar and ndarray" % expr)


def Sub(expr, op, args, **kwargs):
    a = args[0]
    b = args[1]
    if not hasattr(a, "shape") and not hasattr(b, "shape"):
        return a - b
    elif hasattr(a, "shape") and hasattr(b, "shape"):
        if a.shape != b.shape:
            raise Exception("%s: size mismatch: %s-%s" % (expr, a.shape, b.shape))
        return a - b
    else:
        raise Exception("%s: cannot mix scalar and ndarray" % expr)


def Mul(expr, op, args, **kwargs):
    a = args[0]
    b = args[1]
    if not hasattr(a, "shape") or not hasattr(b, "shape"):
        return a * b
    else:
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise Exception("%s: matmul only: %s*%s" % (expr, a.shape, b.shape))
        if a.shape[1] != b.shape[0]:
            raise Exception("%s: size mismatch: %s*%s" % (expr, a.shape, b.shape))
        return np.matmul(a, b)


# Custom
def Pow(expr, op, args, **kwargs):
    a = args[0]
    b = args[1]
    if hasattr(b, "shape") or b % 1.0:
        raise Exception("%s: power type mismatch: %s" % (expr, b.shape))
    else:
        if not hasattr(a, "shape"):
            return a ** b
        else:
            if len(a.shape) != 2:
                raise Exception("%s: numpy.dot() only: %s" % (expr, a.shape))
            else:
                return np.linalg.matrix_power(np.array(a), int(b))


def Flatten(expr, op, args, **kwargs):
    x = args[0]
    if not hasattr(x, "shape"):
        raise Exception("%s: ndarray only: %s" % (expr, imgs))
    return x.reshape((x.shape[0], -1))


def ReLU(expr, op, args, **kwargs):
    x = args[0]
    return x * (x > 0)


def Linear(expr, op, args, **kwargs):
    x = args[0]
    if not hasattr(x, "shape"):
        raise Exception("%s: ndarray only: %s" % (expr, x))
    if "weight" not in op.parameters or "bias" not in op.parameters:
        raise Exception("%s: missing weight or bias" % expr)
    weight = op.parameters["weight"]
    bias = op.parameters["bias"]
    if not hasattr(weight, "shape") or not hasattr(bias, "shape"):
        raise Exception("%s: ndarray only for weight or bias" % expr)
    in_features = op.parameters["in_features"]
    out_features = op.parameters["out_features"]
    if any(
        [
            len(x.shape) != 2,
            x.shape[1] != in_features,
            weight.shape != (out_features, in_features),
            bias.shape != (out_features,),
        ]
    ):
        raise Exception(
            "%s: size mismatch: %s*%s+%s" % (expr, weight.shape, x.shape, bias.shape)
        )
    return np.einsum("ni,oi->no", x, weight, optimize="optimal") + bias.reshape(
        (1, out_features)
    )


def MaxPool2d(expr, op, args, **kwargs):
    x = args[0]
    if not hasattr(x, "shape"):
        raise Exception("%s: ndarray only: %s" % (expr, x))
    kernel_size = op.parameters["kernel_size"]
    stride = op.parameters["stride"]
    if kernel_size != stride:
        raise Exception("%s: kernel_size != stride" % expr)
    if any([len(x.shape) != 4, x.shape[2] % stride != 0, x.shape[3] % stride != 0]):
        raise Exception("%s: size mismatch: %s" % (expr, x.shape))
    new_shape = (
        x.shape[0],
        x.shape[1],
        x.shape[2] // stride,
        stride,
        x.shape[3] // stride,
        stride,
    )
    return np.nanmax(x.reshape(new_shape), axis=(3, 5))


def Conv2d(expr, op, args, **kwargs):
    x = args[0]
    if not hasattr(x, "shape"):
        raise Exception("%s: ndarray only: %s" % (expr, x))
    if "weight" not in op.parameters or "bias" not in op.parameters:
        raise Exception("%s: missing weight or bias" % expr)
    weight = op.parameters["weight"]
    bias = op.parameters["bias"]
    in_channels = op.parameters["in_channels"]
    out_channels = op.parameters["out_channels"]
    kernel_size = op.parameters["kernel_size"]
    padding = op.parameters["padding"]
    if any(
        [
            len(x.shape) != 4,
            x.shape[1] != in_channels,
            weight.shape != (out_channels, in_channels, kernel_size, kernel_size),
            bias.shape != (out_channels,),
        ]
    ):
        raise Exception("%s: size mismatch: %s" % (expr, x.shape))
    if padding != 0:
        tmp = np.zeros(
            (x.shape[0], x.shape[1], x.shape[2] + 2 * padding, x.shape[3] + 2 * padding)
        )
        tmp[:, :, 1:-2, 1:-2] = x
        x = tmp
    conv_shape = x.shape[:2] + (
        x.shape[2] + 1 - kernel_size,
        x.shape[3] + 1 - kernel_size,
        kernel_size,
        kernel_size,
    )
    conv_strides = x.strides + x.strides[2:]
    conv = np.lib.stride_tricks.as_strided(
        x, shape=conv_shape, strides=conv_strides, writeable=False
    )
    return np.einsum(
        "nihwyx,oiyx->nohw", conv, weight, optimize="optimal"
    ) + bias.reshape((1, out_channels, 1, 1))


class Eval:
    def __init__(self, program):
        self.program = program

    def __call__(self, **kwargs):
        start = time.time()
        values = {}
        for expr in self.program:
            args = [values[ex.id] for ex in expr.inputs]
            if expr.op.op_type not in globals():
                raise Exception("%s: not implemented" % expr)
            values[expr.id] = globals()[expr.op.op_type](expr, expr.op, args, **kwargs)
            # print("numpy op", expr.op.op_type, "time %.2f" % (time.time()-start))
        res = values[self.program[-1].id]
        t = time.time() - start
        if t > 0.1:
            print("numpy time %.2f" % t)
        return res


class Builder:
    def __init__(self):
        self.program = []

    def append(self, expr):
        self.program.append(expr)

    def build(self):
        return Eval(self.program)
