class Expr:
    next_id = 0

    def __init__(self, op, inputs):
        self.op = op
        self.inputs = inputs
        self.id = Expr.next_id
        Expr.next_id += 1

        if not isinstance(op, Op):
            raise Exception("Not an operator: %s" % op)

    def __dfs_post(self, ids, visitor):
        ids[self.id] = True
        for expr in self.inputs:
            if expr.id in ids:
                continue
            expr.__dfs_post(ids, visitor)
        visitor(self)

    def statements(self):
        lines = []
        self.__dfs_post({}, lambda that: lines.append("%s" % that))
        return "\n".join(lines)

    def __str__(self):
        args = ",".join(["t%d" % expr.id for expr in self.inputs])
        return "t%d = %s(%s)" % (self.id, self.op, args)

    def __promote(r):
        if isinstance(r, Expr):
            return r
        else:
            return Const(r)

    def __add__(self, r):
        return Op("", "Add", 2, {})(self, Expr.__promote(r))

    def __sub__(self, r):
        return Op("", "Sub", 2, {})(self, Expr.__promote(r))

    def __mul__(self, r):
        return Op("", "Mul", 2, {})(self, Expr.__promote(r))

    def __neg__(self):
        return Op("", "Neg", 1, {})(self)

    # Custom
    def __pow__(self, r):
        return Op("", "Pow", 2, {})(self, Expr.__promote(r))

    def compile(self, builder):
        self.__dfs_post({}, lambda that: builder.append(that))
        return builder.build()

    def resolve(self, parameters):
        self.__dfs_post({}, lambda that: that.op.resolve(parameters))
        return self


class Op:
    def __init__(self, name, op_type, num_args, parameters):
        self.name = name
        self.op_type = op_type
        self.num_args = num_args
        self.parameters = parameters

    def __call__(self, *inputs):
        if self.num_args >= 0 and self.num_args != len(inputs):
            raise Exception(
                "%s: need %d arguments but found %d"
                % (self, self.num_args, len(inputs))
            )
        for i, expr in enumerate(inputs):
            if not isinstance(expr, Expr):
                raise Exception("%s: arg %d is not an expression: %s" % (self, i, expr))
        return Expr(self, inputs)

    def __str__(self):
        name = "%s.%s" % (self.name, self.op_type)
        if len(self.parameters) == 0:
            return name
        params = ",".join(
            [
                "%s=%s" % (k, v.shape if hasattr(v, "shape") else v)
                for k, v in self.parameters.items()
            ]
        )
        return "%s[%s]" % (name, params)

    def resolve(self, parameters):
        if self.name == "":
            return
        for k, v in parameters.items():
            if k.startswith(self.name + "."):
                self.parameters[k[len(self.name) + 1 :]] = v


def Const(c):
    if isinstance(c, (int, float)):
        c = float(c)
    elif hasattr(c, "shape"):
        c = c.astype(float)
    else:
        raise Exception("Const must be float or int or ndarray: %s" % c)

    return Expr(Op("", "Const", 0, {"value": c}), [])


def Input(n):
    return Expr(Op(n, "Input", 0, {}), [])


def Input2d(n, h, w, ic):
    return Expr(Op(n, "Input2d", 0, {"height": h, "width": w, "in_channels": ic}), [])


def MaxPool2d(k, s):
    return Op("", "MaxPool2d", 1, {"kernel_size": k, "stride": s})


def ReLU():
    return Op("", "ReLU", 1, {})


def Flatten():
    return Op("", "Flatten", 1, {})


def Conv2d(n, ic, oc, k, p=0):
    return Op(
        n,
        "Conv2d",
        1,
        {"in_channels": ic, "out_channels": oc, "kernel_size": k, "padding": p},
    )


def Linear(n, i, o):
    return Op(n, "Linear", 1, {"in_features": i, "out_features": o})


def Show():
    return Op("", "Show", 1, {})
