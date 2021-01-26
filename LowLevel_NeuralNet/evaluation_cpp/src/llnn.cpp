#include <stdio.h>

#include "llnn.h"
#include "program.h"
#include "evaluation.h"
#include "tensor.h"

program *create_program()
{
    program *prog = new program;
    printf("create_program: \t program %p\n", prog);
    return prog;
}

void append_expression(
    program *prog,
    int expr_id,
    const char *op_name,
    const char *op_type,
    int inputs[],
    int num_inputs)
{
    printf("append_expression: \t program %p, expr_id %d, op_name %s, op_type %s, inputs %d (",
        prog, expr_id, op_name, op_type, num_inputs);
    for (int i = 0; i != num_inputs; ++i)
        printf("%d,", inputs[i]);
    printf(")\n");
    prog->append_expression(expr_id, op_name, op_type, inputs, num_inputs);
}

int add_op_param_double(
    program *prog,
    const char *key,
    double value)
{
    printf("add_op_param_double: \t program %p, key %s, value %f\n",
        prog, key, value);
    return prog->add_op_param_double(key, value);
}

int add_op_param_ndarray(
    program *prog,
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    printf("add_op_param_ndarray: \t program %p, key %s, value %p dim %d (",
        prog, key, data, dim);
    for (int i = 0; i != dim; ++i)
        printf("%zu,", shape[i]);
    printf(")\n");
    return prog->add_op_param_ndarray(key, dim, shape, data);
}

evaluation *build(program *prog)
{
    evaluation *eval = new evaluation;
    prog->build(eval);
    printf("build: \t evaluation %p\n", eval);
    return eval;
}

void add_kwargs_double(
    evaluation *eval,
    const char *key,
    double value)
{
    printf("add_kwargs_double: \t evaluation %p, key %s, value %f\n",
        eval, key, value);
    eval->add_kwargs_double(key, value);
}

void add_kwargs_ndarray(
    evaluation *eval,
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    printf("add_kwargs_ndarray: \t evaluation %p, key %s, value %p dim %d (",
        eval, key, data, dim);
    for (int i = 0; i != dim; ++i)
        printf("%zu,", shape[i]);
    printf(")\n");
    eval->add_kwargs_ndarray(key, dim, shape, data);
}

int execute(
    evaluation *eval,
    int *p_dim,
    size_t **p_shape,
    double **p_data)
{
    printf("execute: \t evaluation %p, p_dim %p, p_shape %p, p_data %p\n",
        eval, p_dim, p_shape, p_data);
    int ret = eval->execute();
    if (ret != 0)
        return ret;
    tensor &res = eval->get_result();
    *p_dim = res.get_dim();
    *p_shape = res.get_shape_array();
    *p_data = res.get_data_array();
    fflush(stdout);
    return 0;
}
