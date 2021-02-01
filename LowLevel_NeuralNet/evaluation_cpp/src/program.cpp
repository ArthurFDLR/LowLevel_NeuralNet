#include "program.h"
#include "evaluation.h"
#include "expression.h"
#include "tensor.h"

#include <vector>

program::program()
{
}

void program::append_expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int inputs[],
    int num_inputs)
{
    expression new_expression = expression(expr_id, op_name, op_type, inputs, num_inputs);
    expressions_list_.push_back(new_expression);
}

int program::add_op_param_double(
    const char *key,
    double value)
{
    tensor scalar(value);
    expressions_list_.back().add_op_param(key, scalar);
    return 0;
}

int program::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    tensor ndararay(dim, shape, data);
    expressions_list_.back().add_op_param(key, ndararay);
    return 0;
}

void program::build(evaluation *eval)
{
    eval->set_expressions(expressions_list_);
}
