#ifndef LLNN_H
#define LLNN_H

class program;
class evaluation;

// return program handle
extern "C"
program *create_program();

extern "C"
void append_expression(
    program *prog,
    int expr_id,
    const char *op_name,
    const char *op_type,
    int inputs[],
    int num_inputs);

// return 0 for success
extern "C"
int add_op_param_double(
    program *prog,
    const char *key,
    double value);

// return 0 for success
extern "C"
int add_op_param_ndarray(
    program *prog,
    const char *key,
    int dim,
    size_t shape[],
    double data[]);

// return evaluation handle
extern "C"
evaluation *build(
    program *prog);

extern "C"
void add_kwargs_double(
    evaluation *eval,
    const char *key,
    double value);

extern "C"
void add_kwargs_ndarray(
    evaluation *eval,
    const char *key,
    int dim,
    size_t shape[],
    double data[]);

// return 0 for success
extern "C"
int execute(
    evaluation *eval,
    int *p_dim,
    size_t **p_shape,
    double **p_data);

#endif // LLNN_H
