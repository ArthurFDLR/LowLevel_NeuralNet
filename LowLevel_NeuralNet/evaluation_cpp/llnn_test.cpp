/**
 * A simple test program helps you to debug the llnn implementation.
 */

#include <stdio.h>
#include "src/llnn.h"

int main()
{
    program *prog = create_program();

    int inputs0[] = {};
    append_expression(prog, 0, "a", "Input", inputs0, 0);

    //append_expression(prog, 0, "x", "Input", inputs0, 0);
    //add_op_param_double(prog, "value", 0.32);

    //append_expression(prog, 1, "", "Const", inputs0, 0);
    //add_op_param_double(prog, "value", 0.1);

    int inputs2[] = {0};
    append_expression(prog, 1, "", "ReLU", inputs2, 1);

    evaluation *eval = build(prog);
    add_kwargs_double(eval, "a", 5);
    size_t shape_a[] =  {3,3};
    double data_a[] = {1,2,3,4,5,6,7,8,9};
    add_kwargs_ndarray(eval, "a", 2, shape_a, data_a);

    int dim = 0;
    size_t *shape = nullptr;
    double *data = nullptr;
    if (execute(eval, &dim, &shape, &data) != 0)
    {
        printf("evaluation fails\n");
        return -1;
    }

    if (dim == 0)
        printf("res = %f\n", data[0]);
    else
        printf("result as tensor is not supported yet\n");

    return 0;
}
