#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <vector>
#include <string>
#include <map>

#include "tensor.h"

class evaluation;

class expression
{
    friend class evaluation;
public:
    expression(
        int expr_id,
        const char *op_name,
        const char *op_type,
        int *inputs,
        int num_inputs);
    
    void add_op_param(const char *key, tensor param);

    int get_id() const;
    std::string get_op_name() const;
    std::string get_op_type() const;
    int get_input(int index = 0) const;
    size_t get_nbr_input() const;
    //double get_param_double(const char * name);
    tensor get_op_param(std::string name) const;

private:
    int expr_id; 
    std::string op_name;
    std::string op_type;
    std::vector<int> inputs;
    std::map<std::string, tensor> params_;
}; // class expression

#endif // EXPRESSION_H
