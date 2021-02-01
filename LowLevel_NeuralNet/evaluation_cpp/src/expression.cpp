#include "expression.h"

#include <vector>
#include <string>
#include <map>

expression::expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int *inputs,
    int num_inputs)
{
    this->expr_id = expr_id;
    this->op_name = op_name;
    this->op_type = op_type;

    for (int i = 0; i<num_inputs; i++){
        this->inputs.push_back(inputs[i]);
    }
}

void expression::add_op_param(const char *key, tensor param){
    if(! params_.insert(std::pair<std::string, tensor>(key,param)).second){
        params_[key] = param;
    }
}

int expression::get_id() const
{
    return expr_id;
}

size_t expression::get_nbr_input() const
{
    return inputs.size();
}

std::string expression::get_op_name() const
{
    return op_name;
}

std::string expression::get_op_type() const
{
    return op_type;
}

int expression::get_input(int index) const
{
    return inputs[index];
}

tensor expression::get_op_param(std::string name) const
{
    //check if param exists
    return params_.at(name); // Method at() is const, not operator[]
}