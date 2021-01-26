#include <assert.h>

#include <map>
#include <vector>
#include <cstring> // std::strcmp
#include <iostream>
#include <string>

#include "evaluation.h"



// EVALUATION //
////////////////

evaluation::evaluation(const std::vector<expression> &exprs){  
    for (auto &expr: exprs){
        std::shared_ptr<eval_op> p = eval_op_prototypes::instance().locate(expr.get_op_type());
        ops_.push_back(p->clone(expr));
    }
    expressions_list_ = exprs;
}

void evaluation::add_kwargs_double(
    const char *key,
    double value)
{
    tensor scalar(value);
    if(! kwargs_.insert(std::pair<std::string, tensor>(key,scalar)).second){
        kwargs_[key] = scalar;
    }
}

void evaluation::add_kwargs_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    tensor ndarray(dim, shape, data);
    if(! kwargs_.insert(std::pair<std::string, tensor>(key,ndarray)).second){
        kwargs_[key] = ndarray;
    }
}

int evaluation::execute()
{
    variables_.clear(); // Store the results of each expression (the key being the id of the expression)
    for (auto &op : ops_){
        op->eval(variables_, kwargs_);
    }
    return 0;
}

tensor &evaluation::get_result()
{
    return (--variables_.end())->second;
}

void evaluation::set_expressions(const std::vector<expression> &exprs)
{
    for (auto &expr: exprs){
        std::shared_ptr<eval_op> p = eval_op_prototypes::instance().locate(expr.get_op_type());
        ops_.push_back(p->clone(expr));
    }
    expressions_list_ = exprs;
}

// EVAL_OP //
/////////////

eval_op::~eval_op() {}

eval_op::eval_op(const expression &expr){
    expr_id_ = expr.get_id();
    op_name_ = expr.get_op_name();
    op_type_ = expr.get_op_type();
    for (size_t i = 0; i < expr.get_nbr_input(); i++)
    {
        inputs_.push_back(expr.get_input(i));
    }
}

// EVAL_OP_PROTOTYPES //
////////////////////////

eval_op_prototypes::eval_op_prototypes(){
    eval_const::store_prototype(proto_map_);
    eval_input::store_prototype(proto_map_);
    eval_input2d::store_prototype(proto_map_);
    eval_add::store_prototype(proto_map_);
    eval_sub::store_prototype(proto_map_);
    eval_mul::store_prototype(proto_map_);
    eval_relu::store_prototype(proto_map_);
    eval_flatten::store_prototype(proto_map_);
    eval_linear::store_prototype(proto_map_);
    eval_maxpool2d::store_prototype(proto_map_);
    eval_conv2d::store_prototype(proto_map_);
}

eval_op_prototypes &eval_op_prototypes::instance() {
    static eval_op_prototypes instance;
    return instance;
}

std::shared_ptr<eval_op> eval_op_prototypes::locate(std::string name){
    return proto_map_[name];
}


// EVAL_CONST //
////////////////

void eval_const::store_prototype(eval_op_proto_map &proto_map){
    assert(proto_map.find("Const") == proto_map.end());
    proto_map["Const"] = std::make_shared<eval_const>();
}

void eval_const::eval(vars_type &variables, const kwargs_type &kwargs) {
    // check if expr_id_ is key of variables
    variables[expr_id_] = value_;
}

eval_const::eval_const(const expression &expr): eval_op(expr), value_(expr.get_op_param("value")){
}

std::shared_ptr<eval_op> eval_const::clone(const expression &expr){
    return std::make_shared<eval_const>(expr);
}

// EVAL_INPUT //
////////////////

void eval_input::store_prototype(eval_op_proto_map &proto_map){
    assert(proto_map.find("Input") == proto_map.end());
    proto_map["Input"] = std::make_shared<eval_input>();
}

void eval_input::eval(vars_type &variables, const kwargs_type &kwargs) {
    variables[expr_id_] = kwargs.at(op_name_);
}

eval_input::eval_input(const expression &expr): eval_op(expr){}

std::shared_ptr<eval_op> eval_input::clone(const expression &expr){
    return std::make_shared<eval_input>(expr);
}

// EVAL_INPUT2D //
//////////////////

void eval_input2d::store_prototype(eval_op_proto_map &proto_map){
    assert(proto_map.find("Input2d") == proto_map.end());
    proto_map["Input2d"] = std::make_shared<eval_input2d>();
}

void eval_input2d::eval(vars_type &variables, const kwargs_type &kwargs) {
    // kwargs stored in NHWC format but stores in variables in format NCHW
    variables[expr_id_] = (kwargs.at(op_name_)).nhwc2nchw();
}

eval_input2d::eval_input2d(const expression &expr): eval_op(expr){}

std::shared_ptr<eval_op> eval_input2d::clone(const expression &expr){
    return std::make_shared<eval_input2d>(expr);
}

// EVAL_BINARY //
/////////////////

void eval_binary::eval(vars_type &variables, const kwargs_type &kwargs) {
    assert(inputs_.size() == 2);
    auto ita = variables.find(inputs_[0]);
    auto itb = variables.find(inputs_[1]);
    // compatibility errors
    variables[expr_id_] = compute(ita->second, itb->second);
}

// EVAL_ADD //
//////////////

void eval_add::store_prototype(eval_op_proto_map &proto_map){
    assert(proto_map.find("Add") == proto_map.end());
    proto_map["Add"] = std::make_shared<eval_add>();
}

std::shared_ptr<eval_op> eval_add::clone(const expression &expr){
    return std::make_shared<eval_add>(expr);
}

tensor eval_add::compute(const tensor &a, const tensor &b){
    assert(a.same_shape_as(&b));
    tensor c = a + b;
    return c;
}


// EVAL_SUB //
//////////////

void eval_sub::store_prototype(eval_op_proto_map &proto_map){
    assert(proto_map.find("Sub") == proto_map.end());
    proto_map["Sub"] = std::make_shared<eval_sub>();
}

std::shared_ptr<eval_op> eval_sub::clone(const expression &expr){
    return std::make_shared<eval_sub>(expr);
}

tensor eval_sub::compute(const tensor &a, const tensor &b){
    assert(a.same_shape_as(&b));
    tensor c = a - b;
    return c;
}


// EVAL_MUL //
//////////////

void eval_mul::store_prototype(eval_op_proto_map &proto_map){
    assert(proto_map.find("Mul") == proto_map.end());
    proto_map["Mul"] = std::make_shared<eval_mul>();
}

std::shared_ptr<eval_op> eval_mul::clone(const expression &expr){
    return std::make_shared<eval_mul>(expr);
}

tensor eval_mul::compute(const tensor &a, const tensor &b){
    tensor c = a * b;
    return c;
}

// EVAL_RELU //
///////////////

void eval_relu::store_prototype(eval_op_proto_map &proto_map){
    assert(proto_map.find("ReLU") == proto_map.end());
    proto_map["ReLU"] = std::make_shared<eval_relu>();
}

void eval_relu::eval(vars_type &variables, const kwargs_type &kwargs) {
    assert(inputs_.size() == 1);
    //tensor input_tensor = variables.find(inputs_[0])->second;
    variables[expr_id_] = (variables.find(inputs_[0])->second).relu();
}

eval_relu::eval_relu(const expression &expr): eval_op(expr){}

std::shared_ptr<eval_op> eval_relu::clone(const expression &expr){
    return std::make_shared<eval_relu>(expr);
}

// EVAL_FLATTEN //
//////////////////

void eval_flatten::store_prototype(eval_op_proto_map &proto_map){
    assert(proto_map.find("Flatten") == proto_map.end());
    proto_map["Flatten"] = std::make_shared<eval_flatten>();
}

void eval_flatten::eval(vars_type &variables, const kwargs_type &kwargs) {
    assert(inputs_.size() == 1);
    variables[expr_id_] = (variables.find(inputs_[0])->second).flatten();
}

eval_flatten::eval_flatten(const expression &expr): eval_op(expr){}

std::shared_ptr<eval_op> eval_flatten::clone(const expression &expr){
    return std::make_shared<eval_flatten>(expr);
}

// EVAL_LINEAR //
/////////////////

void eval_linear::store_prototype(eval_op_proto_map &proto_map){
    assert(proto_map.find("Linear") == proto_map.end());
    proto_map["Linear"] = std::make_shared<eval_linear>();
}

void eval_linear::eval(vars_type &variables, const kwargs_type &kwargs) {
    assert(inputs_.size() == 1);
    variables[expr_id_] = (variables.find(inputs_[0])->second).linear(weight_, bias_);
}

eval_linear::eval_linear(const expression &expr):
    eval_op(expr), weight_(expr.get_op_param("weight")), bias_(expr.get_op_param("bias")){}

std::shared_ptr<eval_op> eval_linear::clone(const expression &expr){
    return std::make_shared<eval_linear>(expr);
}

// EVAL_MAXPOOL2D //
////////////////////

void eval_maxpool2d::store_prototype(eval_op_proto_map &proto_map){
    assert(proto_map.find("MaxPool2d") == proto_map.end());
    proto_map["MaxPool2d"] = std::make_shared<eval_maxpool2d>();
}

void eval_maxpool2d::eval(vars_type &variables, const kwargs_type &kwargs) {
    assert(inputs_.size() == 1);
    variables[expr_id_] = (variables.find(inputs_[0])->second).maxpool2d(kernel_size_, stride_);
}

eval_maxpool2d::eval_maxpool2d(const expression &expr):
    eval_op(expr), kernel_size_(expr.get_op_param("kernel_size")), stride_(expr.get_op_param("stride")){}

std::shared_ptr<eval_op> eval_maxpool2d::clone(const expression &expr){
    return std::make_shared<eval_maxpool2d>(expr);
}

// EVAL_CONV2D //
/////////////////

void eval_conv2d::store_prototype(eval_op_proto_map &proto_map){
    assert(proto_map.find("Conv2d") == proto_map.end());
    proto_map["Conv2d"] = std::make_shared<eval_conv2d>();
}

void eval_conv2d::eval(vars_type &variables, const kwargs_type &kwargs) {
    assert(inputs_.size() == 1);
    variables[expr_id_] = (variables.find(inputs_[0])->second).conv2d(weight_, bias_, padding_);
}

eval_conv2d::eval_conv2d(const expression &expr):
    eval_op(expr), padding_(expr.get_op_param("padding")),
    weight_(expr.get_op_param("weight")), bias_(expr.get_op_param("bias")){}

std::shared_ptr<eval_op> eval_conv2d::clone(const expression &expr){
    return std::make_shared<eval_conv2d>(expr);
}