#ifndef EVALUATION_H
#define EVALUATION_H

#include <map>
#include <vector>
#include <string>
#include <memory>

#include "expression.h"
#include "tensor.h"

typedef std::map<int, tensor> vars_type;
typedef std::map<std::string, tensor> kwargs_type;


class eval_op {
protected:
    int expr_id_;
    std::string op_name_, op_type_;
    std::vector<int> inputs_;
    eval_op(): expr_id_(-1) {}
    eval_op(const expression &expr);
public:
    virtual void eval(vars_type &variables, const kwargs_type &kwargs) = 0;
    virtual ~eval_op();
    virtual std::shared_ptr<eval_op> clone(const expression &expr) = 0;
}; // class eval_op


class evaluation
{
public:
    evaluation(){};
    evaluation(const std::vector<expression> &exprs);
    void set_expressions(const std::vector<expression> &exprs);
    void add_kwargs_double(
        const char *key,
        double value);
    void add_kwargs_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

    int execute();  // return 0 for success

    // return the variable computed by the last expression
    //double &get_result();
    tensor &get_result();

private:
    double result_;
    std::vector<expression> expressions_list_;
    std::vector<std::shared_ptr<eval_op>> ops_;
    kwargs_type kwargs_;
    vars_type variables_;
}; // class evaluation



typedef std::map<std::string, std::shared_ptr<eval_op>> eval_op_proto_map;

class eval_op_prototypes{
    eval_op_proto_map proto_map_;
protected:
    eval_op_prototypes(const eval_op_prototypes &) = delete;
    eval_op_prototypes();
public:
    std::shared_ptr<eval_op> locate(std::string name);
    static eval_op_prototypes &instance();
}; // class eval_op_prototype



class eval_const: public eval_op {
    tensor value_;
public:
    eval_const(){} // Should be protected but dont compile
    eval_const(const expression &expr); // Should be protected but dont compile

    void eval(vars_type &variables, const kwargs_type &kwargs) override;
    std::shared_ptr<eval_op> clone(const expression &expr) override;
    static void store_prototype(eval_op_proto_map &proto_map);
}; // class eval_const


class eval_input: public eval_op {
public:
    eval_input(){} // Should be protected but dont compile
    eval_input(const expression &expr); // Should be protected but dont compile

    void eval(vars_type &variables, const kwargs_type &kwargs) override;
    std::shared_ptr<eval_op> clone(const expression &expr) override;
    static void store_prototype(eval_op_proto_map &proto_map);
}; // class eval_input


class eval_input2d: public eval_op {
public:
    eval_input2d(){} // Should be protected but dont compile
    eval_input2d(const expression &expr); // Should be protected but dont compile

    void eval(vars_type &variables, const kwargs_type &kwargs) override;
    std::shared_ptr<eval_op> clone(const expression &expr) override;
    static void store_prototype(eval_op_proto_map &proto_map);
}; // class eval_input2d


class eval_binary: public eval_op {
    virtual tensor compute(const tensor &a, const tensor &b) = 0;
public:
    eval_binary(){}
    eval_binary(const expression &expr):eval_op(expr){}

    void eval(vars_type &variables, const kwargs_type &kwargs) final;
}; // class eval_binary



class eval_add: public eval_binary {
    tensor compute(const tensor &a, const tensor &b) override;
protected:
public:
    eval_add(){}
    eval_add(const expression &expr):eval_binary(expr){}

    static void store_prototype(eval_op_proto_map &proto_map);
    std::shared_ptr<eval_op> clone(const expression &expr) override;
}; // class eval_add



class eval_sub: public eval_binary {
    tensor compute(const tensor &a, const tensor &b) override;
public:
    eval_sub(){}
    eval_sub(const expression &expr):eval_binary(expr){}

    static void store_prototype(eval_op_proto_map &proto_map);
    std::shared_ptr<eval_op> clone(const expression &expr) override;
}; // class eval_sub


class eval_mul: public eval_binary {
    tensor compute(const tensor &a, const tensor &b) override;
public:
    eval_mul(){}
    eval_mul(const expression &expr):eval_binary(expr){}

    static void store_prototype(eval_op_proto_map &proto_map);
    std::shared_ptr<eval_op> clone(const expression &expr) override;
}; // class eval_mul


class eval_relu: public eval_op {
public:
    eval_relu(){} // Should be protected but dont compile
    eval_relu(const expression &expr); // Should be protected but dont compile

    void eval(vars_type &variables, const kwargs_type &kwargs) override;
    std::shared_ptr<eval_op> clone(const expression &expr) override;
    static void store_prototype(eval_op_proto_map &proto_map);
}; // class eval_relu


class eval_flatten: public eval_op {
public:
    eval_flatten(){} // Should be protected but dont compile
    eval_flatten(const expression &expr); // Should be protected but dont compile

    void eval(vars_type &variables, const kwargs_type &kwargs) override;
    std::shared_ptr<eval_op> clone(const expression &expr) override;
    static void store_prototype(eval_op_proto_map &proto_map);
}; // class eval_flatten


class eval_linear: public eval_op {
    tensor weight_;
    tensor bias_;
public:
    eval_linear(){} // Should be protected but dont compile
    eval_linear(const expression &expr); // Should be protected but dont compile

    void eval(vars_type &variables, const kwargs_type &kwargs) override;
    std::shared_ptr<eval_op> clone(const expression &expr) override;
    static void store_prototype(eval_op_proto_map &proto_map);
}; // class eval_linear


class eval_maxpool2d: public eval_op {
    tensor kernel_size_;
    tensor stride_;
public:
    eval_maxpool2d(){} // Should be protected but dont compile
    eval_maxpool2d(const expression &expr); // Should be protected but dont compile

    void eval(vars_type &variables, const kwargs_type &kwargs) override;
    std::shared_ptr<eval_op> clone(const expression &expr) override;
    static void store_prototype(eval_op_proto_map &proto_map);
}; // class eval_maxpool2d

class eval_conv2d: public eval_op {
    tensor padding_;
    tensor weight_;
    tensor bias_;
public:
    eval_conv2d(){} // Should be protected but dont compile
    eval_conv2d(const expression &expr); // Should be protected but dont compile

    void eval(vars_type &variables, const kwargs_type &kwargs) override;
    std::shared_ptr<eval_op> clone(const expression &expr) override;
    static void store_prototype(eval_op_proto_map &proto_map);
}; // class eval_conv2d

#endif // EVALUATION_H
