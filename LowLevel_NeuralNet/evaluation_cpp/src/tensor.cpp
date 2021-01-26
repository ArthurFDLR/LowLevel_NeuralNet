#include <assert.h>
#include <vector>
#include <iostream>
#include <limits> // Used in get_pool_max

#include "tensor.h"

tensor::tensor(): data_(1,0){}

tensor::tensor(double v): data_(1,v){}

//tensor::tensor(const tensor &t): shape_(t.shape_), data_(t.data_){}

tensor::tensor(int dim, size_t shape[]): shape_(shape, shape+dim){
    assert(dim>=0);
    size_t N = 1;
    for (int i = 0; i < dim; i++) {
        N *= shape[i];
    }
    //N += (N == 0)? 1 : 0;
    data_.assign(N, 0.);
}


tensor::tensor(int dim, size_t shape[], double data[]): shape_(shape, shape+dim){
    assert(dim>=0);
    size_t N = 1;
    for (int i = 0; i < dim; i++) {
        N *= shape[i];
    }
    //N += (N == 0)? 1 : 0;
    data_.assign(data, data+N);
}

size_t tensor::get_dim() const {
    return size_t(shape_.size());
}

double tensor::item() const {
    assert(shape_.empty());
    return data_[0];
}

double &tensor::item() {
    assert(shape_.empty());
    return data_[0];
}

double tensor::at(size_t i) const {
    assert(get_dim() == 1);
    assert(i < shape_[0]);
    return data_[i];
}

double tensor::at(size_t i, size_t j) const {
    assert(get_dim() == 2);
    assert((i < shape_[0]) && (j < shape_[1]));
    return data_[i*shape_[1] + j];
}

double &tensor::at(size_t i, size_t j){
    assert(get_dim() == 2);
    assert((i < shape_[0]) && (j < shape_[1]));
    return data_[i*shape_[1] + j];
}

double tensor::at(position pos) const {
    assert(get_dim() == pos.size());
    assert(pos[0] < shape_[0]);
    size_t index = pos[0];

    for (size_t i = 1; i < get_dim(); i++)
    {
        assert(pos[i] < shape_[i]);
        index *= shape_[i];
        index += pos[i];
    }
    return data_[index];
}

double &tensor::at(position pos){
    assert(get_dim() == pos.size());
    assert(pos[0] < shape_[0]);
    size_t index = pos[0];

    for (size_t i = 1; i < get_dim(); i++)
    {
        assert(pos[i] < shape_[i]);
        index *= shape_[i];
        index += pos[i];
    }
    return data_[index];
}

size_t *tensor::get_shape_array(){
    return shape_.empty() ? nullptr : &shape_[0]; 
}

double *tensor::get_data_array(){
    return &data_[0];
}

bool tensor::same_shape_as(const tensor *t) const{
    return shape_ == t->shape_;
}

tensor tensor::operator+(tensor const &t) const{
    assert(shape_ == t.shape_);
    int dim = get_dim();
    size_t shape[dim];
    std::copy(shape_.begin(), shape_.end(), shape);
    tensor out(dim, shape);

    for (size_t i = 0; i < data_.size(); i++)
    {
        out.data_[i] = data_[i] + t.data_[i];
    }
    return out;
}

tensor tensor::operator-(tensor const &t) const{
    assert(shape_ == t.shape_);
    int dim = get_dim();
    size_t shape[dim];
    std::copy(shape_.begin(), shape_.end(), shape);
    tensor out(dim, shape);

    for (size_t i = 0; i < data_.size(); i++)
    {
        out.data_[i] = data_[i] - t.data_[i];
    }
    return out;
}

tensor tensor::operator*(tensor const &t) const{
    if ((get_dim() == t.get_dim()) && (get_dim() == 2)) {
        return mult_dim2(t);
    } else if (get_dim() == 0 || t.get_dim() == 0){
        return mult_by_scalar(t);
    }
    assert(false && "Multiplication not supported for these shapes.");
}

tensor tensor::operator*(double const &s) const{
    int dim = get_dim();
    size_t shape[dim];
    std::copy(shape_.begin(), shape_.end(), shape);
    tensor out(dim, shape);

    for (size_t i = 0; i < data_.size(); i++)
    {
        out.data_[i] = data_[i] * s;
    }
    return out;
}

tensor tensor::mult_element_wise(tensor const &t) const{
    // Element wise multiplication
    assert(shape_ == t.shape_);

    int dim = get_dim();
    size_t shape[dim];
    std::copy(shape_.begin(), shape_.end(), shape);
    tensor out(dim, shape);

    for (size_t i = 0; i < data_.size(); i++)
    {
        out.data_[i] = data_[i] * t.data_[i];
    }
    return out;
}

tensor tensor::mult_dim2(tensor const &t) const{
    /*  Multiplication of two 2 dimensions matrix */
    assert((get_dim() == t.get_dim()) && (get_dim() == 2));
    assert(shape_[1] == t.shape_[0]);

    size_t common_dim = shape_[1];
    int new_dim = get_dim();
    size_t new_shape[] = {shape_[0], t.shape_[1]};
    tensor out(new_dim, new_shape);

    for (size_t i = 0; i < new_shape[0]; i++){
        for (size_t j = 0; j < new_shape[1]; j++){
            for (size_t k = 0; k < common_dim; k++){
                out.at(i,j) += at(i,k) * t.at(k,j);
            }
        }
    }
    return out;
}

tensor tensor::mult_by_scalar(tensor const &t) const {
    assert(get_dim() == 0 || t.get_dim() == 0);
    return (get_dim() == 0)? t * item() : *this * t.item() ;
}

tensor tensor::relu() const{
    int dim = get_dim();
    size_t shape[dim];
    std::copy(shape_.begin(), shape_.end(), shape);
    tensor out(dim, shape);

    for (size_t i = 0; i < data_.size(); i++)
    {
        out.data_[i] = (data_[i] < 0.0)?0.0:data_[i];
    }
    return out;
}

tensor tensor::flatten() const{
    int dim = get_dim();
    assert(dim>1);
    size_t shape[2];
    shape[0] = shape_[0];
    shape[1] = 1;
    for (size_t i = 1; i < size_t(dim); i++)
    {
        shape[1] *= shape_[i];
    }
    tensor out(2, shape);
    out.data_ = data_; // Recquieres data_ to store data using row-major order
    return out;
}

tensor tensor::nhwc2nchw() const{
    int dim = get_dim();
    assert(dim==4);
    size_t shape[dim];
    shape[0] = shape_[0];
    shape[1] = shape_[3];
    shape[2] = shape_[1];
    shape[3] = shape_[2];
    tensor out(dim, shape);
    for (size_t n = 0; n < shape_[0]; n++)
    {
        for (size_t h = 0; h < shape_[1]; h++)
        {
            for (size_t w = 0; w < shape_[2]; w++)
            {
                for (size_t c = 0; c < shape_[3]; c++)
                {
                    out.at(position({n,c,h,w})) = at(position({n,h,w,c}));
                }
            }
        }
    }
    return out;
}

tensor tensor::linear(tensor const &weight, tensor const &bias) const{
    size_t N = shape_[0];
    size_t I = shape_[1];
    size_t O = bias.shape_[0];

    assert(get_dim() == 2);
    assert(weight.get_dim() == 2);
    assert(bias.get_dim() == 1);
    assert(weight.shape_[0] == O);
    assert(weight.shape_[1] == I);

    size_t shape[] = {N,O};
    tensor out(2, shape);

    for (size_t n = 0; n < N; n++)
    {
        for (size_t o = 0; o < O; o++)
        {
            out.at(n,o) += bias.at(o);
            for (size_t i = 0; i < I; i++)
            {
                out.at(n,o) += weight.at(o,i) * at(n,i);
            }
        }
    }
    return out;
}

tensor tensor::maxpool2d(tensor const &kernel_size, tensor const &stride) const{
    int dim = get_dim();
    size_t kernel_size_int = size_t(kernel_size.item());
    size_t stride_int = size_t(stride.item());
    size_t shape[dim];

    assert(dim==4);
    assert((shape_[2] >= kernel_size_int) && (shape_[3] >= kernel_size_int));

    shape[0] = shape_[0]; //N
    shape[1] = shape_[1]; //C
    shape[2] = shape_[2]/kernel_size_int; //H
    shape[3] = shape_[3]/kernel_size_int; //W
    tensor out(dim, shape);

    size_t offset_h = (shape_[2] % kernel_size_int) / 2;
    size_t offset_w = (shape_[3] % kernel_size_int) / 2;

    for (size_t n = 0; n < shape[0]; n++) 
    {
        for (size_t c = 0; c < shape[1]; c++)
        {   
            for (size_t h = 0; h < shape[2]; h++)
            {
                for (size_t w = 0; w < shape[3]; w++)
                {
                    size_t h_init = offset_h + h * stride_int;
                    size_t w_init = offset_w + w * stride_int;
                    out.at(position({n,c,h,w})) = get_pool_max(n, c, h_init, w_init, kernel_size_int);
                }
            }
        }
    }
    return out;
}

double tensor::get_pool_max(size_t ax_0, size_t ax_1, size_t upl_h, size_t upl_w, size_t kernel_size) const{
    assert(get_dim() == 4);
    assert(upl_h + kernel_size <= shape_[2]);
    assert(upl_w + kernel_size <= shape_[3]);

    double max_value = std::numeric_limits<double>::lowest();
    for (size_t h = upl_h; h < upl_h + kernel_size; h++)
    {
        for (size_t w = upl_w; w < upl_w + kernel_size; w++)
        {
            if (at(position({ax_0,ax_1,h,w})) > max_value)
            {
                max_value = at(position({ax_0,ax_1,h,w}));
            }
        }
    }
    return max_value;
}

tensor tensor::conv2d(tensor const &weight, tensor const &bias, tensor const &padding) const{
    int dim = get_dim();
    assert(size_t(padding.item()) == 0);
    assert(dim==4);
    assert(weight.get_dim() == 4);
    assert(bias.get_dim() == 1);
    assert(bias.shape_[0] == weight.shape_[0]); // C_OUT
    assert(shape_[1] == weight.shape_[1]); //C_IN

    size_t shape[dim];
    size_t kernel_size = weight.shape_[2];
    assert(weight.shape_[3] == kernel_size);

    shape[0] = shape_[0]; //N
    shape[1] = weight.shape_[0]; //C_OUT
    shape[2] = shape_[2] - kernel_size + 1; //H
    shape[3] = shape_[3] - kernel_size + 1; //W
    tensor out(dim, shape);

    for (size_t n = 0; n < shape[0]; n++) 
    {
        for (size_t c_out = 0; c_out < shape[1]; c_out++)
        {   
            for (size_t h_out = 0; h_out < shape[2]; h_out++)
            {
                for (size_t w_out = 0; w_out < shape[3]; w_out++)
                {
                    out.at(position({n,c_out,h_out,w_out})) = bias.at(c_out);

                    for (size_t c_in = 0; c_in < shape_[1]; c_in++)
                    {
                        for (size_t k_h = 0; k_h < kernel_size; k_h++)
                        {
                            for (size_t k_w = 0; k_w < kernel_size; k_w++)
                            {
                                size_t h_in = h_out + k_h;
                                size_t w_in = w_out + k_w;
                                out.at(position({n,c_out,h_out,w_out})) += weight.at(position({c_out,c_in,k_h,k_w})) * at(position({n,c_in,h_in,w_in}));
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}