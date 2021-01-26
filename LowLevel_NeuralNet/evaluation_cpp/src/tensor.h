#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

typedef std::vector<size_t> position;

class tensor {
public:
    tensor();
    explicit tensor(double v);
    tensor(int dim, size_t shape[]); // initialize data with 0s
    explicit tensor(int dim, size_t shape[], double data[]);

    size_t get_dim() const;
    double item() const;
    double &item();

    double at(size_t i) const;
    double at(size_t i, size_t j) const;
    double &at(size_t i, size_t j);

    double at(position pos) const;
    double &at(position pos);

    size_t *get_shape_array();
    double *get_data_array();

    bool same_shape_as(const tensor *t) const;

    tensor operator+(tensor const &t) const;
    tensor operator-(tensor const &t) const;
    tensor operator*(tensor const &t) const;
    tensor operator*(double const &s) const;

    // Neural Network related
    tensor relu() const;
    tensor flatten() const;
    tensor nhwc2nchw() const;
    tensor linear(tensor const &weight, tensor const &bias) const;
    tensor maxpool2d(tensor const &kernel_size, tensor const &stride) const;
    tensor conv2d(tensor const &weight, tensor const &bias, tensor const &padding) const;

private:
    std::vector<size_t> shape_;
    std::vector<double> data_;
    tensor mult_dim2(tensor const &t) const;
    tensor mult_element_wise(tensor const &t) const; // Not used (or tested) yet
    tensor mult_by_scalar(tensor const &t) const;

    /**
     * @brief Only allows tensor of 4 dimensions of format NCHW. Get the maximum value in a block of kernel_size*kernel_size at a specific position.
     * 
     * @param ax_0 First axis position (example number).
     * @param ax_1 Second axis position (channel).
     * @param upl_h Upper-left height position.
     * @param upl_w Upper-left width position.
     * @return Maximum value.
     */
    double get_pool_max(size_t ax_0, size_t ax_1, size_t upl_h, size_t upl_w, size_t kernel_size) const;
    
};

#endif // EVALUATION_H