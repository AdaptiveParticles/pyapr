//
// Created by Joel Jonsson on 09.05.19.
//

#ifndef PYLIBAPR_APRNETOPS_HPP
#define PYLIBAPR_APRNETOPS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/cast.h>

#ifdef PYAPR_HAVE_OPENMP
#include <omp.h>
#endif

//#include "ConfigAPR.h"
//#include "data_structures/APR/APR.hpp"
//#include "data_structures/APR/particles/ParticleData.hpp"

//#include "PyPixelData.hpp"
#include "PyAPRFiltering.hpp"
//#include "data_containers/PyAPR.hpp"

#include <typeinfo>

namespace py = pybind11;

namespace APRNetOps {

    template<typename T>
    void templated_convolve(py::buffer_info &apr_buf, py::buffer_info &input_buf, py::buffer_info &weights_buf,
                            py::buffer_info &bias_buf, py::buffer_info &output_buf, py::buffer_info &dlvl_buf);

    template<typename T>
    void templated_convolve_backward(py::buffer_info &apr_buf, py::buffer_info &input_buf, py::buffer_info &weights_buf,
                                     py::buffer_info &grad_input_buf, py::buffer_info &grad_weights_buf,
                                     py::buffer_info &grad_bias_buf, py::buffer_info &grad_output_buf, py::buffer_info &dlvl_buf);

    template<typename T>
    void templated_convolve3x3(py::buffer_info &apr_buf, py::buffer_info &weights_buf, py::buffer_info &bias_buf,
                               py::buffer_info &output_buf, py::buffer_info &input_buf, py::buffer_info &dlvl_buf);

    template<typename T>
    void templated_convolve3x3_backward(py::buffer_info &apr_buf, py::buffer_info &input_buf, py::buffer_info &weights_buf,
                                        py::buffer_info &grad_input_buf, py::buffer_info &grad_weights_buf,
                                        py::buffer_info &grad_bias_buf, py::buffer_info &grad_output_buf, py::buffer_info &dlvl_buf);

    template<typename T>
    void templated_transposed_conv_2x2(py::buffer_info &apr_buf, py::buffer_info &weights_2x2_buf, py::buffer_info &weights_1x1_buf,
                    py::buffer_info &bias_buf, py::buffer_info &output_buf, py::buffer_info &input_buf, py::buffer_info &dlvl_buf);

    template<typename T>
    void templated_transposed_conv_2x2_backward(py::buffer_info &apr_buf, py::buffer_info &input_buf,
                                                py::buffer_info &weights_2x2_buf, py::buffer_info &weights_1x1_buf,
                                                py::buffer_info &grad_input_buf, py::buffer_info &grad_weights_2x2_buf,
                                                py::buffer_info &grad_weights_1x1_buf, py::buffer_info &grad_bias_buf,
                                                py::buffer_info &grad_output_buf, py::buffer_info &dlvl_buf);


    uint64_t number_parts_after_pool(PyAPR &aPyAPR, int level_delta) {

        unsigned int current_max_level = std::max(aPyAPR.level_max()-level_delta, aPyAPR.level_min());

        auto apr_iterator = aPyAPR.apr.iterator();
        auto tree_iterator = aPyAPR.apr.tree_iterator();

        return PyAPRFiltering::number_parts_at_level(apr_iterator, tree_iterator, current_max_level-1);
    }

    uint64_t number_parts_after_upsampling(PyAPR &aPyAPR, int level_delta) {

        unsigned int current_max_level = std::max(aPyAPR.level_max()-level_delta, aPyAPR.level_min());

        auto apr_iterator = aPyAPR.apr.iterator();
        auto tree_iterator = aPyAPR.apr.tree_iterator();

        uint64_t apr_parts = apr_iterator.total_number_particles(current_max_level + 1);
        uint64_t tree_parts = 0;

        if(current_max_level < tree_iterator.level_max()) {
            tree_parts = tree_iterator.total_number_particles(current_max_level + 1) - tree_iterator.total_number_particles(current_max_level);
        }

        return apr_parts + tree_parts;

    }

    void convolve(py::array &apr_list, py::array &input_features, py::array_t<float> &weights, py::array_t<float> &bias,
                  py::array &output, py::array_t<int32_t> &level_delta) {

        /// requesting buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto bias_buf = bias.request();
        auto output_buf = output.request(true);
        auto dlvl_buf = level_delta.request();

        /// check input/output data type and call templated convolution function if correct
        if(py::isinstance<py::array_t<float>>(input_features)) {
            if(!py::isinstance<py::array_t<float>>(output)) {
                throw std::invalid_argument("input and output arrays passed to convolve must have matching data types");
            }

            templated_convolve<float>(apr_buf, input_buf, weights_buf, bias_buf, output_buf, dlvl_buf);

        } else if(py::isinstance<py::array_t<double>>(input_features)) {
            if(!py::isinstance<py::array_t<double>>(output)) {
                throw std::invalid_argument("input and output arrays passed to convolve must have matching data types");
            }

            templated_convolve<double>(apr_buf, input_buf, weights_buf, bias_buf, output_buf, dlvl_buf);

        } else {
            throw std::invalid_argument("input and output arrays passed to convolve1x1 must be of type float (float32) or double (float64)");
        }
    }


    /**
     *
     * @param apr_list
     * @param grad_output
     * @param input_features
     * @param weights
     * @param grad_input
     * @param grad_weights
     * @param grad_bias
     * @param level_delta
     */
    void convolve_backward(py::array &apr_list, py::array &grad_output, py::array &input_features, py::array_t<float> &weights,
                           py::array &grad_input, py::array_t<float> &grad_weights, py::array_t<float> &grad_bias,
                           py::array_t<int32_t> &level_delta) {

        /// request buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto grad_input_buf = grad_input.request(true);
        auto grad_weights_buf = grad_weights.request(true);
        auto grad_bias_buf = grad_bias.request(true);
        auto grad_output_buf = grad_output.request();
        auto dlvl_buf = level_delta.request();

        /// check input/output data type and call templated convolution function if correct
        if(py::isinstance<py::array_t<float>>(input_features)) {
            if(!py::isinstance<py::array_t<float>>(grad_input) || !py::isinstance<py::array_t<float>>(grad_output)) {
                throw std::invalid_argument("input, grad_input and grad_output arrays passed to convolve_backward must have matching data types");
            }

            templated_convolve_backward<float>(apr_buf, input_buf, weights_buf, grad_input_buf, grad_weights_buf, grad_bias_buf, grad_output_buf, dlvl_buf);

        } else if(py::isinstance<py::array_t<double>>(input_features)) {
            if(!py::isinstance<py::array_t<double>>(grad_input) || !py::isinstance<py::array_t<double>>(grad_output)) {
                throw std::invalid_argument("input, grad_input and grad_output arrays passed to convolve_backward must have matching data types");
            }

            templated_convolve_backward<double>(apr_buf, input_buf, weights_buf, grad_input_buf, grad_weights_buf, grad_bias_buf, grad_output_buf, dlvl_buf);

        } else {
            throw std::invalid_argument("input and output arrays passed to convolve1x1 must be of type float (float32) or double (float64)");
        }
    }

    /**
     *
     * @param apr_list
     * @param input_features
     * @param weights
     * @param bias
     * @param output
     * @param level_delta
     */
    void convolve3x3(py::array &apr_list, py::array &input_features, py::array_t<float> &weights, py::array_t<float> &bias,
                     py::array &output, py::array_t<int32_t> &level_delta) {

        /// requesting buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto bias_buf = bias.request();
        auto output_buf = output.request(true);
        auto dlvl_buf = level_delta.request();

        /// check input/output data type and call templated convolution function if correct
        if(py::isinstance<py::array_t<float>>(input_features)) {
            if(!py::isinstance<py::array_t<float>>(output)) {
                throw std::invalid_argument("input and output arrays passed to convolve3x3 must have matching data types");
            }

            templated_convolve3x3<float>(apr_buf, weights_buf, bias_buf, output_buf, input_buf, dlvl_buf);

        } else if(py::isinstance<py::array_t<double>>(input_features)) {
            if(!py::isinstance<py::array_t<double>>(output)) {
                throw std::invalid_argument("input and output arrays passed to convolve3x3 must have matching data types");
            }

            templated_convolve3x3<double>(apr_buf, weights_buf, bias_buf, output_buf, input_buf, dlvl_buf);

        } else {
            throw std::invalid_argument("input and output arrays passed to convolve3x3 must be of type float (float32) or double (float64)");
        }
    }


    /**
     *
     * @param apr_list
     * @param grad_output
     * @param input_features
     * @param weights
     * @param grad_input
     * @param grad_weights
     * @param grad_bias
     * @param level_delta
     */
    void convolve3x3_backward(py::array &apr_list, py::array &grad_output, py::array &input_features, py::array_t<float> &weights,
                              py::array &grad_input, py::array_t<float> &grad_weights, py::array_t<float> &grad_bias,
                              py::array_t<int32_t> &level_delta) {

        /// request buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto grad_input_buf = grad_input.request(true);
        auto grad_weights_buf = grad_weights.request(true);
        auto grad_bias_buf = grad_bias.request(true);
        auto grad_output_buf = grad_output.request();
        auto dlvl_buf = level_delta.request();

        /// check input/output data type and call templated convolution function if correct
        if(py::isinstance<py::array_t<float>>(input_features)) {
            if(!py::isinstance<py::array_t<float>>(grad_input) || !py::isinstance<py::array_t<float>>(grad_output)) {
                throw std::invalid_argument("input, grad_input and grad_output arrays passed to convolve_backward must have matching data types");
            }

            templated_convolve3x3_backward<float>(apr_buf, input_buf, weights_buf, grad_input_buf, grad_weights_buf, grad_bias_buf, grad_output_buf, dlvl_buf);

        } else if(py::isinstance<py::array_t<double>>(input_features)) {
            if(!py::isinstance<py::array_t<double>>(grad_input) || !py::isinstance<py::array_t<double>>(grad_output)) {
                throw std::invalid_argument("input, grad_input and grad_output arrays passed to convolve_backward must have matching data types");
            }

            templated_convolve3x3_backward<double>(apr_buf, input_buf, weights_buf, grad_input_buf, grad_weights_buf, grad_bias_buf, grad_output_buf, dlvl_buf);

        } else {
            throw std::invalid_argument("input and output arrays passed to convolve1x1 must be of type float (float32) or double (float64)");
        }

    }


    /**
     *
     * @param apr_list
     * @param input_features
     * @param weights
     * @param bias
     * @param output
     * @param level_delta
     */
    void convolve1x1(py::array &apr_list, py::array &input_features, py::array_t<float> &weights, py::array_t<float> &bias,
                     py::array &output, py::array_t<int32_t> &level_delta) {

        /// requesting buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto bias_buf = bias.request();
        auto output_buf = output.request(true);
        auto dlvl_buf = level_delta.request();

        /// check input/output data type
        std::string type_string;

        if(py::isinstance<py::array_t<float_t>>(input_features)) {
            if(!py::isinstance<py::array_t<float_t>>(output)) {
                throw std::invalid_argument("input and output arrays passed to convolve1x1 must have matching data types");
            }
            type_string = "float";
        } else if(py::isinstance<py::array_t<double_t>>(input_features)) {
            if(!py::isinstance<py::array_t<double_t>>(output)) {
                throw std::invalid_argument("input and output arrays passed to convolve1x1 must have matching data types");
            }
            type_string = "double";
        } else {
            throw std::invalid_argument("input and output arrays passed to convolve1x1 must be of type float (float32) or double (float64)");
        }

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto weights_ptr = (float *) weights_buf.ptr;
        auto bias_ptr = (float *) bias_buf.ptr;
        auto dlvl_ptr = (int32_t *) dlvl_buf.ptr;
        auto output_ptr = output_buf.ptr;
        auto input_ptr = input_buf.ptr;

        /// some important numbers implicit in array shapes
        const size_t out_channels = weights_buf.shape[0];
        const size_t in_channels = weights_buf.shape[1];
        const size_t nstencils = weights_buf.shape[2];
        const size_t height = weights_buf.shape[3];
        const size_t width = weights_buf.shape[4];

        const size_t number_in_channels = input_buf.shape[1];
        const size_t nparticles = input_buf.shape[2];

        if(height !=1 || width != 1) {
            std::cerr << "This function assumes a kernel of shape (., ., ., 1, 1) but was given shape (" << height << ", " << width << ")" << std::endl;
        }

        size_t batch_size = apr_buf.size;
        size_t in, out, bn;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(in, out, bn)
#endif
        for(bn = 0; bn < batch_size; ++bn) {

            PyAPR aPyAPR = apr_ptr[bn].cast<PyAPR>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR.apr.level_max() - dlevel, aPyAPR.apr.level_min());

            std::vector<float> stencil_vec;
            stencil_vec.resize(nstencils);

            for(in=0; in<in_channels; ++in) {

                const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

                for (out = 0; out < out_channels; ++out) {

                    const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                    float b = bias_ptr[out];

                    size_t offset = out * in_channels * nstencils + in * nstencils;

                    for(int n=0; n<nstencils; ++n) {

                        stencil_vec[n] = weights_ptr[offset + n];

                    }
                    if(type_string == "float"){

                        PyAPRFiltering::convolve_1x1<float>(aPyAPR.apr, (float *) input_ptr, stencil_vec, b, out, in,
                                                            current_max_level, in_offset, number_in_channels,
                                                            out_offset, (float *) output_ptr);

                    } else if(type_string == "double"){

                        PyAPRFiltering::convolve_1x1<double>(aPyAPR.apr, (double *) input_ptr, stencil_vec, b, out, in,
                                                             current_max_level, in_offset, number_in_channels,
                                                             out_offset, (double *) output_ptr);
                    }
                }
            }
        }
    }

    /**
     *
     * @param apr_list
     * @param grad_output
     * @param input_features
     * @param weights
     * @param grad_input
     * @param grad_weights
     * @param grad_bias
     * @param level_delta
     */
    void convolve1x1_backward(py::array &apr_list, py::array &grad_output, py::array &input_features,
                              py::array_t<float> &weights, py::array &grad_input, py::array_t<float> &grad_weights,
                              py::array_t<float> &grad_bias, py::array_t<int32_t> &level_delta) {

        /// request buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto grad_input_buf = grad_input.request(true);
        auto grad_weights_buf = grad_weights.request(true);
        auto grad_bias_buf = grad_bias.request(true);
        auto grad_output_buf = grad_output.request();
        auto dlvl_buf = level_delta.request();

        /// check input/output data type
        std::string type_string;

        if(py::isinstance<py::array_t<float_t>>(input_features)) {
            if(!py::isinstance<py::array_t<float_t>>(grad_output) || !py::isinstance<py::array_t<float_t>>(grad_input)) {
                throw std::invalid_argument("input, grad_input and grad_output arrays passed to convolve1x1_backward must have matching data types");
            }
            type_string = "float";
        } else if(py::isinstance<py::array_t<double_t>>(input_features)) {
            if(!py::isinstance<py::array_t<double_t>>(grad_output) || !py::isinstance<py::array_t<double_t>>(grad_input)) {
                throw std::invalid_argument("input, grad_input and grad_output arrays passed to convolve1x1_backward must have matching data types");
            }
            type_string = "double";
        } else {
            throw std::invalid_argument("input, grad_input and grad_output arrays passed to convolve1x1_backward must be of type float (float32) or double (float64)");
        }

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto weights_ptr = (float *) weights_buf.ptr;
        auto grad_weights_ptr = (float *) grad_weights_buf.ptr;
        auto grad_bias_ptr = (float *) grad_bias_buf.ptr;
        auto dlvl_ptr = (int32_t *) dlvl_buf.ptr;
        auto input_ptr = input_buf.ptr;
        auto grad_input_ptr = grad_input_buf.ptr;
        auto grad_output_ptr = grad_output_buf.ptr;

        /// some important numbers implicit in array shapes
        const size_t out_channels = weights_buf.shape[0];
        const size_t in_channels = weights_buf.shape[1];
        const size_t nstencils = weights_buf.shape[2];
        const size_t height = weights_buf.shape[3];
        const size_t width = weights_buf.shape[4];

        /// allocate temporary arrays to avoid race condition on the weight and bias gradients
#ifdef PYAPR_HAVE_OPENMP
        size_t num_threads = omp_get_max_threads();
#else
        size_t num_threads = 1;
#endif
        std::vector<float> temp_vec_dw(num_threads*out_channels*in_channels*nstencils*1, 0.0f);
        std::vector<float> temp_vec_db(out_channels*num_threads, 0.0f);

        const size_t number_in_channels = input_buf.shape[1];
        const size_t nparticles = input_buf.shape[2];

        if(height !=1 || width != 1) {
            std::cerr << "This function assumes a kernel of shape (1, 1) but was given shape (" << height << ", " << width << ")" << std::endl;
        }

        size_t batch_size = apr_buf.size;
        size_t in, out, bn;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(in, out, bn)
#endif
        for(bn = 0; bn < batch_size; ++bn) {

#ifdef PYAPR_HAVE_OPENMP
            const size_t thread_id = omp_get_thread_num();
#else
            const size_t thread_id = 0;
#endif
            PyAPR aPyAPR = apr_ptr[bn].cast<PyAPR>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR.apr.level_max() - dlevel, aPyAPR.apr.level_min());

            for(in=0; in<in_channels; ++in) {

                const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

                for (out = 0; out < out_channels; ++out) {

                    const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                    const uint64_t dw_offset = thread_id * out_channels * in_channels *  nstencils +
                                               out * in_channels * nstencils +
                                               in * nstencils;

                    const size_t db_offset = thread_id * out_channels;

                    std::vector<float> stencil_vec;
                    stencil_vec.resize(nstencils);

                    uint64_t offset = out * in_channels * nstencils + in * nstencils;

                    for(size_t n=0; n<nstencils; ++n) {

                        stencil_vec[n] = weights_ptr[offset + n];

                    }
                    if(type_string == "float") {
                        PyAPRFiltering::convolve_1x1_backward<float>(aPyAPR.apr, (float *) input_ptr,
                                                                     in_offset, out_offset, stencil_vec, (float *) grad_output_ptr,
                                                                     (float *) grad_input_ptr, temp_vec_db, temp_vec_dw, dw_offset,
                                                                     db_offset, out, in, current_max_level);
                    } else if(type_string == "double") {
                        PyAPRFiltering::convolve_1x1_backward<double>(aPyAPR.apr, (double *) input_ptr,
                                                                     in_offset, out_offset, stencil_vec, (double *) grad_output_ptr,
                                                                     (double *) grad_input_ptr, temp_vec_db, temp_vec_dw, dw_offset,
                                                                     db_offset, out, in, current_max_level);
                    }
                }//out
            }//in
        }//bn

        /// collect weight gradients from temp_vec_dw
        const uint64_t num_weights = out_channels * in_channels * nstencils;
        unsigned int w;
        size_t thd;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) private(w, thd)
#endif
        for(w = 0; w < num_weights; ++w) {
            float val = 0;
            for(thd = 0; thd < num_threads; ++thd) {
                val += temp_vec_dw[thd*num_weights + w];
            }
            grad_weights_ptr[w] = val / batch_size;
        }

        /// collect bias gradients from temp_vec_db

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) private(w, thd)
#endif
        for(w = 0; w < out_channels; ++w) {
            float val = 0;
            for(thd = 0; thd < num_threads; ++thd) {
                val += temp_vec_db[thd*out_channels + w];
            }
            grad_bias_ptr[w] = val / batch_size;
        }
    }

    /**
     *
     * @param apr_list
     * @param input_features
     * @param output
     * @param level_delta
     * @param index_arr
     */
    void max_pool(py::array &apr_list, py::array &input_features, py::array &output,
                  py::array_t<int32_t> &level_delta, py::array_t<uint64_t> &index_arr) {

        /// request buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto output_buf = output.request(true);
        auto dlvl_buf = level_delta.request();
        auto idx_buf = index_arr.request();

        /// check input/output data type
        std::string type_string;

        if(py::isinstance<py::array_t<float>>(input_features)) {
            if(!py::isinstance<py::array_t<float>>(output)) {
                throw std::invalid_argument("input and output arrays passed to max_pool must have matching data types");
            }
            type_string = "float";
        } else if(py::isinstance<py::array_t<double>>(input_features)) {
            if(!py::isinstance<py::array_t<double>>(output)) {
                throw std::invalid_argument("input and output arrays passed to max_pool must have matching data types");
            }
            type_string = "double";
        } else {
            throw std::invalid_argument("input, grad_input and grad_output arrays passed to convolve1x1_backward must be of type float (float32) or double (float64)");
        }

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto input_ptr = input_buf.ptr;
        auto output_ptr = output_buf.ptr;
        auto dlvl_ptr = (int32_t *) dlvl_buf.ptr;
        auto idx_ptr = (uint64_t *) idx_buf.ptr;

        /// some important numbers implicit in array shapes
        const size_t batch_size = input_buf.shape[0];
        const size_t number_channels = input_buf.shape[1];
        const size_t nparticles_in = input_buf.shape[2];
        const size_t nparticles_out = output_buf.shape[2];

        size_t bn, channel;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(bn, channel)
#endif
        for(bn = 0; bn < batch_size; ++bn) {

            PyAPR aPyAPR = apr_ptr[bn].cast<PyAPR>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR.apr.level_max() - dlevel, aPyAPR.apr.level_min());

            auto apr_iterator = aPyAPR.apr.iterator();
            auto tree_iterator = aPyAPR.apr.tree_iterator();

            const int64_t tree_offset_in  = PyAPRFiltering::compute_tree_offset(apr_iterator, tree_iterator, current_max_level);
            const int64_t tree_offset_out = PyAPRFiltering::compute_tree_offset(apr_iterator, tree_iterator, current_max_level-1);

            for (channel = 0; channel < number_channels; ++channel) {

                const uint64_t in_offset = bn * number_channels * nparticles_in + channel * nparticles_in;
                const uint64_t out_offset = bn * number_channels * nparticles_out + channel * nparticles_out;

                if(type_string == "float") {
                    PyAPRFiltering::max_pool_2x2<float>(aPyAPR.apr, (float*)input_ptr, (float*)output_ptr, idx_ptr, in_offset,
                            out_offset, tree_offset_in, tree_offset_out, current_max_level);
                } else if(type_string == "double") {
                    PyAPRFiltering::max_pool_2x2<double>(aPyAPR.apr, (double*)input_ptr, (double*)output_ptr, idx_ptr, in_offset,
                            out_offset, tree_offset_in, tree_offset_out, current_max_level);
                }
            }
        }
    }


    /**
     *
     * @param grad_output
     * @param grad_input
     * @param max_indices
     */
    void max_pool_backward(py::array &grad_output, py::array &grad_input, py::array_t<int64_t> &max_indices) {

        //APRTimer timer(false);
        //timer.start_timer("max pool backward");

        auto grad_output_buf = grad_output.request();
        auto grad_input_buf = grad_input.request(true);
        auto index_buf = max_indices.request();

        /// check input/output data type
        std::string type_string;

        if(py::isinstance<py::array_t<float>>(grad_input)) {
            if(!py::isinstance<py::array_t<float>>(grad_output)) {
                throw std::invalid_argument("grad_input and grad_output arrays passed to max_pool_backward must have matching data types");
            }
            type_string = "float";
        } else if(py::isinstance<py::array_t<double>>(grad_input)) {
            if(!py::isinstance<py::array_t<double>>(grad_output)) {
                throw std::invalid_argument("grad_input and grad_output arrays passed to max_pool_backward must have matching data types");
            }
            type_string = "double";
        } else {
            throw std::invalid_argument("input, grad_input and grad_output arrays passed to convolve1x1_backward must be of type float (float32) or double (float64)");
        }

        /// request pointers to data
        auto grad_output_ptr = grad_output_buf.ptr;
        auto grad_input_ptr = grad_input_buf.ptr;
        auto index_ptr = (uint64_t *) index_buf.ptr;

        if(type_string == "float"){
            PyAPRFiltering::max_pool_backward_stored_indices<float>((float*)grad_input_ptr, (float*)grad_output_ptr,
                                                                    index_ptr, grad_input_buf.size, index_buf.size);

        } else if(type_string == "double") {
            PyAPRFiltering::max_pool_backward_stored_indices<double>((double*)grad_input_ptr, (double*)grad_output_ptr,
                                                                    index_ptr, grad_input_buf.size, index_buf.size);
        }
    }


    void transposed_conv_2x2(py::array &apr_list, py::array &input_features, py::array_t<float> &weights_2x2,
                             py::array_t<float> &weights_1x1, py::array_t<float> &bias, py::array &output,
                             py::array_t<int32_t> &level_delta) {

        /// requesting buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_2x2_buf = weights_2x2.request();
        auto weights_1x1_buf = weights_1x1.request();
        auto bias_buf = bias.request();
        auto output_buf = output.request(true);
        auto dlvl_buf = level_delta.request();

        /// check input/output data type and call templated convolution function if correct
        if(py::isinstance<py::array_t<float>>(input_features)) {
            if(!py::isinstance<py::array_t<float>>(output)) {
                throw std::invalid_argument("input and output arrays passed to transposed_conv_2x2 must have matching data types");
            }

            templated_transposed_conv_2x2<float>(apr_buf, input_buf, weights_2x2_buf, weights_1x1_buf, bias_buf, output_buf, dlvl_buf);

        } else if(py::isinstance<py::array_t<double>>(input_features)) {
            if(!py::isinstance<py::array_t<double>>(output)) {
                throw std::invalid_argument("input and output arrays passed to transposed_conv_2x2 must have matching data types");
            }

            templated_transposed_conv_2x2<double>(apr_buf, input_buf, weights_2x2_buf, weights_1x1_buf, bias_buf, output_buf, dlvl_buf);

        } else {
            throw std::invalid_argument("input and output arrays passed to transposed_conv_2x2 must be of type float (float32) or double (float64)");
        }
    }


    void transposed_conv_2x2_backward(py::array &apr_list, py::array &input_features, py::array_t<float> &weights_2x2,
                                      py::array_t<float> &weights_1x1, py::array &grad_input, py::array_t<float>& grad_weights_2x2,
                                      py::array_t<float>& grad_weights_1x1, py::array_t<float> &grad_bias, py::array &grad_output,
                                      py::array_t<int32_t> &level_delta) {

        /// requesting buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_2x2_buf = weights_2x2.request();
        auto weights_1x1_buf = weights_1x1.request();
        auto grad_input_buf = grad_input.request();
        auto grad_weights_2x2_buf = grad_weights_2x2.request();
        auto grad_weights_1x1_buf = grad_weights_1x1.request();
        auto grad_bias_buf = grad_bias.request();
        auto grad_output_buf = grad_output.request(true);
        auto dlvl_buf = level_delta.request();

        /// check input/output data type and call templated convolution function if correct
        if(py::isinstance<py::array_t<float>>(input_features)) {
            if(!py::isinstance<py::array_t<float>>(grad_input) || !py::isinstance<py::array_t<float>>(grad_output)) {
                throw std::invalid_argument("input, grad_input and grad_output arrays passed to transposed_conv_2x2_backward must have matching data types");
            }

            templated_transposed_conv_2x2_backward<float>(apr_buf, input_buf, weights_2x2_buf, weights_1x1_buf, grad_input_buf,
                                       grad_weights_2x2_buf, grad_weights_1x1_buf, grad_bias_buf, grad_output_buf, dlvl_buf);


        } else if(py::isinstance<py::array_t<double>>(input_features)) {
            if(!py::isinstance<py::array_t<double>>(grad_input) || !py::isinstance<py::array_t<double>>(grad_output)) {
                throw std::invalid_argument("input, grad_input and grad_output arrays passed to transposed_conv_2x2_backward must have matching data types");
            }

            templated_transposed_conv_2x2_backward<double>(apr_buf, input_buf, weights_2x2_buf, weights_1x1_buf, grad_input_buf,
                                                          grad_weights_2x2_buf, grad_weights_1x1_buf, grad_bias_buf, grad_output_buf, dlvl_buf);
        } else {
            throw std::invalid_argument("input and output arrays passed to transposed_conv_2x2_backward must be of type float (float32) or double (float64)");
        }
    }
}


template<typename T>
void APRNetOps::templated_convolve(py::buffer_info &apr_buf, py::buffer_info &input_buf, py::buffer_info &weights_buf,
                                   py::buffer_info &bias_buf, py::buffer_info &output_buf, py::buffer_info &dlvl_buf) {

    /// pointers to python array data
    auto apr_ptr = (py::object *) apr_buf.ptr;
    auto weights_ptr = (float *) weights_buf.ptr;
    auto bias_ptr = (float *) bias_buf.ptr;
    auto output_ptr = (T *) output_buf.ptr;
    auto input_ptr = (T *) input_buf.ptr;
    auto dlvl_ptr = (int32_t *) dlvl_buf.ptr;

    /// some important numbers implicit in array shapes
    const size_t out_channels = weights_buf.shape[0];
    const size_t in_channels = weights_buf.shape[1];
    const size_t nstencils = weights_buf.shape[2];
    const size_t height = weights_buf.shape[3];
    const size_t width = weights_buf.shape[4];

    const size_t number_in_channels = input_buf.shape[1];
    const size_t nparticles = input_buf.shape[2];

    size_t batch_size = apr_buf.size;
    size_t in, out, bn;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(in, out, bn)
#endif
    for(bn = 0; bn < batch_size; ++bn) {

        PyAPR aPyAPR = apr_ptr[bn].cast<PyAPR>();

        int dlevel = dlvl_ptr[bn];
        const unsigned int current_max_level = std::max(aPyAPR.apr.level_max() - dlevel, aPyAPR.apr.level_min());

        for(in=0; in<in_channels; ++in) {

            const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

            /**** initialize and fill the apr tree ****/
            ParticleData<T> tree_data;

            PyAPRFiltering::fill_tree_mean<T>(aPyAPR.apr, input_ptr, tree_data, in_offset, current_max_level);

            for (out = 0; out < out_channels; ++out) {

                const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                T b = (T) bias_ptr[out];

                std::vector<PixelData<T>> stencil_vec;
                stencil_vec.resize(nstencils);

                for(int n=0; n<nstencils; ++n) {

                    stencil_vec[n].init(height, width, 1);

                    size_t offset = out * in_channels * nstencils * height * width +
                                    in * nstencils * height * width +
                                    n * height * width;

                    for(int idx=0; idx < width*height; ++idx) {
                        stencil_vec[n].mesh[idx] = (T) weights_ptr[offset + idx];
                    }
                }

                PyAPRFiltering::convolve_generic<T>(aPyAPR.apr, input_ptr, stencil_vec, b, out, in, current_max_level,
                                                    in_offset, tree_data, number_in_channels, out_offset, output_ptr);
            }
        }
    }
}


template<typename T>
void APRNetOps::templated_convolve_backward(py::buffer_info &apr_buf, py::buffer_info &input_buf, py::buffer_info &weights_buf,
                                            py::buffer_info &grad_input_buf, py::buffer_info &grad_weights_buf,
                                            py::buffer_info &grad_bias_buf, py::buffer_info &grad_output_buf, py::buffer_info &dlvl_buf) {
    /// pointers to python array data
    auto apr_ptr = (py::object *) apr_buf.ptr;
    auto input_ptr = (T *) input_buf.ptr;
    auto weights_ptr = (float *) weights_buf.ptr;
    auto grad_input_ptr = (T *) grad_input_buf.ptr;
    auto grad_weights_ptr = (float *) grad_weights_buf.ptr;
    auto grad_bias_ptr = (float *) grad_bias_buf.ptr;
    auto grad_output_ptr = (T *) grad_output_buf.ptr;
    auto dlvl_ptr = (int32_t *) dlvl_buf.ptr;

    /// some important numbers implicit in array shapes
    const size_t out_channels = weights_buf.shape[0];
    const size_t in_channels = weights_buf.shape[1];
    const size_t nstencils = weights_buf.shape[2];
    const size_t height = weights_buf.shape[3];
    const size_t width = weights_buf.shape[4];

    /// allocate temporary arrays to avoid race condition on the weight and bias gradients
#ifdef PYAPR_HAVE_OPENMP
    size_t num_threads = omp_get_max_threads();
#else
    size_t num_threads = 1;
#endif
    std::vector<T> temp_vec_dw(num_threads*out_channels*in_channels*nstencils*height*width, 0);
    std::vector<float> temp_vec_db(out_channels*num_threads, 0.0f);

    const size_t number_in_channels = input_buf.shape[1];
    const size_t nparticles = input_buf.shape[2];

    size_t batch_size = apr_buf.size;
    size_t in, out, bn;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(in, out, bn)
#endif
    for(bn = 0; bn < batch_size; ++bn) {

#ifdef PYAPR_HAVE_OPENMP
        const size_t thread_id = omp_get_thread_num();
#else
        const size_t thread_id = 0;
#endif

        PyAPR aPyAPR = apr_ptr[bn].cast<PyAPR>();

        int dlevel = dlvl_ptr[bn];
        const unsigned int current_max_level = std::max(aPyAPR.apr.level_max() - dlevel, aPyAPR.apr.level_min());

        for(in=0; in<in_channels; ++in) {

            const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

            /// initialize and fill the apr tree

            ParticleData<T> tree_data;
            PyAPRFiltering::fill_tree_mean<T>(aPyAPR.apr, input_ptr, tree_data, in_offset, current_max_level);

            ParticleData<T> grad_tree_temp;
            grad_tree_temp.data.resize(tree_data.data.size(), 0.0f);

            for (out = 0; out < out_channels; ++out) {

                const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                const uint64_t dw_offset = thread_id * out_channels * in_channels *  nstencils * height * width +
                                           out * in_channels * nstencils * height * width +
                                           in * nstencils * height * width;

                const size_t db_offset = thread_id * out_channels;

                std::vector<PixelData<T>> stencil_vec;
                stencil_vec.resize(nstencils);

                for(size_t n=0; n<nstencils; ++n) {

                    stencil_vec[n].init(height, width, 1);

                    uint64_t offset = out * in_channels * nstencils * height * width +
                                      in * nstencils * height * width +
                                      n * height * width;

                    for(size_t idx = 0; idx < height * width; ++idx) {
                        stencil_vec[n].mesh[idx] = (T) weights_ptr[offset + idx];
                    }

                }
                PyAPRFiltering::convolve_generic_backward<T>(aPyAPR.apr, input_ptr,
                                                             in_offset, out_offset, stencil_vec, grad_output_ptr,
                                                             grad_input_ptr, temp_vec_db, temp_vec_dw, dw_offset,
                                                             db_offset, grad_tree_temp, tree_data, out, in,
                                                             current_max_level);
            }

            /// push grad_tree_temp to grad_input
            PyAPRFiltering::fill_tree_mean_backward<T>(aPyAPR.apr, grad_input_ptr, grad_tree_temp, in_offset, current_max_level);
        }

    }

    /// collect weight gradients from temp_vec_dw
    const uint64_t num_weights = out_channels * in_channels * nstencils * height * width;
    unsigned int w;
    size_t thd;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) private(w, thd)
#endif
    for(w = 0; w < num_weights; ++w) {
        T val = 0;
        for(thd = 0; thd < num_threads; ++thd) {
            val += temp_vec_dw[thd*num_weights + w];
        }
        grad_weights_ptr[w] = (float) (val / batch_size);
    }

    /// collect bias gradients from temp_vec_db

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) private(w, thd)
#endif
    for(w = 0; w < out_channels; ++w) {
        float val = 0;
        for(thd = 0; thd < num_threads; ++thd) {
            val += temp_vec_db[thd*out_channels + w];
        }
        grad_bias_ptr[w] = val / batch_size;
    }
}


template<typename T>
void APRNetOps::templated_convolve3x3(py::buffer_info &apr_buf, py::buffer_info &weights_buf, py::buffer_info &bias_buf,
                                      py::buffer_info &output_buf, py::buffer_info &input_buf, py::buffer_info &dlvl_buf) {
    /// pointers to python array data
    auto apr_ptr = (py::object *) apr_buf.ptr;
    auto weights_ptr = (float *) weights_buf.ptr;
    auto bias_ptr = (float *) bias_buf.ptr;
    auto output_ptr = (T *) output_buf.ptr;
    auto input_ptr = (T *) input_buf.ptr;
    auto dlvl_ptr = (int32_t *) dlvl_buf.ptr;

    /// some important numbers implicit in array shapes
    const size_t out_channels = weights_buf.shape[0];
    const size_t in_channels = weights_buf.shape[1];
    const size_t nstencils = weights_buf.shape[2];
    const size_t height = weights_buf.shape[3];
    const size_t width = weights_buf.shape[4];

    const size_t number_in_channels = input_buf.shape[1];
    const size_t nparticles = input_buf.shape[2];

    if(height !=3 || width != 3) {
        std::cerr << "This function assumes a kernel of shape (3, 3) but was given shape (" << height << ", " << width << ")" << std::endl;
    }

    size_t batch_size = apr_buf.size;
    size_t in, out, bn;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(in, out, bn)
#endif
    for(bn = 0; bn < batch_size; ++bn) {

        PyAPR aPyAPR = apr_ptr[bn].cast<PyAPR>();

        int dlevel = dlvl_ptr[bn];
        const unsigned int current_max_level = std::max(aPyAPR.level_max() - dlevel, aPyAPR.level_min());

        for(in=0; in<in_channels; ++in) {

            const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

            /**** initialize and fill the apr tree ****/

            ParticleData<T> tree_data;

            PyAPRFiltering::fill_tree_mean<T>(aPyAPR.apr, input_ptr, tree_data, in_offset, current_max_level);

            for (out = 0; out < out_channels; ++out) {

                const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                T b = bias_ptr[out];

                std::vector<PixelData<T>> stencil_vec;
                stencil_vec.resize(nstencils);

                for(int n=0; n<nstencils; ++n) {

                    stencil_vec[n].init(height, width, 1);

                    size_t offset = out * in_channels * nstencils * 9 + in * nstencils * 9 + n * 9;

                    for(int idx=0; idx < width*height; ++idx) {
                        stencil_vec[n].mesh[idx] = weights_ptr[offset + idx];
                    }
                }

                PyAPRFiltering::convolve_3x3<T>(aPyAPR.apr, input_ptr, stencil_vec, b, out, in, current_max_level,
                                                in_offset, tree_data, number_in_channels, out_offset, output_ptr);
            }
        }
    }
}


template<typename T>
void APRNetOps::templated_convolve3x3_backward(py::buffer_info &apr_buf, py::buffer_info &input_buf, py::buffer_info &weights_buf,
                                               py::buffer_info &grad_input_buf, py::buffer_info &grad_weights_buf,
                                               py::buffer_info &grad_bias_buf, py::buffer_info &grad_output_buf,
                                               py::buffer_info &dlvl_buf) {

    /// pointers to python array data
    auto apr_ptr = (py::object *) apr_buf.ptr;
    auto input_ptr = (T *) input_buf.ptr;
    auto weights_ptr = (float *) weights_buf.ptr;
    auto grad_input_ptr = (T *) grad_input_buf.ptr;
    auto grad_weights_ptr = (float *) grad_weights_buf.ptr;
    auto grad_bias_ptr = (float *) grad_bias_buf.ptr;
    auto grad_output_ptr = (T *) grad_output_buf.ptr;
    auto dlvl_ptr = (int32_t *) dlvl_buf.ptr;

    /// some important numbers implicit in array shapes
    const size_t out_channels = weights_buf.shape[0];
    const size_t in_channels = weights_buf.shape[1];
    const size_t nstencils = weights_buf.shape[2];
    //const size_t height = weights_buf.shape[3];
    //const size_t width = weights_buf.shape[4];

    /// allocate temporary arrays to avoid race condition on the weight and bias gradients
#ifdef PYAPR_HAVE_OPENMP
    size_t num_threads = omp_get_max_threads();
#else
    size_t num_threads = 1;
#endif
    std::vector<T> temp_vec_dw(num_threads*out_channels*in_channels*nstencils*9, 0.0f);
    std::vector<float> temp_vec_db(out_channels*num_threads, 0.0f);

    const size_t number_in_channels = input_buf.shape[1];
    const size_t nparticles = input_buf.shape[2];

    size_t batch_size = apr_buf.size;
    size_t in, out, bn;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(in, out, bn)
#endif
    for(bn = 0; bn < batch_size; ++bn) {

#ifdef PYAPR_HAVE_OPENMP
        const size_t thread_id = omp_get_thread_num();
#else
        const size_t thread_id = 0;
#endif
        PyAPR aPyAPR = apr_ptr[bn].cast<PyAPR>();

        int dlevel = dlvl_ptr[bn];
        const unsigned int current_max_level = std::max(aPyAPR.apr.level_max() - dlevel, aPyAPR.apr.level_min());

        for(in=0; in<in_channels; ++in) {

            const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

            /// initialize and fill the apr tree

            ParticleData<T> tree_data;
            PyAPRFiltering::fill_tree_mean<T>(aPyAPR.apr, input_ptr, tree_data, in_offset, current_max_level);

            ParticleData<T> grad_tree_temp;
            grad_tree_temp.data.resize(tree_data.data.size(), 0);

            for (out = 0; out < out_channels; ++out) {

                const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                const uint64_t dw_offset = thread_id * out_channels * in_channels *  nstencils * 9 +
                                           out * in_channels * nstencils * 9 +
                                           in * nstencils * 9;

                const size_t db_offset = thread_id * out_channels;

                std::vector<PixelData<T>> stencil_vec;
                stencil_vec.resize(nstencils);

                for(size_t n=0; n<nstencils; ++n) {

                    stencil_vec[n].init(3, 3, 1);

                    uint64_t offset = out * in_channels * nstencils * 9 + in * nstencils * 9 + n * 9;

                    for(size_t idx = 0; idx < 9; ++idx) {
                        stencil_vec[n].mesh[idx] = (T) weights_ptr[offset + idx];
                    }

                }
                PyAPRFiltering::convolve_3x3_backward<T>(aPyAPR.apr, input_ptr,
                                                         in_offset, out_offset, stencil_vec, grad_output_ptr,
                                                         grad_input_ptr, temp_vec_db, temp_vec_dw, dw_offset,
                                                         db_offset, grad_tree_temp, tree_data, out, in,
                                                         current_max_level);
            }

            /// push grad_tree_temp to grad_input
            PyAPRFiltering::fill_tree_mean_backward<T>(aPyAPR.apr, grad_input_ptr,
                                                       grad_tree_temp, in_offset, current_max_level);
        }
    }

    /// collect weight gradients from temp_vec_dw
    const uint64_t num_weights = out_channels * in_channels * nstencils * 9;
    unsigned int w;
    size_t thd;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) private(w, thd)
#endif
    for(w = 0; w < num_weights; ++w) {
        T val = 0;
        for(thd = 0; thd < num_threads; ++thd) {
            val += temp_vec_dw[thd*num_weights + w];
        }
        grad_weights_ptr[w] = (float) (val / batch_size);
    }

    /// collect bias gradients from temp_vec_db
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) private(w, thd)
#endif
    for(w = 0; w < out_channels; ++w) {
        float val = 0;
        for(thd = 0; thd < num_threads; ++thd) {
            val += temp_vec_db[thd*out_channels + w];
        }
        grad_bias_ptr[w] = val / batch_size;
    }
}


template<typename T>
void APRNetOps::templated_transposed_conv_2x2(py::buffer_info &apr_buf, py::buffer_info &input_buf, py::buffer_info &weights_2x2_buf,
                                              py::buffer_info &weights_1x1_buf, py::buffer_info &bias_buf, py::buffer_info &output_buf,
                                              py::buffer_info &dlvl_buf) {

    /// pointers to python array data
    auto apr_ptr = (py::object *) apr_buf.ptr;
    auto weights_2x2_ptr = (float *) weights_2x2_buf.ptr;
    auto weights_1x1_ptr = (float *) weights_1x1_buf.ptr;
    auto bias_ptr = (float *) bias_buf.ptr;
    auto output_ptr = (T *) output_buf.ptr;
    auto input_ptr = (T *) input_buf.ptr;
    auto dlvl_ptr = (int32_t *) dlvl_buf.ptr;

    /// some important numbers implicit in array shapes
    const int out_channels = weights_2x2_buf.shape[0];
    const int in_channels = weights_2x2_buf.shape[1];
    const int height = weights_2x2_buf.shape[2];
    const int width = weights_2x2_buf.shape[3];

    const size_t nparticles = input_buf.shape[2];

    if(height !=2 || width != 2) {
        std::cerr << "This function assumes a kernel of shape (2, 2) but was given shape (" << height << ", " << width << ")" << std::endl;
        return;
    }

    const int batch_size = apr_buf.size;
    int in, out, bn;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(in, out, bn)
#endif
    for(bn = 0; bn < batch_size; ++bn) {

        PyAPR aPyAPR = apr_ptr[bn].cast<PyAPR>();

        int dlevel = dlvl_ptr[bn];

        if( dlevel == 0 ) {
            throw std::invalid_argument("Received level_delta = 0 in transposed_conv_2x2, but currently cannot go beyond the APR max level");
        }

        const unsigned int current_max_level = std::max(aPyAPR.level_max() - dlevel, aPyAPR.level_min());

        auto apr_it = aPyAPR.apr.iterator();
        auto tree_it = aPyAPR.apr.tree_iterator();

        const int64_t tree_offset_in = PyAPRFiltering::compute_tree_offset(apr_it, tree_it, current_max_level);
        const int64_t tree_offset_out = PyAPRFiltering::compute_tree_offset(apr_it, tree_it, current_max_level + 1);

        for(in=0; in<in_channels; ++in) {

            const uint64_t in_offset = bn * in_channels * nparticles + in * nparticles;

            /**** initialize and fill the apr tree ****/

            for (out = 0; out < out_channels; ++out) {

                const T b = (in == (in_channels-1)) ? bias_ptr[out] : 0;

                const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                PixelData<T> stencil_2x2(2, 2, 1);

                size_t offset = out * in_channels * 4 + in * 4;

                for(int idx=0; idx < 4; ++idx) {
                    stencil_2x2.mesh[idx] = weights_2x2_ptr[offset + idx];
                }

                const T weight_1x1 = weights_1x1_ptr[out * in_channels + in];

                PyAPRFiltering::transposed_conv_2x2<T>(aPyAPR.apr, input_ptr, output_ptr, stencil_2x2, weight_1x1,
                        b, in_offset, out_offset, tree_offset_in, tree_offset_out, current_max_level);

            }
        }
    }
}


template<typename T>
void APRNetOps::templated_transposed_conv_2x2_backward(py::buffer_info &apr_buf, py::buffer_info &input_buf,
                                                       py::buffer_info &weights_2x2_buf, py::buffer_info &weights_1x1_buf,
                                                       py::buffer_info &grad_input_buf, py::buffer_info &grad_weights_2x2_buf,
                                                       py::buffer_info &grad_weights_1x1_buf, py::buffer_info &grad_bias_buf,
                                                       py::buffer_info &grad_output_buf, py::buffer_info &dlvl_buf){

    /// pointers to python array data
    auto apr_ptr = (py::object *) apr_buf.ptr;
    auto input_ptr = (T *) input_buf.ptr;
    auto weights_2x2_ptr = (float *) weights_2x2_buf.ptr;
    auto weights_1x1_ptr = (float *) weights_1x1_buf.ptr;
    auto grad_input_ptr = (T *) grad_input_buf.ptr;
    auto grad_weights_2x2_ptr = (float *) grad_weights_2x2_buf.ptr;
    auto grad_weights_1x1_ptr = (float *) grad_weights_1x1_buf.ptr;
    auto grad_bias_ptr = (float *) grad_bias_buf.ptr;
    auto grad_output_ptr = (T *) grad_output_buf.ptr;
    auto dlvl_ptr = (int32_t *) dlvl_buf.ptr;

    /// some important numbers implicit in array shapes
    const int out_channels = weights_2x2_buf.shape[0];
    const int in_channels = weights_2x2_buf.shape[1];
    //const int height = weights_2x2_buf.shape[2];
    //const int width = weights_2x2_buf.shape[3];

    const size_t nparticles = input_buf.shape[2];

    /// allocate temporary arrays to avoid race condition on the weight and bias gradients
#ifdef PYAPR_HAVE_OPENMP
    const size_t num_threads = omp_get_max_threads();
#else
    const size_t num_threads = 1;
#endif

    std::vector<float> temp_vec_dw_2x2(num_threads*out_channels*in_channels*4, 0.0f);
    std::vector<float> temp_vec_dw_1x1(num_threads*out_channels*in_channels, 0.0f);
    std::vector<float> temp_vec_db(out_channels*num_threads, 0.0f);

    const int batch_size = apr_buf.size;
    int in, out, bn;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(in, out, bn)
#endif
    for(bn = 0; bn < batch_size; ++bn) {

#ifdef PYAPR_HAVE_OPENMP
        const size_t thread_id = omp_get_thread_num();
#else
        const size_t thread_id = 0;
#endif
        PyAPR aPyAPR = apr_ptr[bn].cast<PyAPR>();

        int dlevel = dlvl_ptr[bn];
        const unsigned int current_max_level = std::max(aPyAPR.apr.level_max() - dlevel, aPyAPR.apr.level_min());

        auto apr_it = aPyAPR.apr.iterator();
        auto tree_it = aPyAPR.apr.tree_iterator();

        const int64_t tree_offset_in = PyAPRFiltering::compute_tree_offset(apr_it, tree_it, current_max_level);
        const int64_t tree_offset_out = PyAPRFiltering::compute_tree_offset(apr_it, tree_it, current_max_level + 1);

        for(in=0; in<in_channels; ++in) {

            const uint64_t in_offset = bn * in_channels * nparticles + in * nparticles;

            for (out = 0; out < out_channels; ++out) {

                const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                const size_t dw_2x2_offset = thread_id * out_channels * in_channels * 4 +
                                             out * in_channels * 4 +
                                             in * 4;

                const size_t dw_1x1_offset = thread_id * out_channels * in_channels + out * in_channels + in;

                const size_t db_offset = thread_id * out_channels + out;

                PixelData<T> stencil_2x2(2, 2, 1);

                size_t offset = out * in_channels * 4 + in * 4;

                for(int idx=0; idx < 4; ++idx) {
                    stencil_2x2.mesh[idx] = weights_2x2_ptr[offset + idx];
                }

                const T weight_1x1 = weights_1x1_ptr[out * in_channels + in];

                PyAPRFiltering::transposed_conv_2x2_backward(aPyAPR.apr, input_ptr, stencil_2x2, weight_1x1, grad_input_ptr,
                                                             temp_vec_dw_2x2, temp_vec_dw_1x1, temp_vec_db, grad_output_ptr,
                                                             dw_2x2_offset, dw_1x1_offset, db_offset, in_offset, out_offset,
                                                             tree_offset_in, tree_offset_out, current_max_level);
            }
        }
    }

    /// collect weight gradients from temp_vec_dw
    const int num_weights_2x2 = out_channels * in_channels * 4;
    int w, thd;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) private(w, thd)
#endif
    for(w = 0; w < num_weights_2x2; ++w) {
        float val = 0;
        for(thd = 0; thd < num_threads; ++thd) {
            val += temp_vec_dw_2x2[thd*num_weights_2x2 + w];
        }
        grad_weights_2x2_ptr[w] = val / batch_size;
    }

    const int num_weights_1x1 = out_channels * in_channels;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) private(w, thd)
#endif
    for(w = 0; w < num_weights_1x1; ++w) {
        float val = 0;
        for(thd = 0; thd < num_threads; ++thd) {
            val += temp_vec_dw_1x1[thd*num_weights_1x1 + w];
        }
        grad_weights_1x1_ptr[w] = val / batch_size;
    }

    /// collect bias gradients from temp_vec_db
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) private(w, thd)
#endif
    for(w = 0; w < out_channels; ++w) {
        float val = 0;
        for(thd = 0; thd < num_threads; ++thd) {
            val += temp_vec_db[thd*out_channels + w];
        }
        grad_bias_ptr[w] = val / batch_size;
    }
}

#endif //PYLIBAPR_APRNETOPS_HPP