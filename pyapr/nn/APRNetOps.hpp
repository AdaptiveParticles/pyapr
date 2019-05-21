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
#include "data_structures/APR/APR.hpp"
#include "data_structures/APR/ParticleData.hpp"

//#include "PyPixelData.hpp"
#include "PyAPRFiltering.hpp"
#include "data_containers/PyAPR.hpp"

#include <typeinfo>

namespace py = pybind11;

// TODO: no reason for this (or PyAPRFiltering) to be a class
class APRNetOps {

    PyAPRFiltering filter_fns;

public:

    void convolve(py::array &apr_list, py::array &input_features, py::array &weights, py::array &bias, py::array &output, py::array &level_delta) {

        /// requesting buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto bias_buf = bias.request();
        auto output_buf = output.request(true);
        auto dlvl_buf = level_delta.request();

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto weights_ptr = (float *) weights_buf.ptr;
        auto bias_ptr = (float *) bias_buf.ptr;
        auto output_ptr = (float *) output_buf.ptr;
        auto input_ptr = (float *) input_buf.ptr;
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

            PyAPR* aPyAPR = apr_ptr[bn].attr("apr").cast<PyAPR*>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR->apr.level_max() - dlevel, aPyAPR->apr.level_min());

            auto apr_iterator = aPyAPR->apr.iterator();
            auto tree_iterator = aPyAPR->apr.tree_iterator();

            for(in=0; in<in_channels; ++in) {

                const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

                /**** initialize and fill the apr tree ****/
                ParticleData<float> tree_data;

                filter_fns.fill_tree_mean_py_ptr(aPyAPR->apr, input_ptr, tree_data, apr_iterator,
                                                 tree_iterator, in_offset, current_max_level);

                for (out = 0; out < out_channels; ++out) {

                    const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                    float b = bias_ptr[out];

                    std::vector<PixelData<float>> stencil_vec;
                    stencil_vec.resize(nstencils);

                    for(int n=0; n<nstencils; ++n) {

                        stencil_vec[n].init(height, width, 1);

                        size_t offset = out * in_channels * nstencils * height * width +
                                        in * nstencils * height * width +
                                        n * height * width;

                        for(int idx=0; idx < width*height; ++idx) {
                            stencil_vec[n].mesh[idx] = weights_ptr[offset + idx];
                        }
                    }

                    filter_fns.convolve_batchparallel(aPyAPR->apr, input_ptr, stencil_vec, b, out, in,
                                                      current_max_level, in_offset, tree_data, apr_iterator,
                                                      tree_iterator, number_in_channels, out_offset, output_ptr);
                }
            }
        }
    }


    void convolve_backward(py::array &apr_list, py::array &grad_output, py::array &input_features, py::array &weights, py::array &grad_input, py::array &grad_weights, py::array &grad_bias, py::array &level_delta) {

        /// request buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto grad_input_buf = grad_input.request(true);
        auto grad_weights_buf = grad_weights.request(true);
        auto grad_bias_buf = grad_bias.request(true);
        auto grad_output_buf = grad_output.request();
        auto dlvl_buf = level_delta.request();

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto input_ptr = (float *) input_buf.ptr;
        auto weights_ptr = (float *) weights_buf.ptr;
        auto grad_input_ptr = (float *) grad_input_buf.ptr;
        auto grad_weights_ptr = (float *) grad_weights_buf.ptr;
        auto grad_bias_ptr = (float *) grad_bias_buf.ptr;
        auto grad_output_ptr = (float *) grad_output_buf.ptr;
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
        std::vector<float> temp_vec_dw(num_threads*out_channels*in_channels*nstencils*height*width, 0.0f);
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

            PyAPR* aPyAPR = apr_ptr[bn].attr("apr").cast<PyAPR *>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR->apr.level_max() - dlevel, aPyAPR->apr.level_min());

            auto apr_iterator = aPyAPR->apr.iterator();
            auto tree_iterator = aPyAPR->apr.tree_iterator();

            for(in=0; in<in_channels; ++in) {

                const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

                /// initialize and fill the apr tree

                ParticleData<float> tree_data;
                filter_fns.fill_tree_mean_py_ptr(aPyAPR->apr, input_ptr, tree_data,
                                                 apr_iterator, tree_iterator, in_offset, current_max_level);

                ParticleData<float> grad_tree_temp;
                grad_tree_temp.data.resize(tree_data.data.size(), 0.0f);

                for (out = 0; out < out_channels; ++out) {

                    const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                    const uint64_t dw_offset = thread_id * out_channels * in_channels *  nstencils * height * width +
                                               out * in_channels * nstencils * height * width +
                                               in * nstencils * height * width;

                    const size_t db_offset = thread_id * out_channels;

                    std::vector<PixelData<float>> stencil_vec;
                    stencil_vec.resize(nstencils);

                    for(size_t n=0; n<nstencils; ++n) {

                        stencil_vec[n].init(height, width, 1);

                        uint64_t offset = out * in_channels * nstencils * height * width +
                                          in * nstencils * height * width +
                                          n * height * width;

                        for(size_t idx = 0; idx < height * width; ++idx) {
                            stencil_vec[n].mesh[idx] = weights_ptr[offset + idx];
                        }

                    }
                    filter_fns.convolve_batchparallel_backward(aPyAPR->apr, apr_iterator, tree_iterator, input_ptr,
                                                               in_offset, out_offset, stencil_vec, grad_output_ptr,
                                                               grad_input_ptr, temp_vec_db, temp_vec_dw, dw_offset,
                                                               db_offset, grad_tree_temp, tree_data, out, in,
                                                               current_max_level);
                }

                /// push grad_tree_temp to grad_input
                filter_fns.fill_tree_mean_py_backward_ptr(aPyAPR->apr, apr_iterator, tree_iterator,
                                                          grad_input_ptr, grad_tree_temp, in_offset, current_max_level);
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


    void convolve3x3(py::array &apr_list, py::array &input_features, py::array &weights, py::array &bias, py::array &output, py::array &level_delta) {

        /// requesting buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto bias_buf = bias.request();
        auto output_buf = output.request(true);
        auto dlvl_buf = level_delta.request();

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto weights_ptr = (float *) weights_buf.ptr;
        auto bias_ptr = (float *) bias_buf.ptr;
        auto output_ptr = (float *) output_buf.ptr;
        auto input_ptr = (float *) input_buf.ptr;
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

            PyAPR *aPyAPR = apr_ptr[bn].attr("apr").cast<PyAPR *>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR->apr.level_max() - dlevel, aPyAPR->apr.level_min());

            auto apr_iterator = aPyAPR->apr.iterator();
            auto tree_iterator = aPyAPR->apr.tree_iterator();

            for(in=0; in<in_channels; ++in) {

                const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

                /**** initialize and fill the apr tree ****/

                ParticleData<float> tree_data;

                filter_fns.fill_tree_mean_py_ptr(aPyAPR->apr, input_ptr, tree_data, apr_iterator,
                                                 tree_iterator, in_offset, current_max_level);

                for (out = 0; out < out_channels; ++out) {

                    const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                    float b = bias_ptr[out];

                    std::vector<PixelData<float>> stencil_vec;
                    stencil_vec.resize(nstencils);

                    for(int n=0; n<nstencils; ++n) {

                        stencil_vec[n].init(height, width, 1);

                        size_t offset = out * in_channels * nstencils * 9 + in * nstencils * 9 + n * 9;

                        for(int idx=0; idx < width*height; ++idx) {
                            stencil_vec[n].mesh[idx] = weights_ptr[offset + idx];
                        }
                        /*
                        int idx = 0;
                        for (int y = 0; y < height; ++y) {
                            for (int x = 0; x < width; ++x) {
                                //stencil_vec[n].at(y, x, 0) = weights_ptr[offset + idx];
                                stencil_vec[n].mesh[idx] = weights_ptr[offset + idx];
                                idx++;
                            }
                        }
                        */
                    }
                    filter_fns.convolve3x3_batchparallel(aPyAPR->apr, input_ptr, stencil_vec, b, out, in,
                                                         current_max_level, in_offset, tree_data, apr_iterator,
                                                         tree_iterator, number_in_channels, out_offset, output_ptr);
                }
            }
        }
    }


    void convolve3x3_backward(py::array &apr_list, py::array &grad_output, py::array &input_features, py::array &weights, py::array &grad_input, py::array &grad_weights, py::array &grad_bias, py::array &level_delta) {

        /// request buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto grad_input_buf = grad_input.request(true);
        auto grad_weights_buf = grad_weights.request(true);
        auto grad_bias_buf = grad_bias.request(true);
        auto grad_output_buf = grad_output.request();
        auto dlvl_buf = level_delta.request();

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto input_ptr = (float *) input_buf.ptr;
        auto weights_ptr = (float *) weights_buf.ptr;
        auto grad_input_ptr = (float *) grad_input_buf.ptr;
        auto grad_weights_ptr = (float *) grad_weights_buf.ptr;
        auto grad_bias_ptr = (float *) grad_bias_buf.ptr;
        auto grad_output_ptr = (float *) grad_output_buf.ptr;
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
        std::vector<float> temp_vec_dw(num_threads*out_channels*in_channels*nstencils*9, 0.0f);
        std::vector<float> temp_vec_db(out_channels*num_threads, 0.0f);

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

#ifdef PYAPR_HAVE_OPENMP
            const size_t thread_id = omp_get_thread_num();
#else
            const size_t thread_id = 0;
#endif
            PyAPR *aPyAPR = apr_ptr[bn].attr("apr").cast<PyAPR *>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR->apr.level_max() - dlevel, aPyAPR->apr.level_min());

            auto apr_iterator = aPyAPR->apr.iterator();
            auto tree_iterator = aPyAPR->apr.tree_iterator();

            for(in=0; in<in_channels; ++in) {

                const uint64_t in_offset = bn * number_in_channels * nparticles + in * nparticles;

                /// initialize and fill the apr tree

                ParticleData<float> tree_data;
                filter_fns.fill_tree_mean_py_ptr(aPyAPR->apr, input_ptr, tree_data,
                                                 apr_iterator, tree_iterator, in_offset, current_max_level);

                ParticleData<float> grad_tree_temp;
                grad_tree_temp.data.resize(tree_data.data.size(), 0.0f);

                for (out = 0; out < out_channels; ++out) {

                    const uint64_t out_offset = bn * out_channels * nparticles + out * nparticles;

                    const uint64_t dw_offset = thread_id * out_channels * in_channels *  nstencils * 9 +
                                               out * in_channels * nstencils * 9 +
                                               in * nstencils * 9;

                    const size_t db_offset = thread_id * out_channels;

                    std::vector<PixelData<float>> stencil_vec;
                    stencil_vec.resize(nstencils);

                    for(size_t n=0; n<nstencils; ++n) {

                        stencil_vec[n].init(height, width, 1);

                        uint64_t offset = out * in_channels * nstencils * 9 + in * nstencils * 9 + n * 9;

                        for(size_t idx = 0; idx < 9; ++idx) {
                            stencil_vec[n].mesh[idx] = weights_ptr[offset + idx];
                        }

                    }
                    filter_fns.convolve3x3_batchparallel_backward(aPyAPR->apr, apr_iterator, tree_iterator, input_ptr,
                                                                  in_offset, out_offset, stencil_vec, grad_output_ptr,
                                                                  grad_input_ptr, temp_vec_db, temp_vec_dw, dw_offset,
                                                                  db_offset, grad_tree_temp, tree_data, out, in,
                                                                  current_max_level);
                }

                /// push grad_tree_temp to grad_input
                filter_fns.fill_tree_mean_py_backward_ptr(aPyAPR->apr, apr_iterator, tree_iterator,
                                                          grad_input_ptr, grad_tree_temp, in_offset, current_max_level);
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

    void convolve1x1(py::array &apr_list, py::array &input_features, py::array &weights, py::array &bias, py::array &output, py::array &level_delta) {

        /// requesting buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto bias_buf = bias.request();
        auto output_buf = output.request(true);
        auto dlvl_buf = level_delta.request();

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto weights_ptr = (float *) weights_buf.ptr;
        auto bias_ptr = (float *) bias_buf.ptr;
        auto output_ptr = (float *) output_buf.ptr;
        auto input_ptr = (float *) input_buf.ptr;
        auto dlvl_ptr = (int32_t *) dlvl_buf.ptr;

        /// some important numbers implicit in array shapes
        const size_t out_channels = weights_buf.shape[0];
        const size_t in_channels = weights_buf.shape[1];
        const size_t nstencils = weights_buf.shape[2];
        const size_t height = weights_buf.shape[3];
        const size_t width = weights_buf.shape[4];

        const size_t number_in_channels = input_buf.shape[1];
        const size_t nparticles = input_buf.shape[2];

        if(height !=1 || width != 1) {
            std::cerr << "This function assumes a kernel of shape (3, 3) but was given shape (" << height << ", " << width << ")" << std::endl;
        }

        size_t batch_size = apr_buf.size;
        size_t in, out, bn;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(in, out, bn)
#endif
        for(bn = 0; bn < batch_size; ++bn) {

            PyAPR *aPyAPR = apr_ptr[bn].attr("apr").cast<PyAPR *>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR->apr.level_max() - dlevel, aPyAPR->apr.level_min());

            auto apr_iterator = aPyAPR->apr.iterator();
            auto tree_iterator = aPyAPR->apr.tree_iterator();

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
                    filter_fns.convolve1x1_batchparallel(aPyAPR->apr, input_ptr, stencil_vec, b, out, in,
                                                         current_max_level, in_offset, apr_iterator, tree_iterator,
                                                         number_in_channels, out_offset, output_ptr);
                }
            }
        }
    }


    void convolve1x1_backward(py::array &apr_list, py::array &grad_output, py::array &input_features, py::array &weights, py::array &grad_input, py::array &grad_weights, py::array &grad_bias, py::array &level_delta) {

        /// request buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto weights_buf = weights.request();
        auto grad_input_buf = grad_input.request(true);
        auto grad_weights_buf = grad_weights.request(true);
        auto grad_bias_buf = grad_bias.request(true);
        auto grad_output_buf = grad_output.request();
        auto dlvl_buf = level_delta.request();

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto input_ptr = (float *) input_buf.ptr;
        auto weights_ptr = (float *) weights_buf.ptr;
        auto grad_input_ptr = (float *) grad_input_buf.ptr;
        auto grad_weights_ptr = (float *) grad_weights_buf.ptr;
        auto grad_bias_ptr = (float *) grad_bias_buf.ptr;
        auto grad_output_ptr = (float *) grad_output_buf.ptr;
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
        std::vector<float> temp_vec_dw(num_threads*out_channels*in_channels*nstencils*1, 0.0f);
        std::vector<float> temp_vec_db(out_channels*num_threads, 0.0f);

        const size_t number_in_channels = input_buf.shape[1];
        const size_t nparticles = input_buf.shape[2];

        if(height !=1 || width != 1) {
            std::cerr << "This function assumes a kernel of shape (3, 3) but was given shape (" << height << ", " << width << ")" << std::endl;
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
            PyAPR *aPyAPR = apr_ptr[bn].attr("apr").cast<PyAPR *>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR->apr.level_max() - dlevel, aPyAPR->apr.level_min());

            auto apr_iterator = aPyAPR->apr.iterator();
            auto tree_iterator = aPyAPR->apr.tree_iterator();

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
                    filter_fns.convolve1x1_batchparallel_backward(aPyAPR->apr, apr_iterator, tree_iterator, input_ptr,
                                                                  in_offset, out_offset, stencil_vec, grad_output_ptr,
                                                                  grad_input_ptr, temp_vec_db, temp_vec_dw, dw_offset,
                                                                  db_offset, out, in, current_max_level);
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


    void max_pool(py::array &apr_list, py::array &input_features, py::array &output, py::array &level_delta, py::array &index_arr) {

        //APRTimer timer(false);
        //timer.start_timer("max pool forward (store idx)");

        /// request buffers from python arrays
        auto apr_buf = apr_list.request();
        auto input_buf = input_features.request();
        auto output_buf = output.request(true);
        auto dlvl_buf = level_delta.request();
        auto idx_buf = index_arr.request();

        /// pointers to python array data
        auto apr_ptr = (py::object *) apr_buf.ptr;
        auto input_ptr = (float *) input_buf.ptr;
        auto output_ptr = (float *) output_buf.ptr;
        auto dlvl_ptr = (int32_t *) dlvl_buf.ptr;
        auto idx_ptr = (int64_t *) idx_buf.ptr;

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

            PyAPR *aPyAPR = apr_ptr[bn].attr("apr").cast<PyAPR *>();

            int dlevel = dlvl_ptr[bn];
            const unsigned int current_max_level = std::max(aPyAPR->apr.level_max() - dlevel, aPyAPR->apr.level_min());

            auto apr_iterator = aPyAPR->apr.iterator();
            auto tree_iterator = aPyAPR->apr.tree_iterator();
            auto parent_iterator = aPyAPR->apr.tree_iterator();

            const int64_t tree_offset_in  = filter_fns.compute_tree_offset(apr_iterator, tree_iterator, current_max_level);
            const int64_t tree_offset_out = filter_fns.compute_tree_offset(apr_iterator, tree_iterator, current_max_level-1);

            for (channel = 0; channel < number_channels; ++channel) {

                const uint64_t in_offset = bn * number_channels * nparticles_in + channel * nparticles_in;
                const uint64_t out_offset = bn * number_channels * nparticles_out + channel * nparticles_out;

                filter_fns.max_pool_batchparallel(aPyAPR->apr, input_ptr, output_ptr, idx_ptr, in_offset, out_offset,
                                                  tree_offset_in, tree_offset_out, apr_iterator, tree_iterator,
                                                  parent_iterator, current_max_level);
            }
        }
        //timer.stop_timer();
    }


    void max_pool_backward(py::array &grad_output, py::array &grad_input, py::array &max_indices) {

        //APRTimer timer(false);
        //timer.start_timer("max pool backward");

        auto grad_output_buf = grad_output.request();
        auto grad_input_buf = grad_input.request(true);
        auto index_buf = max_indices.request();

        auto grad_output_ptr = (float *) grad_output_buf.ptr;
        auto grad_input_ptr = (float *) grad_input_buf.ptr;
        auto index_ptr = (int64_t *) index_buf.ptr;

        size_t i;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) private(i)
#endif
        for (i = 0; i < index_buf.size; ++i) {
            int64_t idx = index_ptr[i];
            if (idx > -1) {
                grad_input_ptr[idx] = grad_output_ptr[i];
            }
        }
        //timer.stop_timer();
    }

};

#endif //PYLIBAPR_APRNETOPS_HPP
