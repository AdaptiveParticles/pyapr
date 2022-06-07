
#ifndef PYLIBAPR_PYAPRFILTER_HPP
#define PYLIBAPR_PYAPRFILTER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#include "data_structures/APR/APR.hpp"
#include "numerics/APRReconstruction.hpp"
#include "numerics/APRFilter.hpp"
#include "numerics/APRNumerics.hpp"
#include "numerics/APRStencil.hpp"

#ifdef PYAPR_USE_CUDA
#include "numerics/APRIsoConvGPU333.hpp"
#include "numerics/APRIsoConvGPU555.hpp"
#endif

namespace py = pybind11;
using namespace py::literals;

template<typename T>
void get_ds_stencil_vec(py::buffer_info& stencil_buf, VectorData<T>& stencil_vec, int num_levels,
                        bool normalize = false, bool rescale = false) {

    auto* stencil_ptr = static_cast<T*>(stencil_buf.ptr);

    PixelData<T> stencil;
    stencil.init_from_mesh(stencil_buf.shape[2], stencil_buf.shape[1], stencil_buf.shape[0], stencil_ptr);

    if(rescale) {
        APRStencil::get_rescaled_stencils(stencil, stencil_vec, num_levels);
    } else {
        APRStencil::get_downsampled_stencils(stencil, stencil_vec, num_levels, normalize);
    }
}

template<typename T>
void get_ds_stencil_vec(py::buffer_info& stencil_buf, std::vector<PixelData<T>>& stencil_vec, int num_levels,
                        bool normalize = false, bool rescale = false) {

    auto* stencil_ptr = static_cast<T*>(stencil_buf.ptr);

    PixelData<T> stencil;
    stencil.init_from_mesh(stencil_buf.shape[2], stencil_buf.shape[1], stencil_buf.shape[0], stencil_ptr);

    if(rescale) {
        APRStencil::get_rescaled_stencils(stencil, stencil_vec, num_levels);
    } else {
        APRStencil::get_downsampled_stencils(stencil, stencil_vec, num_levels, normalize);
    }
}


template<typename inputType, typename stencilType>
void convolve(APR& apr, const ParticleData<inputType>& input_parts, ParticleData<stencilType>& output_parts,
              py::array_t<stencilType>& stencil, bool use_stencil_downsample, bool normalize_stencil,
              bool use_reflective_boundary, bool rescale_stencil) {

    auto stencil_buf = stencil.request();
    std::vector<PixelData<stencilType>> stencil_vec;
    int nlevels = use_stencil_downsample ? apr.level_max() - apr.level_min() : 1;
    get_ds_stencil_vec(stencil_buf, stencil_vec, nlevels, normalize_stencil, rescale_stencil);

    APRFilter::convolve(apr, stencil_vec, input_parts, output_parts, use_reflective_boundary);
}


template<typename inputType, typename stencilType>
void convolve_pencil(APR& apr, ParticleData<inputType>& input_parts, ParticleData<stencilType>& output_parts,
                     py::array_t<stencilType>& stencil, bool use_stencil_downsample, bool normalize_stencil,
                     bool use_reflective_boundary, bool rescale_stencil) {

    auto stencil_buf = stencil.request();
    std::vector<PixelData<stencilType>> stencil_vec;
    int nlevels = use_stencil_downsample ? apr.level_max() - apr.level_min() : 1;
    get_ds_stencil_vec(stencil_buf, stencil_vec, nlevels, normalize_stencil, rescale_stencil);

    APRFilter::convolve_pencil(apr, stencil_vec, input_parts, output_parts, use_reflective_boundary);
}


#ifdef PYAPR_USE_CUDA

template<typename inputType, typename stencilType>
void convolve_cuda(APR& apr, ParticleData<inputType>& input_parts, ParticleData<stencilType>& output_parts,
                   py::array_t<stencilType>& stencil, bool use_stencil_downsample, bool normalize_stencil,
                   bool use_reflective_boundary, bool rescale_stencil) {

    auto stencil_buf = stencil.request();
    VectorData<stencilType> stencil_vec;
    int nlevels = use_stencil_downsample ? apr.level_max() - apr.level_min() : 1;
    get_ds_stencil_vec(stencil_buf, stencil_vec, nlevels, normalize_stencil, rescale_stencil);

    VectorData<stencilType> tree_data;
    auto access = apr.gpuAPRHelper();
    auto tree_access = apr.gpuTreeHelper();

    if(stencil_buf.shape[0] == 3) {
        isotropic_convolve_333_direct(access, tree_access, input_parts.data, output_parts.data, stencil_vec,
                                      tree_data, use_reflective_boundary);
    } else {
        isotropic_convolve_555_direct(access, tree_access, input_parts.data, output_parts.data, stencil_vec,
                                      tree_data, use_reflective_boundary);
    }
}

#endif


template<int size_z, int size_x, int size_y>
void bindMedianFilter(py::module &m) {
    std::string name = "median_filter_" + std::to_string(size_z) + std::to_string(size_x) + std::to_string(size_y);
    m.def(name.c_str(), &APRFilter::median_filter<size_y, size_x, size_z, uint16_t, uint16_t>, "median filter",
          "apr"_a, "input_parts"_a, "output_parts"_a);
    m.def(name.c_str(), &APRFilter::median_filter<size_y, size_x, size_z, float, float>, "median filter",
          "apr"_a, "input_parts"_a, "output_parts"_a);
}

template<int size_z, int size_x, int size_y>
void bindMinFilter(py::module &m) {
    std::string name = "min_filter_" + std::to_string(size_z) + std::to_string(size_x) + std::to_string(size_y);
    m.def(name.c_str(), &APRFilter::min_filter<size_y, size_x, size_z, uint16_t, uint16_t>, "min filter",
          "apr"_a, "input_parts"_a, "output_parts"_a);
    m.def(name.c_str(), &APRFilter::min_filter<size_y, size_x, size_z, float, float>, "min filter",
          "apr"_a, "input_parts"_a, "output_parts"_a);
}

template<int size_z, int size_x, int size_y>
void bindMaxFilter(py::module &m) {
    std::string name = "max_filter_" + std::to_string(size_z) + std::to_string(size_x) + std::to_string(size_y);
    m.def(name.c_str(), &APRFilter::max_filter<size_y, size_x, size_z, uint16_t, uint16_t>, "max filter",
          "apr"_a, "input_parts"_a, "output_parts"_a);
    m.def(name.c_str(), &APRFilter::max_filter<size_y, size_x, size_z, float, float>, "max filter",
          "apr"_a, "input_parts"_a, "output_parts"_a);
}


template<typename inputType, typename stencilType>
void bindConvolution(py::module &m) {
    m.def("convolve", &convolve<inputType, stencilType>, "Convolve an APR with a stencil",
          "apr"_a, "input_parts"_a, "output_parts"_a, "stencil"_a, "use_stencil_downsample"_a=true,
          "normalize_stencil"_a=false, "use_reflective_boundary"_a=false, "rescale_stencil"_a=false);

    m.def("convolve_pencil", &convolve_pencil<inputType, stencilType>, "Convolve an APR with a stencil",
          "apr"_a, "input_parts"_a, "output_parts"_a, "stencil"_a, "use_stencil_downsample"_a=true,
          "normalize_stencil"_a=false, "use_reflective_boundary"_a=false, "rescale_stencil"_a=false);

#ifdef PYAPR_USE_CUDA
    m.def("convolve_cuda", &convolve_cuda<inputType, stencilType>, "Convolve an APR with a stencil on the GPU",
          "apr"_a, "input_parts"_a, "output_parts"_a, "stencil"_a, "use_stencil_downsample"_a=true,
          "normalize_stencil"_a=false, "use_reflective_boundary"_a=false, "rescale_stencil"_a=false);
#endif
}

template<typename inputType, typename outputType>
void bindGradient(py::module &m) {

    m.def("gradient_cfd", &APRNumerics::gradient_cfd<inputType, outputType>,
          "adaptive gradient filter using central finite differences",
          "apr"_a, "input_parts"_a, "output_parts"_a, "dim"_a, "delta"_a=1.0f);

    m.def("gradient_sobel", &APRNumerics::gradient_sobel<inputType, outputType>,
          "adaptive sobel gradient filter",
          "apr"_a, "input_parts"_a, "output_parts"_a, "dim"_a, "delta"_a=1.0f);

    m.def("gradient_magnitude_cfd", &APRNumerics::gradient_magnitude_cfd<inputType, outputType>,
          "adaptive gradient magnitude using central finite differences",
          "apr"_a, "input_parts"_a, "output_parts"_a, "deltas"_a);

    m.def("gradient_magnitude_sobel", &APRNumerics::gradient_magnitude_sobel<inputType, outputType>,
          "adaptive gradient magnitude using sobel filters",
          "apr"_a, "input_parts"_a, "output_parts"_a, "deltas"_a);
}


template<typename inputType, typename outputType>
void bindStdFilter(py::module &m) {
    m.def("local_std", &APRNumerics::local_std<inputType, outputType>,
          "compute local standard deviation around each particle",
          "apr"_a, "input_parts"_a, "output_parts"_a, "size"_a);
}

void AddFilter(py::module &m) {

    bindConvolution<uint8_t, float>(m);
    bindConvolution<uint16_t, float>(m);
    bindConvolution<uint64_t, float>(m);
    bindConvolution<float, float>(m);

    bindGradient<uint8_t, float>(m);
    bindGradient<uint16_t, float>(m);
    bindGradient<float, float>(m);

    bindStdFilter<uint8_t, float>(m);
    bindStdFilter<uint16_t, float>(m);
    bindStdFilter<float, float>(m);

    auto m3 = m.def_submodule("rank_filters");

    bindMedianFilter<3, 3, 3>(m3);
    bindMedianFilter<5, 5, 5>(m3);
    bindMedianFilter<7, 7, 7>(m3);
    bindMedianFilter<9, 9, 9>(m3);
    bindMedianFilter<11, 11, 11>(m3);

    bindMedianFilter<1, 3, 3>(m3);
    bindMedianFilter<1, 5, 5>(m3);
    bindMedianFilter<1, 7, 7>(m3);
    bindMedianFilter<1, 9, 9>(m3);
    bindMedianFilter<1, 11, 11>(m3);

    bindMinFilter<1, 3, 3>(m3);
    bindMinFilter<1, 5, 5>(m3);
    bindMinFilter<3, 3, 3>(m3);
    bindMinFilter<5, 5, 5>(m3);

    bindMaxFilter<1, 3, 3>(m3);
    bindMaxFilter<1, 5, 5>(m3);
    bindMaxFilter<3, 3, 3>(m3);
    bindMaxFilter<5, 5, 5>(m3);

}

#endif //PYLIBAPR_PYAPRFILTER_HPP