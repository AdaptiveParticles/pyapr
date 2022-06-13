
#ifndef PYLIBAPR_BINDRICHARDSONLUCY_HPP
#define PYLIBAPR_BINDRICHARDSONLUCY_HPP


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "data_structures/APR/APR.hpp"
#include "data_containers/src/BindParticleData.hpp"
#include "numerics/APRReconstruction.hpp"
#include "numerics/APRFilter.hpp"
#include "numerics/APRNumerics.hpp"
#include "numerics/APRStencil.hpp"

#ifdef PYAPR_USE_CUDA
#include "numerics/APRNumericsGPU.hpp"
#endif

namespace py = pybind11;
using namespace py::literals;


namespace PyAPRRL {

    template<typename inputType, typename stencilType>
    void
    richardson_lucy_cpu(APR &apr, PyParticleData <inputType> &input_parts, PyParticleData <stencilType> &output_parts,
                        py::array_t <stencilType> &stencil, int niter, bool use_stencil_downsample,
                        bool normalize_stencil, bool resume) {

        auto stencil_buf = stencil.request();
        auto *stencil_ptr = static_cast<stencilType *>(stencil_buf.ptr);
        PixelData <stencilType> psf;
        psf.init_from_mesh(stencil_buf.shape[2], stencil_buf.shape[1], stencil_buf.shape[0], stencil_ptr);

        APRNumerics::richardson_lucy(apr, input_parts, output_parts, psf, niter, use_stencil_downsample, normalize_stencil, resume);
    }


    template<typename inputType, typename stencilType>
    void richardson_lucy_tv_cpu(APR &apr, PyParticleData <inputType> &input_parts,
                                PyParticleData <stencilType> &output_parts,
                                py::array_t <stencilType> &stencil, int niter, float reg_factor,
                                bool use_stencil_downsample,
                                bool normalize_stencil, bool resume) {

        auto stencil_buf = stencil.request();
        auto *stencil_ptr = static_cast<float *>(stencil_buf.ptr);
        PixelData<float> psf;
        psf.init_from_mesh(stencil_buf.shape[2], stencil_buf.shape[1], stencil_buf.shape[0], stencil_ptr);

        APRNumerics::richardson_lucy_tv(apr, input_parts, output_parts, psf, niter, reg_factor,
                                        use_stencil_downsample, normalize_stencil, resume);
    }


#ifdef PYAPR_USE_CUDA

    template<typename inputType, typename stencilType>
    void richardson_lucy_cuda(APR& apr, PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
                              py::array_t<stencilType>& stencil, int niter, bool use_stencil_downsample, bool normalize_stencil,
                              bool resume) {

        auto stencil_buf = stencil.request();
        auto* stencil_ptr = static_cast<stencilType*>(stencil_buf.ptr);
        PixelData<stencilType> stencil_pd;
        stencil_pd.init_from_mesh(stencil_buf.shape[0], stencil_buf.shape[1], stencil_buf.shape[2], stencil_ptr);

        auto access = apr.gpuAPRHelper();
        auto tree_access = apr.gpuTreeHelper();

        APRNumericsGPU::richardson_lucy(access, tree_access, input_parts.data, output_parts.data, stencil_pd, niter,
                                        use_stencil_downsample, normalize_stencil, resume);
    }
#endif
}


template<typename inputType>
void bindRichardsonLucy(py::module& m) {
    m.def("richardson_lucy", &PyAPRRL::richardson_lucy_cpu<inputType, float>, "APR RL deconvolution",
          "apr"_a, "input_parts"_a, "output_parts"_a, "stencil"_a, "niter"_a, "use_stencil_downsample"_a=true,
          "normalize_stencil"_a=false, "resume"_a=false);

    m.def("richardson_lucy_tv", &PyAPRRL::richardson_lucy_tv_cpu<inputType, float>,
          "APR RL deconvolution with total variation regularization",
          "apr"_a, "input_parts"_a, "output_parts"_a, "stencil"_a, "niter"_a, "reg_factor"_a,
          "use_stencil_downsample"_a=true, "normalize_stencil"_a=false, "resume"_a=false);

#ifdef PYAPR_USE_CUDA
    m.def("richardson_lucy_cuda", &PyAPRRL::richardson_lucy_cuda<inputType, float>, "APR RL deconvolution on GPU",
          "apr"_a, "input_parts"_a, "output_parts"_a, "stencil"_a, "niter"_a, "use_stencil_downsample"_a=true,
          "normalize_stencil"_a=false, "resume"_a=false);
#endif
}



void AddRichardsonLucy(py::module &m) {

    bindRichardsonLucy<uint8_t>(m);
    bindRichardsonLucy<uint16_t>(m);
    bindRichardsonLucy<float>(m);
}

#endif //PYLIBAPR_BINDRICHARDSONLUCY_HPP

