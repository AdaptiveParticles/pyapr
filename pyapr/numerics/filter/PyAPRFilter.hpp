#include "data_structures/APR/APR.hpp"
#include "data_containers/PyAPR.hpp"
#include "numerics/APRReconstruction.hpp"
#include "numerics/APRFilter.hpp"

#ifdef PYAPR_USE_CUDA
#include "numerics/APRNumericsGPU.hpp"
#include "numerics/PixelNumericsGPU.hpp"
#endif

namespace py = pybind11;

template<typename T>
void get_ds_stencil_vec(py::buffer_info& stencil_buf, VectorData<T>& stencil_vec, int num_levels, bool normalize = false) {

    auto* stencil_ptr = static_cast<T*>(stencil_buf.ptr);

    PixelData<T> stencil;
    stencil.init_from_mesh(stencil_buf.shape[0], stencil_buf.shape[1], stencil_buf.shape[2], stencil_ptr); // may lead to memory issues

    get_downsampled_stencils(stencil, stencil_vec, num_levels, normalize);
}

template<typename T>
void get_ds_stencil_vec(py::buffer_info& stencil_buf, std::vector<PixelData<T>>& stencil_vec, int num_levels, bool normalize = false) {

    auto* stencil_ptr = static_cast<T*>(stencil_buf.ptr);

    PixelData<T> stencil;
    stencil.init_from_mesh(stencil_buf.shape[0], stencil_buf.shape[1], stencil_buf.shape[2], stencil_ptr); // may lead to memory issues

    get_downsampled_stencils(stencil, stencil_vec, num_levels, normalize);
}


template<typename inputType, typename stencilType>
void convolve(PyAPR& apr, PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
                  py::array_t<stencilType>& stencil, bool use_stencil_downsample, bool normalize_stencil, bool use_reflective_boundary) {

    auto stencil_buf = stencil.request();
    int stencil_size;

    if( stencil_buf.ndim == 3 ) {
        stencil_size = stencil_buf.shape[0];
        if( ((stencil_size != 3) && (stencil_size != 5)) || (stencil_buf.shape[1] != stencil_size) || (stencil_buf.shape[2] != stencil_size) ) {
            throw std::invalid_argument("stencil must have shape (3, 3, 3) or (5, 5, 5)");
        }
    } else {
        throw std::invalid_argument("stencil must have 3 dimensions");
    }

    std::vector<PixelData<stencilType>> stencil_vec;
    int nlevels = use_stencil_downsample ? apr.level_max() - apr.level_min() : 1;
    get_ds_stencil_vec(stencil_buf, stencil_vec, nlevels, normalize_stencil);

    APRFilter filter_fns;
    filter_fns.boundary_cond = use_reflective_boundary;
    filter_fns.convolve(apr.apr, stencil_vec, input_parts.parts, output_parts.parts);
}


template<typename inputType, typename stencilType>
void convolve_pencil(PyAPR& apr, PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
                     py::array_t<stencilType>& stencil, bool use_stencil_downsample, bool normalize_stencil, bool use_reflective_boundary) {

    auto stencil_buf = stencil.request();
    int stencil_size;

    if( stencil_buf.ndim == 3 ) {
        stencil_size = stencil_buf.shape[0];
        if( ((stencil_size != 3) && (stencil_size != 5)) || (stencil_buf.shape[1] != stencil_size) || (stencil_buf.shape[2] != stencil_size) ) {
            throw std::invalid_argument("stencil must have shape (3, 3, 3) or (5, 5, 5)");
        }
    } else {
        throw std::invalid_argument("stencil must have 3 dimensions");
    }

    std::vector<PixelData<stencilType>> stencil_vec;
    int nlevels = use_stencil_downsample ? apr.level_max() - apr.level_min() : 1;
    get_ds_stencil_vec(stencil_buf, stencil_vec, nlevels, normalize_stencil);

    APRFilter filter_fns;
    filter_fns.boundary_cond = use_reflective_boundary;
    filter_fns.convolve_pencil(apr.apr, stencil_vec, input_parts.parts, output_parts.parts);
}


#ifdef PYAPR_USE_CUDA

template<typename inputType, typename stencilType>
void convolve_cuda(PyAPR& apr, PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
                   py::array_t<stencilType>& stencil, bool use_stencil_downsample, bool normalize_stencil, bool use_reflective_boundary) {

    auto stencil_buf = stencil.request();
    int stencil_size;

    if( stencil_buf.ndim == 3 ) {
        stencil_size = stencil_buf.shape[0];
        if( ((stencil_size != 3) && (stencil_size != 5)) || (stencil_buf.shape[1] != stencil_size) || (stencil_buf.shape[2] != stencil_size) ) {
            throw std::invalid_argument("stencil must have shape (3, 3, 3) or (5, 5, 5)");
        }
    } else {
        throw std::invalid_argument("stencil must have 3 dimensions");
    }

    // copy stencil to VectorData TODO: add a copy-free option
    int num_stencil_elements = stencil_size * stencil_size * stencil_size;
    VectorData<stencilType> stencil_vec;
    stencil_vec.resize(num_stencil_elements);
    auto* stencil_ptr = static_cast<stencilType*>(stencil_buf.ptr);
    std::copy(stencil_ptr, stencil_ptr+num_stencil_elements, stencil_vec.begin());

    VectorData<stencilType> tree_data;
    auto access = apr.apr.gpuAPRHelper();
    auto tree_access = apr.apr.gpuTreeHelper();

    if(stencil_size == 3) {
        isotropic_convolve_333(access, tree_access, input_parts.parts.data, output_parts.parts.data, stencil_vec,
                               tree_data, use_reflective_boundary, use_stencil_downsample, normalize_stencil);
    } else {
        isotropic_convolve_555(access, tree_access, input_parts.parts.data, output_parts.parts.data, stencil_vec,
                               tree_data, use_reflective_boundary, use_stencil_downsample, normalize_stencil);
    }
}


template<typename inputType, typename stencilType>
void richardson_lucy_cuda(PyAPR& apr, PyParticleData<inputType>& input_parts, PyParticleData<stencilType>& output_parts,
                          py::array_t<stencilType>& stencil, int niter, bool use_stencil_downsample, bool normalize_stencil) {

    auto stencil_buf = stencil.request();
    int stencil_size;

    if( stencil_buf.ndim == 3 ) {
        stencil_size = stencil_buf.shape[0];
        if( ((stencil_size != 3) && (stencil_size != 5)) || (stencil_buf.shape[1] != stencil_size) || (stencil_buf.shape[2] != stencil_size) ) {
            throw std::invalid_argument("stencil must have shape (3, 3, 3) or (5, 5, 5)");
        }
    } else {
        throw std::invalid_argument("stencil must have 3 dimensions");
    }

    // copy stencil to VectorData
    auto* stencil_ptr = static_cast<stencilType*>(stencil_buf.ptr);
    PixelData<stencilType> stencil_pd;
    stencil_pd.init_from_mesh(stencil_size, stencil_size, stencil_size, stencil_ptr);

    auto access = apr.apr.gpuAPRHelper();
    auto tree_access = apr.apr.gpuTreeHelper();

    richardson_lucy(access, tree_access, input_parts.parts.data, output_parts.parts.data, stencil_pd, niter, use_stencil_downsample, normalize_stencil);
}


template<typename inputType, typename stencilType>
void richardson_lucy_pixel_cuda(py::array_t<inputType> input, py::array_t<stencilType> output, py::array_t<stencilType>& stencil, int niter) {

    auto stencil_buf = stencil.request();
    auto input_buf = input.request();
    auto output_buf = output.request();

    size_t npixels = 1;
    for( int i = 0; i < 3; i++) {
        if(input_buf.shape[i] == 0 || input_buf.shape[i] != output_buf.shape[i]) {
            throw std::invalid_argument("input and output must have the same shape and 3 non-zero dimensions");
        }
    }

    int stencil_size;

    if( stencil_buf.ndim == 3 ) {
        stencil_size = stencil_buf.shape[0];
        if( ((stencil_size != 3) && (stencil_size != 5)) || (stencil_buf.shape[1] != stencil_size) || (stencil_buf.shape[2] != stencil_size) ) {
            throw std::invalid_argument("stencil must have shape (3, 3, 3) or (5, 5, 5)");
        }
    } else {
        throw std::invalid_argument("stencil must have 3 dimensions");
    }

    PixelData<inputType> input_pd;
    PixelData<stencilType> output_pd;
    PixelData<stencilType> psf_pd;

    auto* input_ptr = static_cast<inputType*>(input_buf.ptr);
    auto* output_ptr = static_cast<stencilType*>(output_buf.ptr);
    auto* psf_ptr = static_cast<stencilType*>(stencil_buf.ptr);

    input_pd.init_from_mesh(input_buf.shape[2], input_buf.shape[1], input_buf.shape[0], input_ptr);
    output_pd.init_from_mesh(output_buf.shape[2], output_buf.shape[1], output_buf.shape[0], output_ptr);
    psf_pd.init_from_mesh(stencil_size, stencil_size, stencil_size, psf_ptr);

    richardson_lucy_pixel(input_pd, output_pd, psf_pd, niter);
}

#endif


void AddPyAPRFilter(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());
    m2.def("convolve", &convolve<float, float>, "Convolve an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);
    m2.def("convolve", &convolve<uint16_t, float>, "Convolve an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);

    m2.def("convolve_pencil", &convolve_pencil<float, float>, "Convolve an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);
    m2.def("convolve_pencil", &convolve_pencil<uint16_t, float>, "Convolve an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);

#ifdef PYAPR_USE_CUDA
    m2.def("convolve_cuda", &convolve_cuda<float, float>, "Filter an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);
    m2.def("convolve_cuda", &convolve_cuda<uint16_t, float>, "Filter an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false, py::arg("use_reflective_boundary")=false);

    m2.def("richardson_lucy", &richardson_lucy_cuda<float, float>, "run APR LR deconvolution on GPU",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"), py::arg("niter"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false);
    m2.def("richardson_lucy", &richardson_lucy_cuda<uint16_t, float>, "run APR LR deconvolution on GPU",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"), py::arg("niter"),
            py::arg("use_stencil_downsample")=true, py::arg("normalize_stencil")=false);

    m2.def("richardson_lucy_pixel", &richardson_lucy_pixel_cuda<float, float>, "run pixel LR deconvolution on GPU");
    m2.def("richardson_lucy_pixel", &richardson_lucy_pixel_cuda<uint16_t, float>, "run pixel LR deconvolution on GPU");
#endif
}