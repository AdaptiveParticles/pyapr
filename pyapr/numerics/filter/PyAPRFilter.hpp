#include "data_structures/APR/APR.hpp"
#include "data_containers/PyAPR.hpp"
#include "numerics/APRReconstruction.hpp"
#include "numerics/APRIsoConvGPU.hpp"

namespace py = pybind11;


void get_ds_stencil_vec(py::buffer_info& stencil_buf, VectorData<float>& stencil_vec, int num_levels, bool normalize = false) {

    float* stencil_ptr = (float*) stencil_buf.ptr;

    PixelData<float> stencil(stencil_buf.shape[0], stencil_buf.shape[1], stencil_buf.shape[2]);

    for(int i = 0; i < stencil.mesh.size(); ++i) {
        stencil.mesh[i] = stencil_ptr[i];
    }

    get_downsampled_stencils(stencil, stencil_vec, num_levels, normalize);
}


void convolve_cuda(PyAPR& apr, PyParticleData<float>& input_parts, PyParticleData<float>& output_parts, py::array_t<float>& stencil, bool downsample_stencil, bool normalize_stencil) {

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

    const int nelements = stencil_size * stencil_size * stencil_size;

    auto access = apr.apr.gpuAPRHelper();
    auto tree_access = apr.apr.gpuTreeHelper();

    tree_access.init_gpu();
    access.init_gpu(access.total_number_particles(tree_access.level_max()), tree_access);

    output_parts.parts.init(access.total_number_particles());

    VectorData<float> stencil_vec;

    if(downsample_stencil) {
        get_ds_stencil_vec(stencil_buf, stencil_vec, access.level_max()-access.level_min(), false);
    } else {
        stencil_vec.resize(nelements);
        float* ptr = (float*)stencil_buf.ptr;
        std::copy(ptr, ptr+nelements, stencil_vec.begin());
    }

    ScopedCudaMemHandler<float*, JUST_ALLOC> stencil_gpu(stencil_vec.data(), stencil_vec.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> input_gpu(input_parts.data(), input_parts.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> output_gpu(output_parts.data(), output_parts.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> tree_data_gpu(NULL, tree_access.total_number_particles());

    input_gpu.copyH2D();
    stencil_gpu.copyH2D();

    downsample_avg_alt(access, tree_access, input_gpu.get(), tree_data_gpu.get());
    cudaDeviceSynchronize();

    if(stencil_size == 3) {
        isotropic_convolve_333(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get(), downsample_stencil);
    } else {
        if(downsample_stencil) {
            isotropic_convolve_555_ds(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get());
        } else {
            isotropic_convolve_555(access, tree_access, input_gpu.get(), output_gpu.get(), stencil_gpu.get(), tree_data_gpu.get());
        }
    }
    cudaDeviceSynchronize();

    output_gpu.copyD2H();
    cudaDeviceSynchronize();
}


void richardson_lucy_cuda(PyAPR& apr, PyParticleData<float>& input_parts, PyParticleData<float>& output_parts,
                          py::array_t<float>& stencil, int niter, bool downsample_stencil, bool normalize_stencil) {
    APRTimer timer(true);
    timer.start_timer("prep");

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

    const int nelements = stencil_size * stencil_size * stencil_size;

    auto access = apr.apr.gpuAPRHelper();
    auto tree_access = apr.apr.gpuTreeHelper();

    tree_access.init_gpu();
    access.init_gpu(access.total_number_particles(tree_access.level_max()), tree_access);

    output_parts.parts.init(access.total_number_particles());

    PixelData<float> pd_stencil(stencil_size, stencil_size, stencil_size);
    float* stenc_ptr = (float*) stencil_buf.ptr;
    for(int i = 0; i < pd_stencil.mesh.size(); ++i) {
        pd_stencil.mesh[i] = stenc_ptr[i];
    }

    ScopedCudaMemHandler<float*, JUST_ALLOC> input_gpu(input_parts.data(), input_parts.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> output_gpu(output_parts.data(), output_parts.size());

    input_gpu.copyH2D();
    cudaDeviceSynchronize();
    timer.stop_timer();

    timer.start_timer("deconv");

    richardson_lucy(access, tree_access, input_gpu.get(), output_gpu.get(), pd_stencil, niter, downsample_stencil, normalize_stencil);
    cudaDeviceSynchronize();

    timer.stop_timer();

    timer.start_timer("copy D2H");

    output_gpu.copyD2H();
    cudaDeviceSynchronize();

    timer.stop_timer();
}


void richardson_lucy_pixel_cuda(py::array_t<float> input, py::array_t<float> output, py::array_t<float>& stencil, int niter) {

    auto stencil_buf = stencil.request();
    auto input_buf = input.request();
    auto output_buf = output.request();

    std::vector<int> dims(3);

    size_t npixels = 1;
    for( int i = 0; i < 3; i++) {
        if(input_buf.shape[i] == 0 || input_buf.shape[i] != output_buf.shape[i]) {
            throw std::invalid_argument("input and output must have 3 equal and non-zero dimensions");
        }
        npixels *= input_buf.shape[i];
        dims[i] = input_buf.shape[i];
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

    const int nelements = stencil_size * stencil_size * stencil_size;

    VectorData<float> psf_flipped;
    psf_flipped.resize(nelements);
    float* psf_ptr = (float*) stencil_buf.ptr;

    for(int i = 0; i < nelements; ++i) {
        psf_flipped[i] = psf_ptr[nelements-1-i];
    }

    ScopedCudaMemHandler<float*, JUST_ALLOC> psf_flipped_gpu(psf_flipped.data(), psf_flipped.size());
    ScopedCudaMemHandler<float*, JUST_ALLOC> psf_gpu(psf_ptr, nelements);
    ScopedCudaMemHandler<float*, JUST_ALLOC> input_gpu((float*) input_buf.ptr, npixels);
    ScopedCudaMemHandler<float*, JUST_ALLOC> output_gpu((float*) output_buf.ptr, npixels);

    input_gpu.copyH2D();
    psf_gpu.copyH2D();
    psf_flipped_gpu.copyH2D();
    cudaDeviceSynchronize();

    richardson_lucy_pixel(input_gpu.get(), output_gpu.get(), psf_gpu.get(), psf_flipped_gpu.get(), stencil_size, npixels, niter, dims);
    cudaDeviceSynchronize();

    output_gpu.copyD2H();
    cudaDeviceSynchronize();
}


void AddPyAPRFilter(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());
    m2.def("convolve_cuda", &convolve_cuda, "Filter an APR with a stencil",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"),
            py::arg("downsample_stencil")=true, py::arg("normalize_stencil")=false);
    m2.def("richardson_lucy", &richardson_lucy_cuda, "run APR LR deconvolution on GPU",
            py::arg("apr"), py::arg("input_parts"), py::arg("output_parts"), py::arg("stencil"), py::arg("niter"),
            py::arg("downsample_stencil")=true, py::arg("normalize_stencil")=false);
    m2.def("richardson_lucy_pixel", &richardson_lucy_pixel_cuda, "run pixel LR deconvolution on GPU");
}