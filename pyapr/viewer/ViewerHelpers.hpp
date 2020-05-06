#include "data_structures/APR/APR.hpp"
#include "data_containers/PyPixelData.hpp"
#include "data_containers/PyAPR.hpp"
#include "numerics/APRReconstruction.hpp"
#include "numerics/APRCompress.hpp"
#include "numerics/APRRaycaster.hpp"
#include "numerics/APRTreeNumerics.hpp"

namespace py = pybind11;

 int min_occupied_level(PyAPR &aPyAPR){

    auto apr_it = aPyAPR.apr.iterator();

    for(int i = apr_it.level_min(); i <= apr_it.level_max(); i++){
        if(apr_it.particles_level_end(i) > 0){
            return i;
        }
    }

    return 0;
 }


/**
 *
 * @param aPyAPR       a PyAPR object
 * @param particles    a PyParticleData object
 * @param particles_tree    a PyParticleData object
 * @type       std::string Type of downsampling
 */
template<typename T, typename S>
void get_down_sample_parts(PyAPR &aPyAPR, PyParticleData<T> &particles,PyParticleData<S> &tree_particles){

   APRTreeNumerics::fill_tree_max(aPyAPR.apr, particles.parts,tree_particles.parts);

};


/**
 *
 * @param aPyAPR       a PyAPR object
 * @param particles    a PyParticleData object
 * @param              py:array object (should be initialized to correct size  for level already)
 * @z                  slice of image at levels resolution
 * @level              the level of particles to be added.
 */
template<typename T>
void fill_slice(PyAPR &aPyAPR, PyParticleData<T> &particles, py::array_t<T> &input, int z, int level) {

        uint16_t min_val = std::numeric_limits<T>::min();

        auto buf = input.request();

        auto *ptr = static_cast<T*>(buf.ptr);

        PixelData<T> input_img;

        input_img.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr); // may lead to memory issues

        auto apr_it = aPyAPR.apr.iterator();

        std::fill(input_img.mesh.begin(),input_img.mesh.end(),min_val);

        int x;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_it)
#endif
    for(x = 0; x < apr_it.x_num(level); ++x){
        for(apr_it.begin(level,z,x); apr_it < apr_it.end(); apr_it++){

            input_img.at(apr_it.y(), x, 0) = particles.parts[apr_it];

        }
    }

}


/**
 *
 * @param aPyAPR       a PyAPR object
 * @param particles    a PyParticleData object
 * @param              py:array object (should be initialized to correct size  for level already)
 * @z                  slice of image at levels resolution
 * @level              the level of particles to be added.
 */
template<typename T>
void fill_slice_level(PyAPR &aPyAPR, PyParticleData<T> &particles, py::array_t<T> &input, int z, int level) {

        uint16_t min_val = std::numeric_limits<T>::min();

        auto buf = input.request();

        auto *ptr = static_cast<T*>(buf.ptr);

        PixelData<T> input_img;

        input_img.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr); // may lead to memory issues

        auto apr_it = aPyAPR.apr.iterator();

        std::fill(input_img.mesh.begin(),input_img.mesh.end(),min_val);

        int x;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_it)
#endif
    for(x = 0;x < apr_it.x_num(level);++x){
        for(apr_it.begin(level,z,x);apr_it < apr_it.end();apr_it++){

            input_img.at(apr_it.y(),x,0) = level;

        }
    }

}

/**
 *
 * @param aPyAPR       a PyAPR object
 * @param particles    a PyParticleData object
 * @param              py:array object (should be initialized to correct size  for level already)
 * @z                  slice of image at levels resolution
 * @level              the level of particles to be added.
 */
void compress_and_fill_slice(PyAPR &aPyAPR, PyParticleData<uint16_t> &particles,py::array &input,int z,int level) {

        uint16_t min_val = std::numeric_limits<uint16_t>::min();

        auto buf = input.request();

        auto *ptr = static_cast<uint16_t*>(buf.ptr);

        PixelData<uint16_t> input_img;

        input_img.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr); // may lead to memory issues

        auto apr_it = aPyAPR.apr.iterator();

        std::fill(input_img.mesh.begin(),input_img.mesh.end(),min_val);

        auto cr =  particles.parts.compressor;

        int x;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_it)
#endif
    for(x = 0;x < apr_it.x_num(level);++x){
        for(apr_it.begin(level,z,x);apr_it < apr_it.end();apr_it++){

            uint16_t particle = particles.parts[apr_it];
            //lossy encode particle
            if(cr.get_quantization_factor() > 0){

                particle = cr.calculate_symbols<uint16_t,float>(cr.variance_stabilitzation<float>(particle));
                //decode particle
                particle = cr.inverse_variance_stabilitzation<float>(cr.inverse_calculate_symbols<float,uint16_t>(particle));
            }

            input_img.at(apr_it.y(),x,0) = particle;

        }
    }

}


void AddViewerHelpers(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());
    m2.def("get_down_sample_parts", &get_down_sample_parts<uint16_t, uint16_t>, "creates down-sampled tree particles");
    m2.def("get_down_sample_parts", &get_down_sample_parts<uint16_t, float>, "creates down-sampled tree particles");
    m2.def("get_down_sample_parts", &get_down_sample_parts<float, uint16_t>, "creates down-sampled tree particles");
    m2.def("get_down_sample_parts", &get_down_sample_parts<float, float>, "creates down-sampled tree particles");
    m2.def("fill_slice", &fill_slice<uint16_t>, "fills an array with particles at that level and slice z");
    m2.def("fill_slice", &fill_slice<float>, "fills an array with particles at that level and slice z");
    m2.def("fill_slice_level", &fill_slice_level<uint16_t>, "fills an array particle level at that location");
    m2.def("fill_slice_level", &fill_slice_level<float>, "fills an array particle level at that location");
    m2.def("min_occupied_level", &min_occupied_level, "Returns the minimum occupied level in the APR");
    m2.def("compress_and_fill_slice", &compress_and_fill_slice, "compresses and fills the slice");
}