#include "data_structures/APR/APR.hpp"
#include "data_containers/PyPixelData.hpp"
#include "data_containers/PyAPR.hpp"
#include "numerics/APRReconstruction.hpp"

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
 * @param              py:array object (should be initialized to correct size  for level already)
 * @z                  slice of image at levels resolution
 * @level              the level of particles to be added.
 */
void fill_slice(PyAPR &aPyAPR, PyParticleData<uint16_t> &particles,py::array &input,int z,int level) {

        uint16_t min_val = std::numeric_limits<uint16_t>::min();

        auto buf = input.request();

        auto *ptr = static_cast<uint16_t*>(buf.ptr);

        PixelData<uint16_t> input_img;

        input_img.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr); // may lead to memory issues

        auto apr_it = aPyAPR.apr.iterator();

        std::fill(input_img.mesh.begin(),input_img.mesh.end(),min_val);

        int x;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_it)
#endif
    for(x = 0;x < apr_it.x_num(level);++x){
        for(apr_it.begin(level,z,x);apr_it < apr_it.end();apr_it++){

            input_img.at(apr_it.y(),x,0) = particles.parts[apr_it];

        }
    }

}

void AddViewerHelpers(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());
    m2.def("fill_slice", &fill_slice, "fills an array with particles at that level and slice z");
    m2.def("min_occupied_level", &min_occupied_level, "Returns the minimum occupied level in the APR");
}