
#ifndef PYLIBAPR_BINDVIEWERHELPERS_HPP
#define PYLIBAPR_BINDVIEWERHELPERS_HPP

#include "data_structures/APR/APR.hpp"
#include "data_containers/src/BindParticleData.hpp"
#include "numerics/APRReconstruction.hpp"
#include "numerics/APRCompress.hpp"
#include "numerics/APRTreeNumerics.hpp"

namespace py = pybind11;
using namespace py::literals;


int min_occupied_level(APR& apr){

    auto apr_it = apr.iterator();

    for(int i = apr_it.level_min(); i <= apr_it.level_max(); i++){
        if(apr_it.particles_level_end(i) > 0){
            return i;
        }
    }

    return 0;
}



template<typename T>
void fill_slice(APR& apr, PyParticleData<T>& particles, py::array_t<T>& input, int z, int level) {

        uint16_t min_val = std::numeric_limits<T>::min();

        auto buf = input.request();

        auto *ptr = static_cast<T*>(buf.ptr);

        PixelData<T> input_img;

        input_img.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr); // may lead to memory issues

        auto apr_it = apr.iterator();

        std::fill(input_img.mesh.begin(), input_img.mesh.end(), min_val);

        int x;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_it)
#endif
    for(x = 0; x < apr_it.x_num(level); ++x){
        for(apr_it.begin(level,z,x); apr_it < apr_it.end(); apr_it++){

            input_img.at(apr_it.y(), x, 0) = particles[apr_it];

        }
    }

}


template<typename T>
void fill_slice_level(APR& apr, PyParticleData<T>& particles, py::array_t<T>& input, int z, int level) {

        uint16_t min_val = std::numeric_limits<T>::min();

        auto buf = input.request();

        auto *ptr = static_cast<T*>(buf.ptr);

        PixelData<T> input_img;

        input_img.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr); // may lead to memory issues

        auto apr_it = apr.iterator();

        std::fill(input_img.mesh.begin(), input_img.mesh.end(), min_val);

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



void compress_and_fill_slice(APR& apr, PyParticleData<uint16_t>& particles, py::array_t<uint16_t>& input, int z, int level) {

        uint16_t min_val = std::numeric_limits<uint16_t>::min();

        auto buf = input.request();

        auto *ptr = static_cast<uint16_t*>(buf.ptr);

        PixelData<uint16_t> input_img;

        input_img.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr); // may lead to memory issues

        auto apr_it = apr.iterator();

        std::fill(input_img.mesh.begin(), input_img.mesh.end(), min_val);

        auto cr =  particles.compressor;

        int x;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_it)
#endif
    for(x = 0;x < apr_it.x_num(level);++x){
        for(apr_it.begin(level,z,x); apr_it < apr_it.end(); apr_it++){

            uint16_t particle = particles[apr_it];
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


template<typename T>
PixelData<float> get_points(APR& apr, PyParticleData<T>& particles, int z) {

    auto apr_it = apr.iterator();
    int num_parts_in_slice = 0;

    for(int level = apr_it.level_max(); level > apr_it.level_min(); --level) {
        int step_size = std::pow(2, (int)apr_it.level_max() - level);
        int64_t ifirst = apr_it.begin(level, z/step_size, 0);

        apr_it.begin(level, z/step_size, apr_it.x_num(level)-1);
        int64_t ilast = apr_it.end();

        num_parts_in_slice += (ilast-ifirst);
    }

    PixelData<float> arr(num_parts_in_slice, 4, 1);
    int64_t i = 0;
    for(int level = apr_it.level_max(); level > apr_it.level_min(); --level) {
        int step_size = std::pow(2, (int)apr_it.level_max() - level);
        for(int x = 0; x < apr_it.x_num(level); ++x) {
            for(apr_it.begin(level, z/step_size, x); apr_it < apr_it.end(); ++apr_it) {
                arr.mesh[i] = ((float)x+0.5f) * step_size;
                arr.mesh[i+num_parts_in_slice] = ((float)apr_it.y()+0.5f) * step_size;
                arr.mesh[i+2*num_parts_in_slice] = step_size;
                arr.mesh[i+3*num_parts_in_slice] = particles[apr_it];

                i++;
            }
        }
    }
    return arr;
}


void AddViewerHelpers(py::module &m) {
    m.def("fill_slice", &fill_slice<uint8_t>, "fills an array with particles at that level and slice z");
    m.def("fill_slice", &fill_slice<uint16_t>, "fills an array with particles at that level and slice z");
    m.def("fill_slice", &fill_slice<uint64_t>, "fills an array with particles at that level and slice z");
    m.def("fill_slice", &fill_slice<float>, "fills an array with particles at that level and slice z");
    m.def("fill_slice_level", &fill_slice_level<uint8_t>, "fills an array particle level at that location");
    m.def("fill_slice_level", &fill_slice_level<uint16_t>, "fills an array particle level at that location");
    m.def("fill_slice_level", &fill_slice_level<uint64_t>, "fills an array particle level at that location");
    m.def("fill_slice_level", &fill_slice_level<float>, "fills an array particle level at that location");
    m.def("min_occupied_level", &min_occupied_level, "Returns the minimum occupied level in the APR", "apr"_a);
    m.def("compress_and_fill_slice", &compress_and_fill_slice, "compresses and fills the slice");
    m.def("get_points", &get_points<uint8_t>, py::return_value_policy::take_ownership, "extract particles in a given slice as an array of coordinates and properties");
    m.def("get_points", &get_points<uint16_t>, py::return_value_policy::take_ownership, "extract particles in a given slice as an array of coordinates and properties");
    m.def("get_points", &get_points<uint64_t>, py::return_value_policy::take_ownership, "extract particles in a given slice as an array of coordinates and properties");
    m.def("get_points", &get_points<float>, py::return_value_policy::take_ownership, "extract particles in a given slice as an array of coordinates and properties");
}

#endif //PYLIBAPR_BINDVIEWERHELPERS_HPP
