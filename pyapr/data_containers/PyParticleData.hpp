#ifndef PYLIBAPR_PYPARTICLEDATA_HPP
#define PYLIBAPR_PYPARTICLEDATA_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "data_containers/PyAPR.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"

namespace py = pybind11;

/**
 * @tparam T data type of particle values
 */
template<typename T>
class PyParticleData {

public:

    ParticleData<T> parts; //#TODO: should this be called something else?

    PyParticleData() {}
    PyParticleData(uint64_t num_particles){ parts.init(num_particles); }

    PyParticleData(ParticleData<T> &aInput) {
        parts.swap(aInput);
    }

    /**
     * @return pointer to the data
     */
    T* data() {return parts.data.data();}

    /**
     * @return size of the data (number of particle values)
     */
    uint64_t size() const { return parts.data.size(); }


    /**
     * Sample the particle values from an image (numpy array) for a given APR (computed from the same image).
     *
     * @param aPyAPR  PyAPR object
     * @param img     numpy array representing the image
     */
    void sample_image(PyAPR &aPyAPR, py::array &img) {
        auto buf = img.request(false);
        unsigned int y_num, x_num, z_num;

        std::vector<ssize_t> shape = buf.shape;

        if(buf.ndim == 3) {
            y_num = shape[2];
            x_num = shape[1];
            z_num = shape[0];
        } else if (buf.ndim == 2) {
            y_num = shape[1];
            x_num = shape[0];
            z_num = 1;
        } else if (buf.ndim == 1) {
            y_num = shape[0];
            x_num = 1;
            z_num = 1;
        } else {
            throw std::invalid_argument("input array must be of dimension at most 3");
        }

        auto ptr = (T*) buf.ptr;

        PixelData<T> pd;
        pd.init_from_mesh(y_num, x_num, z_num, ptr);

        parts.sample_parts_from_img_downsampled(aPyAPR.apr, pd);
    }

    /**
     * Fill the particle values with their corresponding resolution levels.
     *
     * @param aPyAPR  a PyAPR object
     */
    void fill_with_levels(PyAPR &aPyAPR) {

        auto it = aPyAPR.apr.iterator();

        for (unsigned int level = it.level_min(); level <= it.level_max(); ++level) {

            T lvl = static_cast<T>(level);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
            for (int z = 0; z < it.z_num(level); ++z) {
                for (int x = 0; x < it.x_num(level); ++x) {
                    for (it.begin(level, z, x);it <it.end();it++) {

                        parts[it] = lvl;
                    }
                }
            }
        }
    }
};


template<typename DataType>
void AddPyParticleData(pybind11::module &m, const std::string &aTypeString) {
    using TypeParticles = PyParticleData<DataType>;
    std::string typeStr = aTypeString + "Particles";
    py::class_<TypeParticles>(m, typeStr.c_str(), py::buffer_protocol())
            .def(py::init())
            .def("__len__", [](const TypeParticles &p){ return p.size(); })
            .def("sample_image", &TypeParticles::sample_image, "sample particle values from an image (numpy array)")
            .def("fill_with_levels", &TypeParticles::fill_with_levels, "fill particle values with levels")
            .def_buffer([](TypeParticles &p) -> py::buffer_info{
                return py::buffer_info(
                        p.data(),
                        sizeof(DataType),
                        py::format_descriptor<DataType>::format(),
                        1,
                        {p.size()},
                        {sizeof(DataType)}
                );
            });
}

#endif //PYLIBAPR_PYPARTICLEDATA_HPP