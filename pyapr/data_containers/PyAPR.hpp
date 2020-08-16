//
// Created by Joel Jonsson on 29.06.18.
//

#ifndef PYLIBAPR_PYAPR_HPP
#define PYLIBAPR_PYAPR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

//#include "ConfigAPR.h"
#include "data_structures/APR/APR.hpp"
#include "data_containers/iterators/PyLinearIterator.hpp"

namespace py = pybind11;

class PyAPR {

public:

    APR apr;

    PyAPR () {}

    int level_min() {
        return apr.level_min();
    }

    int level_max() {
        return apr.level_max();
    }

    int total_number_particles() {
        return apr.total_number_particles();
    }

    int x_num(int level){
        return apr.x_num(level);
    }

    int y_num(int level){
        return apr.y_num(level);
    }

    int z_num(int level){
        return apr.z_num(level);
    }

    PyLinearIterator iterator() {
        return PyLinearIterator(apr, false);
    }

    PyLinearIterator tree_iterator() {
        return PyLinearIterator(apr, true);
    }

    py::tuple org_dims() {
        return py::make_tuple(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2));
    }

    APRParameters get_parameters() {
        return apr.get_apr_parameters();
    }
};

// -------- wrapper -------------------------------------------------
void AddPyAPR(pybind11::module &m, const std::string &modulename) {
    py::class_<PyAPR>(m, modulename.c_str())
            .def(py::init())
            .def("__repr__", [](PyAPR& a) {
                return "PyAPR: " + std::to_string(a.total_number_particles()) + " particles. Original image shape: (z, x, y) = (" + \
                        std::to_string(a.apr.org_dims(2)) + ", " + std::to_string(a.apr.org_dims(1)) + ", " + std::to_string(a.apr.org_dims(0)) \
                         + "). Computational Ratio = " + std::to_string(a.apr.computational_ratio());
            })
            .def("total_number_particles", &PyAPR::total_number_particles, "return number of particles")
            .def("level_min", &PyAPR::level_min, "return the minimum resolution level")
            .def("level_max", &PyAPR::level_max, "return the maximum resolution level")
            .def("x_num", &PyAPR::x_num,  "Gives the maximum bounds in the x direction for the given level")
            .def("y_num", &PyAPR::y_num,  "Gives the maximum bounds in the y direction for the given level")
            .def("z_num", &PyAPR::z_num,  "Gives the maximum bounds in the z direction for the given level")
            .def("org_dims", &PyAPR::org_dims, "returns the original pixel image dimensions as a tuple (y, x, z)")
            .def("iterator", &PyAPR::iterator, "return a linear iterator for APR particles")
            .def("tree_iterator", &PyAPR::tree_iterator, "return a linear iterator for tree particles")
            .def("get_parameters", &PyAPR::get_parameters, "return the parameters used to create the APR");
}

#endif //PYLIBAPR_PYAPR_HPP
