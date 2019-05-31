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

    void init_tree() {
        apr.init_tree();
    }

    int x_num(int level){
        return apr.apr_access.x_num[level];
    }

    int y_num(int level){
        return apr.apr_access.y_num[level];
    }

    int z_num(int level){
        return apr.apr_access.z_num[level];
    }


};

// -------- wrapper -------------------------------------------------
void AddPyAPR(pybind11::module &m, const std::string &modulename) {
    py::class_<PyAPR>(m, modulename.c_str())
            .def(py::init())
            .def("total_number_particles", &PyAPR::total_number_particles, "return number of particles")
            .def("level_min", &PyAPR::level_min, "return the minimum resolution level")
            .def("level_max", &PyAPR::level_max, "return the maximum resolution level")
            .def("init_tree", &PyAPR::init_tree, "initialize APRTree")
            .def("x_num", &PyAPR::x_num,  "Gives the maximum bounds in the x direction for the given level")
            .def("y_num", &PyAPR::y_num,  "Gives the maximum bounds in the y direction for the given level")
            .def("z_num", &PyAPR::z_num,  "Gives the maximum bounds in the z direction for the given level");
}

#endif //PYLIBAPR_PYAPR_HPP
