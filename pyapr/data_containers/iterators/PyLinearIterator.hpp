//
// Created by Joel Jonsson on 07.05.20.
//

#ifndef PYLIBAPR_PYLINEARITERATOR_HPP
#define PYLIBAPR_PYLINEARITERATOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "data_structures/APR/APR.hpp"

namespace py = pybind11;

class PyLinearIterator {

public:

    LinearIterator it;

    PyLinearIterator() {}

    PyLinearIterator(APR& apr, bool tree=false) {
        if(tree) {
            it = apr.tree_iterator();
        } else {
            it = apr.iterator();
        }
    }

    int level_min() {
        return it.level_min();
    }

    int level_max() {
        return it.level_max();
    }

    int total_number_particles(int level=0) {
        return it.total_number_particles(level);
    }

    int x_num(int level){
        return it.x_num(level);
    }

    int y_num(int level){
        return it.y_num(level);
    }

    int z_num(int level){
        return it.z_num(level);
    }

    int y(size_t index) const {
        return it.get_y(index);
    }

    size_t begin(int level, int z, int x) {
        return it.begin(level, z, x);
    }

    size_t end() {
        return it.end();
    }

};

// -------- wrapper -------------------------------------------------
void AddPyLinearIterator(pybind11::module &m, const std::string &modulename) {
    py::class_<PyLinearIterator>(m, modulename.c_str())
            .def(py::init())
            .def("total_number_particles", &PyLinearIterator::total_number_particles, "return number of particles",
                 py::arg("level")=0)
            .def("level_min", &PyLinearIterator::level_min, "return the minimum resolution level")
            .def("level_max", &PyLinearIterator::level_max, "return the maximum resolution level")
            .def("x_num", &PyLinearIterator::x_num,  "Gives the maximum bounds in the x direction for the given level")
            .def("y_num", &PyLinearIterator::y_num,  "Gives the maximum bounds in the y direction for the given level")
            .def("z_num", &PyLinearIterator::z_num,  "Gives the maximum bounds in the z direction for the given level")
            .def("y", &PyLinearIterator::y, "returns the y-coordinate of a given particle index")
            .def("begin", &PyLinearIterator::begin, "returns the index of the first particle at a given combination of level, z, and x",
                        py::arg("level"), py::arg("z"), py::arg("z"))
            .def("end", &PyLinearIterator::end, "returns the end index of the current row (l, z, x combination)");
}

#endif //PYLIBAPR_PYLINEARITERATOR_HPP
