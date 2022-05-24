//
// Created by Joel Jonsson on 07.05.20.
//

#ifndef PYLIBAPR_PYLINEARITERATOR_HPP
#define PYLIBAPR_PYLINEARITERATOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "data_structures/APR/APR.hpp"

namespace py = pybind11;


// -------- wrapper -------------------------------------------------
void AddLinearIterator(pybind11::module &m) {

    using namespace py::literals;

    py::class_<LinearIterator>(m, "LinearIterator")
            .def(py::init())
            .def("total_number_particles", &LinearIterator::total_number_particles,
                 "return number of particles up to level (by default level = 0 -> all levels)", "level"_a=0)
            .def("level_min", &LinearIterator::level_min, "return the minimum resolution level")
            .def("level_max", &LinearIterator::level_max, "return the maximum resolution level")
            .def("x_num", &LinearIterator::x_num,  "Gives the maximum bounds in the x direction for the given level", "level"_a)
            .def("y_num", &LinearIterator::y_num,  "Gives the maximum bounds in the y direction for the given level", "level"_a)
            .def("z_num", &LinearIterator::z_num,  "Gives the maximum bounds in the z direction for the given level", "level"_a)
            .def("y", &LinearIterator::get_y, "returns the y-coordinate of a given particle index", "idx"_a)
            .def("begin", &LinearIterator::begin,
                 "returns the index of the first particle in the sparse row (level, z, x)", "level"_a, "z"_a, "x"_a)
            .def("end", &LinearIterator::end,
                 "returns the (exclusive) end index of the current sparse row (level, z, x) opened using 'begin'");
}

#endif //PYLIBAPR_PYLINEARITERATOR_HPP
