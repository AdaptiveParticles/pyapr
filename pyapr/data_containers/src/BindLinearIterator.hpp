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

auto _find_particle= [](LinearIterator& it, const int z, const int x, const int y) -> uint64_t {
    if (z < 0 || z >= it.z_num(it.level_max()) ||
        x < 0 || x >= it.x_num(it.level_max()) ||
        y < 0 || y >= it.y_num(it.level_max())) {
            throw std::invalid_argument("LinearIterator::find_particle : coordinates (" + std::to_string(z) +
                                        ", " + std::to_string(x) + ", " + std::to_string(y) + ") out of bounds");
    }
    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        int z_l = z / it.level_size(level);
        int x_l = x / it.level_size(level);
        int y_l = y / it.level_size(level);
        for(it.begin(level, z_l, x_l); it < it.end(); ++it) {
            if(it.y() == y_l) {
                return it.global_index();
            }
        }
    }
    throw std::runtime_error("no particle found at (" + std::to_string(z) + ", " + std::to_string(x) +
                             ", " + std::to_string(y) + ")");
};


auto _find_coordinates = [](LinearIterator& it, const uint64_t idx) -> py::tuple {

    if(idx >= it.total_number_particles()) {
        throw std::invalid_argument("index " + std::to_string(idx) + " is out of bounds");
    }

    int level = it.level_min(), z=0, x=0, y=0;

    while(it.particles_level_end(level) <= idx) {
        level++;
    }

    const int z_num = it.z_num(level);
    const int x_num = it.x_num(level);

    it.begin(level, z, x_num-1);
    while(it.end() <= idx && z < z_num) {
        it.begin(level, ++z, x_num-1);
    }

    it.begin(level, z, x);
    while(it.end() <= idx && x < x_num) {
        it.begin(level, z, ++x);
    }

    y = it.get_y(idx);
    return py::make_tuple(level, z, x, y);
};


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
                 "returns the (exclusive) end index of the current sparse row (level, z, x) opened using 'begin'")
            .def("find_particle", _find_particle,
                 "return the particle index corresponding to a given pixel location", "z"_a, "x"_a, "y"_a)
            .def("find_coordinates", _find_coordinates,
                 "return the location of the particle at a given index as a tuple (level, z, x, y)", "idx"_a);
}

#endif //PYLIBAPR_PYLINEARITERATOR_HPP
