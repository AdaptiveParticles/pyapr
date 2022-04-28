//
// Created by Joel Jonsson on 29.06.18.
//

#ifndef PYLIBAPR_PYAPR_HPP
#define PYLIBAPR_PYAPR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#include <data_structures/APR/APR.hpp>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<APR*>)

void AddPyAPR(pybind11::module &m, const std::string &modulename) {
    py::class_<APR>(m, modulename.c_str())
            .def(py::init())
            .def("__repr__", [](APR& a) {
                return "APR(shape [" + std::to_string(a.org_dims(2)) + ", " + std::to_string(a.org_dims(1)) +
                        ", " + std::to_string(a.org_dims(0)) + "], " + std::to_string(a.total_number_particles()) + " particles)";})
            .def_readwrite("name", &APR::name)
            .def("total_number_particles", &APR::total_number_particles, "return number of particles")
            .def("total_number_tree_particles", &APR::total_number_tree_particles, "return number of interior tree particles")
            .def("level_min", &APR::level_min, "return the minimum resolution level")
            .def("level_max", &APR::level_max, "return the maximum resolution level")
            .def("x_num", &APR::x_num,  "Gives the maximum bounds in the x direction for the given level")
            .def("y_num", &APR::y_num,  "Gives the maximum bounds in the y direction for the given level")
            .def("z_num", &APR::z_num,  "Gives the maximum bounds in the z direction for the given level")
            .def("iterator", &APR::iterator, "Return a linear iterator for APR particles")
            .def("tree_iterator", &APR::tree_iterator, "Return a linear iterator for interior APRTree particles")
            .def("org_dims", &APR::org_dims, "returns the original image size in a specified dimension (y, x, z)")
            .def("shape", [](APR& self){return py::make_tuple(self.org_dims(2), self.org_dims(1), self.org_dims(0));}, "returns the original pixel image dimensions as a tuple (z, x, y)")
            .def("get_parameters", &APR::get_apr_parameters, "return the parameters used to create the APR")
            .def("computational_ratio", &APR::computational_ratio, "return the computational ratio (number of pixels in original image / number of particles in the APR)");

    py::bind_vector<std::vector<APR*>>(m, "APRPtrVector", py::module_local(false));
}


#endif //PYLIBAPR_PYAPR_HPP
