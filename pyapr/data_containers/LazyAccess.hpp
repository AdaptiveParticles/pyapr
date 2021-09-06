//
// Created by joel on 03.09.21.
//

#ifndef PYLIBAPR_LAZYACCESS_HPP
#define PYLIBAPR_LAZYACCESS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <data_structures/APR/access/LazyAccess.hpp>

namespace py = pybind11;

void AddLazyAccess(pybind11::module &m, const std::string &modulename) {

    using namespace pybind11::literals;

    py::class_<LazyAccess>(m, modulename.c_str())
            .def(py::init())
            .def("init", &LazyAccess::init, "initialize LazyAccess from an open APRFile", "aprFile"_a)
            .def("init_tree", &LazyAccess::init_tree, "initialize LazyAccess for tree data from an open APRFile", "aprFile"_a)
            .def("open", &LazyAccess::open, "open file for reading")
            .def("close", &LazyAccess::close, "close file")
            .def("org_dims", &LazyAccess::org_dims, "original image dimensions");

}

#endif //PYLIBAPR_LAZYACCESS_HPP
