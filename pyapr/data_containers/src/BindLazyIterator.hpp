//
// Created by joel on 03.09.21.
//

#ifndef PYLIBAPR_LAZYITERATOR_HPP
#define PYLIBAPR_LAZYITERATOR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <data_structures/APR/access/LazyAccess.hpp>
#include <data_structures/APR/iterators/LazyIterator.hpp>

namespace py = pybind11;

void AddLazyIterator(pybind11::module &m) {

    py::class_<LazyIterator>(m, "LazyIterator")
            .def(py::init())
            .def(py::init([](LazyAccess& access){ return new LazyIterator(access); }));
}

#endif //PYLIBAPR_LAZYITERATOR_HPP
