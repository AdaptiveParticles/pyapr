//
// Created by Joel Jonsson on 05.11.20.
//
#ifndef PYLIBAPR_PYAPRCONVERTERBATCH_HPP
#define PYLIBAPR_PYAPRCONVERTERBATCH_HPP

#include "algorithm/APRConverterBatch.hpp"
#include "data_containers/PyAPR.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T>
class PyAPRConverterBatch {

    APRConverterBatch<T> converter;

public:

    PyAPRConverterBatch() {}

    /**
     * Set the block size to use in the conversion pipeline
     */
    void set_block_size(int z_block_size) { converter.z_block_size = z_block_size; }

    int get_block_size() { return converter.z_block_size; }

    /**
     *  Set the number of "ghost slices" to use on each side of the block during conversion
     */
    void set_ghost_size(int ghost_size) { converter.ghost_z = ghost_size; }

    int get_ghost_size() { return converter.ghost_z; }

    /**
     * set the verbose flag of the APRConverter.
     * @param val
     */
    void set_verbose(bool val) {
        converter.verbose = val;
    }

    /**
     * Set the parameters to be used during conversion.
     * @param par
     */
    void set_parameters(APRParameters &par) { converter.par = par; }

    APRParameters get_parameters() { return converter.par; }

    /**
     * Compute the APR using the image
     * @param aPyAPR
     * @return True if successful, False otherwise
     */
    bool get_apr(PyAPR &aPyAPR) {
        return converter.get_apr(aPyAPR.apr);
    }
};

template<typename DataType>
void AddPyAPRConverterBatch(pybind11::module &m, const std::string &aTypeString) {
    using converter = PyAPRConverterBatch<DataType>;
    std::string typeStr = aTypeString + "ConverterBatch";
    py::class_<converter>(m, typeStr.c_str())
            .def(py::init())
            .def("get_apr", &converter::get_apr, "compute APR from an image (input as a numpy array)")
            .def("set_parameters", &converter::set_parameters, "set parameters")
            .def("get_parameters", &converter::get_parameters, "get parameters")
            .def("set_verbose", &converter::set_verbose,
                 "should timings and additional information be printed during conversion?")
            .def("set_block_size", &converter::set_block_size,
                 "The image is read and processed in blocks of this number of z-slices")
            .def("get_block_size", &converter::get_block_size, "return the block size")
            .def("set_ghost_size", &converter::set_ghost_size,
                 "set the number of \'ghost slices\' to use on each side of the block during conversion")
            .def("get_ghost_size", &converter::get_ghost_size, "return the number of ghost slices used");
}



#endif //PYLIBAPR_PYAPRCONVERTERBATCH_HPP
