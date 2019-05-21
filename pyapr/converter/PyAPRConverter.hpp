//
// Created by Joel Jonsson on 21.05.19.
//
#ifndef PYLIBAPR_PYAPRCONVERTER_HPP
#define PYLIBAPR_PYAPRCONVERTER_HPP

#include "algorithm/APRConverter.hpp"
#include "data_containers/PyAPR.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T>
class PyAPRConverter {

    APRConverter<T> converter;

public:

    PyAPRConverter() {}

    /**
     * set the verbose flag of the APRConverter.
     * @param val
     */
    void set_verbose(bool val) {
        converter.verbose = val;
    }

    /**
     * set the parameters to be used during conversion.
     * @param par
     */
    void set_parameters(APRParameters &par) {
        converter.par = par;
    }

    /**
     * Compute an APR from a given image, given as a numpy array.
     * @param aPyAPR
     * @param img
     */
    void get_apr(PyAPR &aPyAPR, py::array &img) {

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

        converter.get_apr(aPyAPR.apr, pd);
    }
};

template<typename DataType>
void AddPyAPRConverter(pybind11::module &m, const std::string &aTypeString) {
    using converter = PyAPRConverter<DataType>;
    std::string typeStr = aTypeString + "Converter";
    py::class_<converter>(m, typeStr.c_str())
            .def(py::init())
            .def("get_apr", &converter::get_apr, "compute APR from an image (input as a numpy array)")
            .def("set_parameters", &converter::set_parameters, "set parameters")
            .def("set_verbose", &converter::set_verbose,
                 "should timings and additional information be printed during conversion?");
}



#endif //PYLIBAPR_PYAPRCONVERTER_HPP
