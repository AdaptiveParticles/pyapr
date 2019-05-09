//
// Created by Krzysztof Gonciarz on 5/7/18.
// Modified by Joel Jonsson on 2/5/19.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ConfigAPR.h"
#include "data_structures/APR/APR.hpp"

#include "PyPixelData.hpp"
#include "PyAPR.hpp"
#include "APRNetOps.hpp"


namespace py = pybind11;

// -------- Check if properly configured in CMAKE -----------------------------
#ifndef APR_PYTHON_MODULE_NAME
#error "Name of APR module (python binding) is not defined!"
#endif

// -------- Definition of python module ---------------------------------------
PYBIND11_MODULE(APR_PYTHON_MODULE_NAME, m) {
    m.doc() = "python binding for LibAPR library";
    m.attr("__version__") = py::str(ConfigAPR::APR_VERSION);

    //wrap the PyAPR class for different data types
    AddPyAPR<uint8_t>(m, "Byte");
    AddPyAPR<uint16_t>(m, "Short");
    AddPyAPR<float>(m, "Float");

    //wrap the PyPixelData class for different data types
    AddPyPixelData<uint8_t>(m, "Byte");
    AddPyPixelData<uint16_t>(m, "Short");
    AddPyPixelData<float>(m, "Float");

    // wrap the APRParameters class
    py::class_<APRParameters>(m, "APRParameters")
            .def(py::init())
            .def_readwrite("dx", &APRParameters::dx)
            .def_readwrite("dy", &APRParameters::dy)
            .def_readwrite("dz", &APRParameters::dz)
            .def_readwrite("psfx", &APRParameters::psfx)
            .def_readwrite("psfy", &APRParameters::psfy)
            .def_readwrite("psfz", &APRParameters::psfz)
            .def_readwrite("Ip_th", &APRParameters::Ip_th)
            .def_readwrite("SNR_min", &APRParameters::SNR_min)
            .def_readwrite("lmbda", &APRParameters::lambda) // lambda is reserved in python
            .def_readwrite("min_signal", &APRParameters::min_signal)
            .def_readwrite("rel_error", &APRParameters::rel_error)
            .def_readwrite("sigma_th", &APRParameters::sigma_th)
            .def_readwrite("sigma_th_max", &APRParameters::sigma_th_max)
            .def_readwrite("extra_smooth", &APRParameters::extra_smooth)
            .def_readwrite("noise_sd_estimate", &APRParameters::noise_sd_estimate)
            .def_readwrite("background_intensity_estimate", &APRParameters::background_intensity_estimate)
            .def_readwrite("auto_parameters", &APRParameters::auto_parameters)
            .def_readwrite("full_resolution", &APRParameters::full_resolution)
            .def_readwrite("normalized_input", &APRParameters::normalized_input)
            .def_readwrite("neighborhood_optimization", &APRParameters::neighborhood_optimization)
            .def_readwrite("output_steps", &APRParameters::output_steps)
            .def_readwrite("name", &APRParameters::name)
            .def_readwrite("output_dir", &APRParameters::output_dir)
            .def_readwrite("input_image_name", &APRParameters::input_image_name)
            .def_readwrite("input_dir", &APRParameters::input_dir)
            .def_readwrite("mask_file", &APRParameters::mask_file);


    // wrap the APRNet operations
    py::class_<APRNetOps>(m, "APRNetOps")
            .def(py::init())
            .def("convolve", &APRNetOps::convolve, "nxn convolution with thread parallelism over the batch")
            .def("convolve_backward", &APRNetOps::convolve_backward, "backpropagation through convolve")
            .def("convolve3x3", &APRNetOps::convolve3x3, "3x3 convolution with thread parallelism over the batch")
            .def("convolve3x3_backward", &APRNetOps::convolve3x3_backward, "backpropagation through convolve1x1")
            .def("convolve1x1", &APRNetOps::convolve1x1, "1x1 convolution with thread parallelism over the batch")
            .def("convolve1x1_backward", &APRNetOps::convolve1x1_backward, "backpropagation through convolve1x1")
            .def("max_pool", &APRNetOps::max_pool, "max pooling of (current) max level particles")
            .def("max_pool_backward", &APRNetOps::max_pool_backward, "backpropagation through max_pool");

}