//
// Created by Krzysztof Gonciarz on 5/7/18.
// Modified by Joel Jonsson on 2/5/19.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ConfigAPR.h"
#include "data_structures/APR/APR.hpp"

#include "data_containers/PyPixelData.hpp"
#include "data_containers/PyAPR.hpp"
#include "data_containers/PyAPRParameters.hpp"
#include "data_containers/PyParticleData.hpp"
#include "nn/APRNetOps.hpp"
#include "numerics/reconstruction/PyAPRReconstruction.hpp"
#include "converter/PyAPRConverter.hpp"
#include "io/PyAPRFile.hpp"
#include "viewer/ViewerHelpers.hpp"

namespace py = pybind11;

// -------- Check if properly configured in CMAKE -----------------------------
#ifndef APR_PYTHON_MODULE_NAME
#error "Name of APR module (python binding) is not defined!"
#endif

// -------- Definition of python module ---------------------------------------
PYBIND11_MODULE(APR_PYTHON_MODULE_NAME, m) {
    m.doc() = "python binding for LibAPR library";
    m.attr("__version__") = py::str(ConfigAPR::APR_VERSION);

    py::module data_containers = m.def_submodule("data_containers");
    //wrap the PyAPR class
    AddPyAPR(data_containers, "APR");

    //wrap the PyPixelData class for different data types
    AddPyPixelData<uint8_t>(data_containers, "Byte");
    AddPyPixelData<uint16_t>(data_containers, "Short");
    AddPyPixelData<float>(data_containers, "Float");

    // wrap the APRParameters class
    AddPyAPRParameters(data_containers);

    // wrap the PyParticleData class for different data types
    AddPyParticleData<float>(data_containers, "Float");
    AddPyParticleData<uint16_t>(data_containers, "Short");


    py::module nn = m.def_submodule("nn");
    // wrap the APRNet operations #TODO: shouldnt be a class
    py::class_<APRNetOps>(nn, "APRNetOps")
            .def(py::init())
            .def("convolve", &APRNetOps::convolve, "nxn convolution with thread parallelism over the batch")
            .def("convolve_backward", &APRNetOps::convolve_backward, "backpropagation through convolve")
            .def("convolve3x3", &APRNetOps::convolve3x3, "3x3 convolution with thread parallelism over the batch")
            .def("convolve3x3_backward", &APRNetOps::convolve3x3_backward, "backpropagation through convolve1x1")
            .def("convolve1x1", &APRNetOps::convolve1x1, "1x1 convolution with thread parallelism over the batch")
            .def("convolve1x1_backward", &APRNetOps::convolve1x1_backward, "backpropagation through convolve1x1")
            .def("max_pool", &APRNetOps::max_pool, "max pooling of (current) max level particles")
            .def("max_pool_backward", &APRNetOps::max_pool_backward, "backpropagation through max_pool");

    py::module numerics = m.def_submodule("numerics");
    AddPyAPRReconstruction(numerics, "reconstruction");

    // wrap APRConverter for different data types
    py::module converter = m.def_submodule("converter");
    AddPyAPRConverter<float>(converter, "Float");
    AddPyAPRConverter<uint16_t>(converter, "Short");

    py::module io = m.def_submodule("io");
    AddPyAPRFile(io, "APRFile");

    py::module viewer = m.def_submodule("viewer");
    AddViewerHelpers(viewer,"viewerHelp");

}