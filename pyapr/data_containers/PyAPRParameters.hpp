#include "algorithm/APRParameters.hpp"

void AddPyAPRParameters(pybind11::module &m) {
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
        .def_readwrite("gradient_smoothing", &APRParameters::lambda) // lambda is reserved in python
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
}