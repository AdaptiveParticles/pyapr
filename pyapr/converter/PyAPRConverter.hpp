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

    void get_apr_step1(PyAPR &aPyAPR, py::array &img){

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

        converter.par.output_steps = true;

        converter.par.grad_th = 20;

        converter.get_lrf(aPyAPR.apr, pd);
        //converter.get_apr(aPyAPR.apr, pd);
        //converter.get_ds(aPyAPR.apr);

    }

    void get_apr_step2(PyAPR &aPyAPR, APRParameters& par){

        converter.par = par;

        converter.get_ds(aPyAPR.apr);

    }

     static inline uint32_t asmlog_2(const uint32_t x) {
        if (x == 0) return 0;
        return (31 - __builtin_clz (x));
    }

    void get_level_slice(int z_slice, py::array &input, APRParameters& par,PyAPR &aPyAPR){

        PixelData<T>& grad_ = converter.grad_temp;
        PixelData<float>& lis_ = converter.local_scale_temp;
        PixelData<float>& smooth_image_ = converter.local_scale_temp2;

        converter.par = par;

        auto buf = input.request();
        auto *ptr = static_cast<float*>(buf.ptr);
        PixelData<float> level_output;
        level_output.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr); // may lead to memory issues

        const float scale_factor = pow(2,aPyAPR.apr.level_max())/par.rel_error;

        const uint32_t l_max = aPyAPR.apr.level_max();

        for(int x = 0; x < level_output.x_num; x++){
            for(int y = 0;y < level_output.y_num; y++){

               auto grad_temp = grad_.at(y,x,z_slice);
               auto lis_temp = lis_.at(y,x,z_slice);
               auto intensity = smooth_image_.at(y,x,z_slice);

               if(grad_temp < par.grad_th){
                    grad_temp = 0;
               }

               if(lis_temp < par.sigma_th){
                    lis_temp = par.sigma_th;
               }

               if(intensity < par.Ip_th){
                    grad_temp = 0;
               }

               auto level = std::min(asmlog_2(scale_factor*grad_temp/lis_temp),l_max);
               if(level < (l_max-1)){
                    level = l_max - 2;
               }

               level_output.at(y,x,0) = level;

            }
        }

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
                 "should timings and additional information be printed during conversion?")
            .def("get_level_slice", &converter::get_level_slice,
                 "gets the current level slice for the applied parameters")
            .def("get_apr_step1", &converter::get_apr_step1,
                 "Interactive APR generation")
            .def("get_apr_step2", &converter::get_apr_step2,
                 "Interactive APR generation");
}



#endif //PYLIBAPR_PYAPRCONVERTER_HPP
