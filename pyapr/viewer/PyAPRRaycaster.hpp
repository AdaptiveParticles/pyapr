#ifndef PYLIBAPR_PYRAYCASTER_HPP
#define PYLIBAPR_PYRAYCASTER_HPP



#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "numerics/APRRaycaster.hpp"

namespace py = pybind11;

/**
 * @tparam T data type of particle values
 */

class PyAPRRaycaster {

    APRRaycaster apr_raycaster;

    float current_angle = -3.14;

    ReconPatch rp;


public:

    PyAPRRaycaster() {
        rp.level_delta = 0;
    }

    void set_angle(float angle){
        current_angle = angle;
    }

    void set_level_delta(int level_delta){
        this->rp.level_delta = level_delta;
    }

    void set_z_anisotropy(float aniso){
        apr_raycaster.scale_z = aniso;
    }

    void set_radius(float radius){
        apr_raycaster.radius_factor = radius;
    }


    void get_view(PyAPR &aPyAPR, PyParticleData<uint16_t> &particles,PyParticleData<float> &particles_tree,py::array &input) {

        auto buf = input.request();

        auto *ptr = static_cast<uint16_t*>(buf.ptr);

        PixelData<uint16_t> input_img;

        input_img.init_from_mesh(buf.shape[0], buf.shape[1], 1, ptr); // may lead to memory issues

        apr_raycaster.theta_0 = current_angle; //start
        apr_raycaster.theta_final = 0.001; //stop radians
        apr_raycaster.theta_delta = (apr_raycaster.theta_final - apr_raycaster.theta_0); //steps


        apr_raycaster.scale_down = pow(2,rp.level_delta);

        apr_raycaster.perform_raycast_patch(aPyAPR.apr,particles.parts,particles_tree.parts,input_img,rp,[] (const uint16_t& a,const uint16_t& b) {return std::max(a,b);});

        //apr_raycaster.perform_raycast(aPyAPR.apr,particles.parts,input_img,[] (const uint16_t& a,const uint16_t& b) {return std::max(a,b);});
    }


};

void AddPyAPRRaycaster(pybind11::module &m,  const std::string &modulename) {

     py::class_<PyAPRRaycaster>(m, modulename.c_str())
        .def(py::init())
        .def("set_angle",&PyAPRRaycaster::set_angle, "demo")
        .def("set_level_delta",&PyAPRRaycaster::set_level_delta, "demo")
        .def("set_z_anisotropy",&PyAPRRaycaster::set_z_anisotropy, "demo")
        .def("set_radius",&PyAPRRaycaster::set_radius, "demo")
        .def("get_view", &PyAPRRaycaster::get_view, "demo");
}

#endif //PYLIBAPR_PYPARTICLEDATA_HPP