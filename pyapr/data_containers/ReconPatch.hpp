//
// Created by joel on 01.03.21.
//

#ifndef PYLIBAPR_RECONPATCH_HPP
#define PYLIBAPR_RECONPATCH_HPP

#include "numerics/APRReconstruction.hpp"


struct PyReconPatch : ReconPatch {
    inline bool check_limits_py(PyAPR& apr) {
        return check_limits(apr.apr);
    }

    inline size_t size() {
        return (size_t)(z_end-z_begin)*(x_end-x_begin)*(y_end-y_begin);
    }
};

void AddReconPatch(pybind11::module &m) {

    py::class_<ReconPatch>(m, "_ReconPatchCPP");

    py::class_<PyReconPatch, ReconPatch>(m, "ReconPatch")
            .def(py::init())
            .def("__repr__", [](ReconPatch& p) {
                return "ReconPatch: z: [" + std::to_string(p.z_begin) + ", " + std::to_string(p.z_end) + ")" +
                        ", x: [" + std::to_string(p.x_begin) + ", " + std::to_string(p.x_end) + ")" +
                        ", y: [" + std::to_string(p.y_begin) + ", " + std::to_string(p.y_end) + ")" +
                        ", level_delta = " + std::to_string(p.level_delta);
            })
            .def_readwrite("z_begin", &PyReconPatch::z_begin)
            .def_readwrite("z_end", &PyReconPatch::z_end)
            .def_readwrite("x_begin", &PyReconPatch::x_begin)
            .def_readwrite("x_end", &PyReconPatch::x_end)
            .def_readwrite("y_begin", &PyReconPatch::y_begin)
            .def_readwrite("y_end", &PyReconPatch::y_end)
            .def_readwrite("level_delta", &PyReconPatch::level_delta)
            .def("check_limits", &PyReconPatch::check_limits_py, "check patch limits against APR dimensions")
            .def("size", &PyReconPatch::size, "return the number of pixels in the patch region");
}


#endif //PYLIBAPR_RECONPATCH_HPP
