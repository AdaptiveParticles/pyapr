//
// Created by joel on 01.03.21.
//

#ifndef PYLIBAPR_RECONPATCH_HPP
#define PYLIBAPR_RECONPATCH_HPP

#include "numerics/APRReconstruction.hpp"

inline void _check_patch(ReconPatch& patch, APR& apr) {

    int max_img_y = ceil(apr.org_dims(0)*pow(2.0, patch.level_delta));
    int max_img_x = ceil(apr.org_dims(1)*pow(2.0, patch.level_delta));
    int max_img_z = ceil(apr.org_dims(2)*pow(2.0, patch.level_delta));

    patch.y_begin = std::max(0, patch.y_begin);
    patch.x_begin = std::max(0, patch.x_begin);
    patch.z_begin = std::max(0, patch.z_begin);

    patch.y_end = patch.y_end < 0 ? max_img_y : std::min(max_img_y, patch.y_end);
    patch.x_end = patch.x_end < 0 ? max_img_x : std::min(max_img_x, patch.x_end);
    patch.z_end = patch.z_end < 0 ? max_img_z : std::min(max_img_z, patch.z_end);

    if(patch.y_begin >= patch.y_end || patch.x_begin >= patch.x_end || patch.z_begin >= patch.z_end) {
        std::wcerr << "ReconPatch expects begin < end in all dimensions" << std::endl;
    }
}

void check_patch(ReconPatch& patch, PyAPR& apr) {
    _check_patch(patch, apr.apr);
}

void AddReconPatch(pybind11::module &m) {
    py::class_<ReconPatch>(m, "ReconPatch")
            .def(py::init())
            .def("__repr__", [](ReconPatch& p) {
                return "ReconPatch: z: [" + std::to_string(p.z_begin) + ", " + std::to_string(p.z_end) + ")" +
                        ", x: [" + std::to_string(p.x_begin) + ", " + std::to_string(p.x_end) + ")" +
                        ", y: [" + std::to_string(p.y_begin) + ", " + std::to_string(p.y_end) + ")" +
                        ", level_delta = " + std::to_string(p.level_delta);
            })
            .def_readwrite("z_begin", &ReconPatch::z_begin)
            .def_readwrite("z_end", &ReconPatch::z_end)
            .def_readwrite("x_begin", &ReconPatch::x_begin)
            .def_readwrite("x_end", &ReconPatch::x_end)
            .def_readwrite("y_begin", &ReconPatch::y_begin)
            .def_readwrite("y_end", &ReconPatch::y_end)
            .def_readwrite("level_delta", &ReconPatch::level_delta);

    m.def("check_patch", &check_patch);
}


#endif //PYLIBAPR_RECONPATCH_HPP
