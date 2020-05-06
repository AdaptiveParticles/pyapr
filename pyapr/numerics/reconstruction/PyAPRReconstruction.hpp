#include "data_structures/APR/APR.hpp"
#include "data_containers/PyPixelData.hpp"
#include "data_containers/PyAPR.hpp"
#include "numerics/APRReconstruction.hpp"

namespace py = pybind11;

// #TODO: should this be a templated class?


/**
 *
 * @param aPyAPR        a PyAPR object
 * @param intensities   a PyParticleData object
 * @return              the reconstructed image as a PyPixelData object
 */
template<typename DataType>
PyPixelData<DataType> recon_pc(PyAPR &aPyAPR, PyParticleData<DataType> &intensities) {

    PixelData<DataType> recon(aPyAPR.apr.y_num(aPyAPR.level_max()),
                           aPyAPR.apr.x_num(aPyAPR.level_max()),
                           aPyAPR.apr.z_num(aPyAPR.level_max()));

    APRReconstruction::interp_img(aPyAPR.apr, recon, intensities.parts);

    return PyPixelData<DataType>(recon);
}

template<typename DataType>
PyPixelData<DataType> recon_smooth(PyAPR &aPyAPR, PyParticleData<DataType> &intensities) {

    PixelData<DataType> recon(aPyAPR.apr.y_num(aPyAPR.level_max()),
                              aPyAPR.apr.x_num(aPyAPR.level_max()),
                              aPyAPR.apr.z_num(aPyAPR.level_max()));

    APRReconstruction::interp_parts_smooth(aPyAPR.apr, recon, intensities.parts);

    return PyPixelData<DataType>(recon);
}

template<typename DataType>
PyPixelData<DataType> recon_pc_patch(PyAPR &aPyAPR, PyParticleData<DataType> &intensities, int z_begin, int z_end,
                                     int x_begin, int x_end, int y_begin, int y_end) {

    ReconPatch patch_info;
    patch_info.z_begin = z_begin;
    patch_info.z_end = z_end;
    patch_info.x_begin = x_begin;
    patch_info.x_end = x_end;
    patch_info.y_begin = y_begin;
    patch_info.y_end = y_end;
    patch_info.level_delta = 0;

    ParticleData<float> tree_parts; // not needed unless level_delta < 0 (?)
    PixelData<DataType> recon;

    APRReconstruction::interp_image_patch(aPyAPR.apr, recon, intensities.parts, tree_parts, patch_info);

    return PyPixelData<DataType>(recon);
}

template<typename DataType>
PyPixelData<DataType> recon_smooth_patch(PyAPR &aPyAPR, PyParticleData<DataType> &intensities, int z_begin, int z_end,
                                     int x_begin, int x_end, int y_begin, int y_end) {

    APRReconstruction recon_fn;

    ReconPatch patch_info;
    patch_info.z_begin = z_begin;
    patch_info.z_end = z_end;
    patch_info.x_begin = x_begin;
    patch_info.x_end = x_end;
    patch_info.y_begin = y_begin;
    patch_info.y_end = y_end;
    patch_info.level_delta = 0;

    ParticleData<DataType> tree_parts; // not needed unless level_delta < 0 (?)
    PixelData<DataType> recon;

    recon_fn.interp_parts_smooth_patch(aPyAPR.apr, recon, intensities.parts, tree_parts, patch_info);

    return PyPixelData<DataType>(recon);
}

void AddPyAPRReconstruction(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());
    m2.def("recon_pc", &recon_pc<float>, "piecewise constant reconstruction",
            py::arg("APR"), py::arg("parts"));
    m2.def("recon_pc", &recon_pc<uint16_t>, "piecewise constant reconstruction",
            py::arg("APR"), py::arg("parts"));


    m2.def("recon_smooth", &recon_smooth<float>, "smooth reconstruction",
            py::arg("APR"), py::arg("parts"));
    m2.def("recon_smooth", &recon_smooth<uint16_t>, "smooth reconstruction",
            py::arg("APR"), py::arg("parts"));


    m2.def("recon_patch", &recon_pc_patch<float>, "piecewise constant patch reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("z_begin"), py::arg("z_end"),
           py::arg("x_begin"), py::arg("x_end"), py::arg("y_begin"), py::arg("y_end"));
    m2.def("recon_patch", &recon_pc_patch<uint16_t>, "piecewise constant patch reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("z_begin"), py::arg("z_end"),
           py::arg("x_begin"), py::arg("x_end"), py::arg("y_begin"), py::arg("y_end"));


    m2.def("recon_patch_smooth", &recon_smooth_patch<float>, "smooth patch reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("z_begin"), py::arg("z_end"),
           py::arg("x_begin"), py::arg("x_end"), py::arg("y_begin"), py::arg("y_end"));
    m2.def("recon_patch_smooth", &recon_smooth_patch<uint16_t>, "smooth patch reconstruction",
           py::arg("APR"), py::arg("parts"), py::arg("z_begin"), py::arg("z_end"),
           py::arg("x_begin"), py::arg("x_end"), py::arg("y_begin"), py::arg("y_end"));
}

