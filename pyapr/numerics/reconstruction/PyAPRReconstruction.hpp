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
//template<typename DataType>
PyPixelData<float> recon_pc(PyAPR &aPyAPR, PyParticleData<float> &intensities) {

    PixelData<float> recon(aPyAPR.apr.y_num(aPyAPR.level_max()),
                           aPyAPR.apr.x_num(aPyAPR.level_max()),
                           aPyAPR.apr.z_num(aPyAPR.level_max()));

    APRReconstruction::interp_img(aPyAPR.apr, recon, intensities.parts);

    return PyPixelData<float>(recon);

}

void AddPyAPRReconstruction(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());
    m2.def("recon_pc", &recon_pc, "piecewise constant reconstruction");
}

