//
// Created by Joel Jonsson on 22.05.19.
//

#ifndef PYLIBAPR_PYAPRFILE_HPP
#define PYLIBAPR_PYAPRFILE_HPP

#include "io/APRFile.hpp"
#include "data_containers/PyAPR.hpp"
#include "data_containers/PyParticleData.hpp"

namespace py = pybind11;

//FIXME
class PyAPRFile {

    APRFile file;

public:

    PyAPRFile() {}

    void write_apr(PyAPR &aPyAPR) {

        file.write_apr(aPyAPR.apr);

    }

    void write_particles(PyAPR &aPyAPR, std::string particles_name, PyParticleData<float> &partdata) {

        file.write_particles(aPyAPR.apr, particles_name, partdata.parts);

    }

};

// -------- wrapper -------------------------------------------------
void AddPyAPRFile(pybind11::module &m, const std::string &modulename) {
    py::class_<PyAPRFile>(m, modulename.c_str())
            .def(py::init())
            .def("write_apr", &PyAPRFile::write_apr, "write apr to file")
            .def("write_particles", &PyAPRFile::write_particles, "write particles to file");
}

#endif //PYLIBAPR_PYAPRFILE_HPP
