//
// Created by Joel Jonsson on 22.05.19.
//

#ifndef PYLIBAPR_PYAPRFILE_HPP
#define PYLIBAPR_PYAPRFILE_HPP

#include "io/APRFile.hpp"
#include "data_containers/PyAPR.hpp"
#include "data_containers/PyParticleData.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

class PyAPRFile {

    APRFile file;

public:

    PyAPRFile() {}

    bool open(std::string file_name,std::string read_write = "WRITE") {
        return file.open(file_name, read_write);
    }

    void close(){
        file.close();
    }

    void set_read_write_tree(bool with_tree_flag_){
        file.set_read_write_tree(with_tree_flag_);
    }

    void write_apr(PyAPR &aPyAPR, uint64_t t = 0, std::string channel_name = "t") {
        file.write_apr(aPyAPR.apr, t, channel_name);
    }

    void write_apr_append(PyAPR &aPyAPR) {
        file.write_apr_append(aPyAPR.apr);
    }

    template<typename DataType>
    void write_particles(PyAPR &aPyAPR, std::string particles_name, PyParticleData<DataType>& particles, uint64_t t = 0,
                         bool apr_or_tree = true, std::string channel_name = "t") {

        file.write_particles(aPyAPR.apr, particles_name, particles.parts, t, apr_or_tree, channel_name);
    }

    void read_apr(PyAPR &aPyAPR, uint64_t t = 0, std::string channel_name = "t"){
        file.read_apr(aPyAPR.apr, t, channel_name);
    }

    template<typename DataType>
    void read_particles(PyAPR &aPyAPR, std::string particles_name, PyParticleData<DataType>& particles, uint64_t t = 0,
                        bool apr_or_tree = true, std::string channel_name = "t"){

        file.read_particles(aPyAPR.apr, particles_name, particles.parts, t, apr_or_tree, channel_name);
    }

};

// -------- wrapper -------------------------------------------------
void AddPyAPRFile(pybind11::module &m, const std::string &modulename) {
    py::class_<PyAPRFile>(m, modulename.c_str())
            .def(py::init())
            .def("open", &PyAPRFile::open, py::arg("file_name"), py::arg("read_write")="WRITE", "open a file for reading and/or writing")
            .def("close", &PyAPRFile::close, "close the file")
            .def("set_read_write_tree", &PyAPRFile::set_read_write_tree, "set whether the internal APR Tree internal access should also be written and read.")
            .def("write_apr", &PyAPRFile::write_apr, py::arg("aPyAPR"), py::arg("t")=0, py::arg("channel_name")="t", "write apr to file")
            .def("write_apr_append", &PyAPRFile::write_apr_append, "write the APR to file and append it as the next time point")
            .def("write_particles", &PyAPRFile::write_particles<uint8_t>, py::arg("aPyAPR"), py::arg("particles_name"),
                 py::arg("particles"), py::arg("t")=0, py::arg("apr_or_tree")=true, py::arg("channel_name")="t", "write particles to file")
            .def("write_particles", &PyAPRFile::write_particles<uint16_t>, py::arg("aPyAPR"), py::arg("particles_name"),
                 py::arg("particles"), py::arg("t")=0, py::arg("apr_or_tree")=true, py::arg("channel_name")="t", "write particles to file")
            .def("write_particles", &PyAPRFile::write_particles<float>, py::arg("aPyAPR"), py::arg("particles_name"),
                 py::arg("particles"), py::arg("t")=0, py::arg("apr_or_tree")=true, py::arg("channel_name")="t", "write particles to file")
            .def("read_apr", &PyAPRFile::read_apr, py::arg("aPyAPR"), py::arg("t")=0, py::arg("channel_name")="t", "read an APR from file")
            .def("read_particles", &PyAPRFile::read_particles<uint8_t>, py::arg("aPyAPR"), py::arg("particles_name"), py::arg("particles"),
                 py::arg("t")=0, py::arg("apr_or_tree")=true, py::arg("channel_name")="t", "read particles from file")
            .def("read_particles", &PyAPRFile::read_particles<uint16_t>, py::arg("aPyAPR"), py::arg("particles_name"), py::arg("particles"),
                 py::arg("t")=0, py::arg("apr_or_tree")=true, py::arg("channel_name")="t", "read particles from file")
            .def("read_particles", &PyAPRFile::read_particles<float>, py::arg("aPyAPR"), py::arg("particles_name"), py::arg("particles"),
                 py::arg("t")=0, py::arg("apr_or_tree")=true, py::arg("channel_name")="t", "read particles from file");
}

#endif //PYLIBAPR_PYAPRFILE_HPP
