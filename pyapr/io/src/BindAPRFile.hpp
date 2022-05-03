//
// Created by Joel Jonsson on 22.05.19.
//

#ifndef PYLIBAPR_PYAPRFILE_HPP
#define PYLIBAPR_PYAPRFILE_HPP

#include "io/APRFile.hpp"
#include "data_containers/src/BindParticleData.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

class PyAPRFile : public APRFile {
public:
    PyAPRFile() : APRFile() {}

    template<typename DataType>
    bool read_particles_py(APR& apr, std::string particles_name, PyParticleData<DataType>& particles,
                           bool apr_or_tree = true, uint64_t t = 0, std::string channel_name = "t") {
        return this->read_particles(apr, particles_name, particles, apr_or_tree, t, channel_name);
    }

    template<typename DataType>
    bool read_particles_py2(std::string particles_name, PyParticleData<DataType>& particles,
                           bool apr_or_tree = true, uint64_t t = 0, std::string channel_name = "t") {
        return this->read_particles(particles_name, particles, apr_or_tree, t, channel_name);
    }
};

// -------- wrapper -------------------------------------------------
void AddAPRFile(pybind11::module &m, const std::string &modulename) {

    using namespace py::literals;

    py::class_<APRFile>(m, "APRFile_CPP");

    py::class_<PyAPRFile, APRFile>(m, modulename.c_str())
            .def(py::init())
            .def("open", &PyAPRFile::open, "open a file for reading and/or writing",
                 "file_name"_a, "read_write"_a="WRITE")
            .def("close", &PyAPRFile::close, "close the file")
            .def("set_write_linear_flag", &PyAPRFile::set_write_linear_flag, "write linear access structure?",
                 "flag"_a)

            .def("get_particles_names", &PyAPRFile::get_particles_names, "return list of field names for stored particle values",
                 "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")
            .def("get_channel_names", &PyAPRFile::get_channel_names, "return list of channel names")

            .def("get_particle_type", &PyAPRFile::get_particle_type, "return type (string) of particle dataset",
                 "particles_name"_a, "apr_or_tree"_a=false, "t"_a=0, "channel_name"_a="t")

            .def("write_apr", &PyAPRFile::write_apr, "write apr to file",
                 "apr"_a, "t"_a=0, "channel_name"_a="t", "write_tree"_a=true)
            .def("read_apr", &PyAPRFile::read_apr, "read an APR from file",
                 "apr"_a, "t"_a=0, "channel_name"_a="t")

            .def("write_particles", &PyAPRFile::write_particles<uint8_t>, "write particles to file",
                 "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")
            .def("write_particles", &PyAPRFile::write_particles<uint16_t>, "write particles to file",
                 "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")
            .def("write_particles", &PyAPRFile::write_particles<uint64_t>, "write particles to file",
                 "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")
            .def("write_particles", &PyAPRFile::write_particles<float>, "write particles to file",
                 "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")

            .def("read_particles", &PyAPRFile::read_particles_py<uint8_t>, "read particles from file",
                 "apr"_a, "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")
            .def("read_particles", &PyAPRFile::read_particles_py<uint16_t>, "read particles from file",
                 "apr"_a, "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")
            .def("read_particles", &PyAPRFile::read_particles_py<uint64_t>, "read particles from file",
                 "apr"_a, "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")
            .def("read_particles", &PyAPRFile::read_particles_py<float>, "read particles from file",
                 "apr"_a, "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")

            .def("read_particles", &PyAPRFile::read_particles_py2<uint8_t>, "read particles from file",
                 "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")
            .def("read_particles", &PyAPRFile::read_particles_py2<uint16_t>, "read particles from file",
                 "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")
            .def("read_particles", &PyAPRFile::read_particles_py2<uint64_t>, "read particles from file",
                 "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")
            .def("read_particles", &PyAPRFile::read_particles_py2<float>, "read particles from file",
                 "particles_name"_a, "particles"_a, "apr_or_tree"_a=true, "t"_a=0, "channel_name"_a="t")

            .def("current_file_size_GB", &PyAPRFile::current_file_size_GB, "get current file size in GB")
            .def("current_file_size_MB", &PyAPRFile::current_file_size_MB, "get current file size in MB");
}

#endif //PYLIBAPR_PYAPRFILE_HPP
