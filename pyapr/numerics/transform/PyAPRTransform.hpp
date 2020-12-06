
#ifndef PYLIBAPR_PYAPRTRANSFORM_HPP
#define PYLIBAPR_PYAPRTRANSFORM_HPP

#include "data_structures/APR/APR.hpp"

#include "data_containers/PyAPR.hpp"
#include "data_containers/PyParticleData.hpp"
#include <math.h>
#include <stdlib.h>
#include <algorithm>

namespace py = pybind11;


template<typename T>
void dilation_py(PyAPR& apr, PyParticleData<T>& parts, bool binary=false, int radius=1) {

    APRTimer timer(true);

    timer.start_timer("dilation");
    if(binary) {
        dilation_binary(apr.apr, parts.parts, radius);
    } else {
        dilation(apr.apr, parts.parts, radius);
    }
    timer.stop_timer();
}


template<typename T>
void dilation(APR& apr, ParticleData<T>& parts, const int radius) {
    auto apr_it = apr.random_iterator();
    auto neigh_it = apr.random_iterator();

    ParticleData<T> output;
    output.init(apr);

    for(int i = 0; i < radius; i++) {
        for(int level = apr.level_max(); level > apr.level_min(); --level) {

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it)
#endif
            for(int z = 0; z < apr_it.z_num(level); ++z) {
                for(int x = 0; x < apr_it.x_num(level); ++x) {
                    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        uint64_t ct_id = apr_it;
                        T val = parts[ct_id];

                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        // Edges are bidirectional, so we only need the positive directions
                        for (int direction = 0; direction < 6; direction++) {
                            apr_it.find_neighbours_in_direction(direction);

                            // For each face, there can be 0-4 neighbours
                            for (int index = 0; index < apr_it.number_neighbours_in_direction(direction); ++index) {
                                if (neigh_it.set_neighbour_iterator(apr_it, direction, index)) {
                                    //will return true if there is a neighbour defined

                                    val = std::max(val, parts[neigh_it]);
                                }
                            }
                        }
                        output[ct_id] = val;
                    }
                }
            }
        }
        output.swap(parts);
    }
}


template<typename T>
void dilation_binary(APR& apr, ParticleData<T>& parts, const int radius) {
    auto apr_it = apr.random_iterator();
    auto neigh_it = apr.random_iterator();

    ParticleData<T> output;
    output.init(apr);

    for(int i = 0; i < radius; i++) {
        for(int level = apr.level_max(); level > apr.level_min(); --level) {

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it)
#endif
            for(int z = 0; z < apr_it.z_num(level); ++z) {
                for(int x = 0; x < apr_it.x_num(level); ++x) {
                    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        uint64_t ct_id = apr_it;
                        if(find_positive_neighbour(apr_it, neigh_it, parts)) {
                            output[ct_id] = 1;
                        } else {
                            output[ct_id] = parts[ct_id];
                        }
                    }
                }
            }
        }
        output.swap(parts);
    }
}


template<typename T>
void erosion_py(PyAPR& apr, PyParticleData<T>& parts, bool binary=false, int radius=1) {

    APRTimer timer(true);
    timer.start_timer("erosion");
    if(binary) {
        erosion_binary(apr.apr, parts.parts, radius);
    } else {
        erosion(apr.apr, parts.parts, radius);
    }
    timer.stop_timer();
}


template<typename T>
void erosion(APR& apr, ParticleData<T>& parts, const int radius) {
    auto apr_it = apr.random_iterator();
    auto neigh_it = apr.random_iterator();

    ParticleData<T> output;
    output.init(apr);

    for(int i = 0; i < radius; i++) {
        for(int level = apr.level_max(); level > apr.level_min(); --level) {

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it)
#endif
            for(int z = 0; z < apr_it.z_num(level); ++z) {
                for(int x = 0; x < apr_it.x_num(level); ++x) {
                    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        uint64_t ct_id = apr_it;
                        T val = parts[ct_id];

                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        // Edges are bidirectional, so we only need the positive directions
                        for (int direction = 0; direction < 6; direction++) {
                            apr_it.find_neighbours_in_direction(direction);

                            // For each face, there can be 0-4 neighbours
                            for (int index = 0; index < apr_it.number_neighbours_in_direction(direction); ++index) {
                                if (neigh_it.set_neighbour_iterator(apr_it, direction, index)) {
                                    //will return true if there is a neighbour defined

                                    val = std::min(val, parts[neigh_it]);
                                }
                            }
                        }
                        output[ct_id] = val;
                    }
                }
            }
        }
        output.swap(parts);
    }
}


template<typename T>
void erosion_binary(APR& apr, ParticleData<T>& parts, const int radius) {
    auto apr_it = apr.random_iterator();
    auto neigh_it = apr.random_iterator();

    ParticleData<T> output;
    output.init(apr);

    for(int i = 0; i < radius; i++) {
        for(int level = apr.level_max(); level > apr.level_min(); --level) {

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it)
#endif
            for(int z = 0; z < apr_it.z_num(level); ++z) {
                for(int x = 0; x < apr_it.x_num(level); ++x) {
                    for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                        uint64_t ct_id = apr_it;
                        if(find_zero_neighbour(apr_it, neigh_it, parts)) {
                            output[ct_id] = 0;
                        } else {
                            output[ct_id] = parts[ct_id];
                        }
                    }
                }
            }
        }
        output.swap(parts);
    }
}


template<typename APRIteratorType, typename NeighbourIteratorType, typename T>
inline bool find_zero_neighbour(APRIteratorType& apr_it, NeighbourIteratorType& neigh_it, ParticleData<T>& parts) {
    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
    // Edges are bidirectional, so we only need the positive directions
    for (int direction = 0; direction < 6; direction++) {
        apr_it.find_neighbours_in_direction(direction);

        // For each face, there can be 0-4 neighbours
        for (int index = 0; index < apr_it.number_neighbours_in_direction(direction); ++index) {
            if (neigh_it.set_neighbour_iterator(apr_it, direction, index)) {
                //will return true if there is a neighbour defined
                if(!parts[neigh_it]) {
                    return true;
                }
            }
        }
    }
    return false;
}


template<typename APRIteratorType, typename NeighbourIteratorType, typename T>
inline bool find_positive_neighbour(APRIteratorType& apr_it, NeighbourIteratorType& neigh_it, ParticleData<T>& parts) {
    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
    // Edges are bidirectional, so we only need the positive directions
    for (int direction = 0; direction < 6; direction++) {
        apr_it.find_neighbours_in_direction(direction);

        // For each face, there can be 0-4 neighbours
        for (int index = 0; index < apr_it.number_neighbours_in_direction(direction); ++index) {
            if (neigh_it.set_neighbour_iterator(apr_it, direction, index)) {
                //will return true if there is a neighbour defined
                if(parts[neigh_it]) {
                    return true;
                }
            }
        }
    }
    return false;
}


template<typename T>
void find_perimeter_py(PyAPR& apr, PyParticleData<T>& parts, PyParticleData<T>& perimeter) {
    find_perimeter(apr.apr, parts.parts, perimeter.parts);
}


template<typename T>
void find_perimeter(APR& apr, ParticleData<T>& parts, ParticleData<T>& perimeter) {
    auto apr_it = apr.random_iterator();
    auto neigh_it = apr.random_iterator();

    perimeter.init(apr);  //values are initialized to 0

    for(int level = apr.level_max(); level > apr.level_min(); --level) {

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(apr_it, neigh_it)
#endif
        for(int z = 0; z < apr_it.z_num(level); ++z) {
            for(int x = 0; x < apr_it.x_num(level); ++x) {
                for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                    uint64_t ct_id = apr_it;
                    if(parts[ct_id] > 0){
                        if(find_zero_neighbour(apr_it, neigh_it, parts)) {
                            perimeter[ct_id] = 1;
                        }
                    }
                }
            }
        }
    }
}


void AddPyAPRTransform(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());
    m2.def("dilation", &dilation_py<uint16_t>,
           "computes a morphological dilation of the input. each particle takes the value of the maximum of its face-side neighbbours",
           py::arg("apr"), py::arg("parts"), py::arg("binary")=false, py::arg("radius")=1);
    m2.def("dilation", &dilation_py<float>,
           "computes a morphological dilation of the input. each particle takes the value of the maximum of its face-side neighbbours",
           py::arg("apr"), py::arg("parts"), py::arg("binary")=false, py::arg("radius")=1);
    m2.def("erosion", &erosion_py<uint16_t>,
           "computes a morphological erosion of the input. each particle takes the value of the minimum of its face-side neighbbours",
           py::arg("apr"), py::arg("parts"), py::arg("binary")=false, py::arg("radius")=1);
    m2.def("erosion", &erosion_py<float>,
           "computes a morphological erosion of the input. each particle takes the value of the minimum of its face-side neighbbours",
           py::arg("apr"), py::arg("parts"), py::arg("binary")=false, py::arg("radius")=1);

    m2.def("find_perimeter", &find_perimeter_py<uint16_t>, "find all positive particles with at least one zero neighbour",
           py::arg("apr"), py::arg("parts"), py::arg("perimeter"));
    m2.def("find_perimeter", &find_perimeter_py<float>, "find all positive particles with at least one zero neighbour",
           py::arg("apr"), py::arg("parts"), py::arg("perimeter"));
}

#endif //PYLIBAPR_PYAPRTRANSFORM_HPP
