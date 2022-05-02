
#ifndef PYLIBAPR_BINDMORPHOLOGY_HPP
#define PYLIBAPR_BINDMORPHOLOGY_HPP

#include <pybind11/pybind11.h>

#include "data_structures/APR/APR.hpp"
#include "data_containers/src/BindParticleData.hpp"

#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <set>

namespace py = pybind11;
using namespace py::literals;


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
void binary_dilation(APR& apr, ParticleData<T>& parts, const int radius) {
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
void binary_erosion(APR& apr, ParticleData<T>& parts, const int radius) {
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


/**
 * Remove objects smaller than a given volume (in pixels). Assumes that each object is labeled with a distinct number,
 * @param min_volume
 *
 * for example the output of connected_component.
 * @param apr
 * @param object_labels
 * @param min_volume
 */
template<typename T>
void remove_small_objects(APR& apr, PyParticleData<T>& object_labels, const uint64_t min_volume) {
    auto max_label = object_labels.max();

    std::vector<uint64_t> bin_counts(max_label+1, 0);
    auto it = apr.iterator();
    const int ndim = it.number_dimensions();

    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        const int particle_volume = std::pow(2, ndim*(it.level_max() - level));

        for(uint64_t idx = it.particles_level_begin(level); idx < it.particles_level_end(level); ++idx) {
            bin_counts[object_labels[idx]] += particle_volume;
        }
    }

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t idx = 0; idx < it.total_number_particles(); ++idx) {
        if(bin_counts[object_labels[idx]] < min_volume) {
            object_labels[idx] = 0;
        }
    }
}


/**
 * Remove objects larger than a given volume (in pixels). Assumes that each object is labeled with a distinct number,
 * for example the output of connected_component.
 * @param apr
 * @param object_labels
 * @param min_volume
 */
template<typename T>
void remove_large_objects(APR& apr, PyParticleData<T>& object_labels, const uint64_t max_volume) {
    auto max_label = object_labels.max();

    std::vector<uint64_t> bin_counts(max_label+1, 0);
    auto it = apr.iterator();
    const int ndim = it.number_dimensions();

    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        const int particle_volume = std::pow(2, ndim*(it.level_max() - level));

        for(uint64_t idx = it.particles_level_begin(level); idx < it.particles_level_end(level); ++idx) {
            bin_counts[object_labels[idx]] += particle_volume;
        }
    }

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(size_t idx = 0; idx < it.total_number_particles(); ++idx) {
        if(bin_counts[object_labels[idx]] > max_volume) {
            object_labels[idx] = 0;
        }
    }
}




template<typename T>
void remove_edge_objects(APR& apr, PyParticleData<T>& object_labels, const T background_label=0,
                         const bool z_edges=false, const bool x_edges=true, const bool y_edges=true) {

    auto it = apr.iterator();

    // step 1: find unique labels intersecting with the volume edges in the specified directions
    std::set<T> edge_labels;

    if(z_edges) {
        for(int level = it.level_max(); level >= it.level_min(); --level) {
            for(auto z : {0, it.z_num(level)-1}) {
                for(int x = 0; x < it.x_num(level); ++x) {
                    for(it.begin(level, z, x); it < it.end(); ++it) {
                        if(object_labels[it] != background_label) {
                            edge_labels.insert(object_labels[it]);
                        }
                    }
                }
            }
        }
    }

    if(x_edges) {
        for(int level = it.level_max(); level >= it.level_min(); --level) {
            for(auto x : {0, it.x_num(level)-1}) {
                for(int z = 0; z < it.z_num(level); ++z) {
                    for(it.begin(level, z, x); it < it.end(); ++it) {
                        if(object_labels[it] != background_label) {
                            edge_labels.insert(object_labels[it]);
                        }
                    }
                }
            }
        }
    }

    if(y_edges) {
        for(int level = it.level_max(); level >= it.level_min(); --level) {
            for(int z = 0; z < it.z_num(level); ++z) {
                for(int x = 0; x < it.x_num(level); ++x) {
                    it.begin(level, z, x);
                    if(it < it.end()) {
                        if( (it.y() == 0) && (object_labels[it] != background_label) ) {
                            edge_labels.insert(object_labels[it]);
                        }

                        const uint64_t last_idx = it.end() - 1;
                        if( (it.get_y(last_idx) == (it.y_num(level)-1)) && (object_labels[it] != background_label) ) {
                            edge_labels.insert(object_labels[it]);
                        }
                    }
                }
            }
        }
    }

    // if no edge labels were found, we are done
    if(edge_labels.empty()) { return; }

    // set edge labels to background_label
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
    for(size_t idx = 0; idx < object_labels.size(); ++idx) {
        if(object_labels[idx] != background_label) {
            // if current label is in edge_labels, set it to the background value
            if(edge_labels.find(object_labels[idx]) != edge_labels.end()) {
                object_labels[idx] = background_label;
            }
        }
    }
}


template<typename T>
void bindDilation(py::module &m) {
    m.def("dilation", &dilation<T>, "computes a grayscale dilation of the input",
          "apr"_a, "parts"_a, "radius"_a=1);

    m.def("binary_dilation", &binary_dilation<T>, "computes a binary dilation of the input",
          "apr"_a, "parts"_a, "radius"_a=1);
}

template<typename T>
void bindErosion(py::module &m) {
    m.def("erosion", &erosion<T>, "computes a grayscale erosion of the input",
          "apr"_a, "parts"_a, "radius"_a=1);

    m.def("binary_erosion", &binary_erosion<T>, "computes a binary erosion of the input",
          "apr"_a, "parts"_a, "radius"_a=1);
}


template<typename T>
void bindFindPerimeter(py::module &m) {
    m.def("find_perimeter", &find_perimeter<T>,
          "find all positive particles with at least one zero neighbour",
          "apr"_a, "parts"_a, "perimeter"_a);
}


template<typename T>
void bindRemoveObjects(py::module &m) {
    m.def("remove_small_objects", &remove_small_objects<T>, "remove objects smaller than a threshold",
          "apr"_a, "object_labels"_a, "min_volume"_a);

    m.def("remove_large_objects", &remove_large_objects<T>, "remove objects larger than a threshold"
          "apr"_a, "object_labels"_a, "max_volume"_a);

    m.def("remove_edge_objects", &remove_edge_objects<T>, "remove object labels intersecting with an edge",
          "apr"_a, "object_labels"_a, "background_label"_a=0, "z_edges"_a=true, "x_edges"_a=true, "y_edges"_a=true);
}


void AddMorphology(py::module &m) {

    bindDilation<uint8_t>(m);
    bindDilation<uint16_t>(m);
    bindDilation<uint64_t>(m);
    bindDilation<float>(m);

    bindErosion<uint8_t>(m);
    bindErosion<uint16_t>(m);
    bindErosion<uint64_t>(m);
    bindErosion<float>(m);

    bindFindPerimeter<uint8_t>(m);
    bindFindPerimeter<uint16_t>(m);
    bindFindPerimeter<uint64_t>(m);
    bindFindPerimeter<float>(m);

    bindRemoveObjects<uint8_t>(m);
    bindRemoveObjects<uint16_t>(m);
    bindRemoveObjects<uint64_t>(m);
}

#endif //PYLIBAPR_BINDMORPHOLOGY_HPP
