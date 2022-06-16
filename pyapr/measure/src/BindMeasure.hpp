
#ifndef PYLIBAPR_BINDMEASURE_HPP
#define PYLIBAPR_BINDMEASURE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#include "data_structures/APR/APR.hpp"
#include "data_containers/src/BindParticleData.hpp"

#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <set>

namespace py = pybind11;
using namespace py::literals;


template<typename T>
void find_objects(APR& apr, PyParticleData<T>& labels, py::array_t<int>& min_coords, py::array_t<int>& max_coords) {
    // assumes min_coords initialized to a >= the maximum (original) dimension and max_coords to 0
    auto minc = min_coords.mutable_unchecked<2>(); // Will throw if ndim != 2 or flags.writeable is false
    auto maxc = max_coords.mutable_unchecked<2>();

    auto it = apr.iterator();
    const int z_num = it.z_num(it.level_max());
    const int x_num = it.x_num(it.level_max());
    const int y_num = it.y_num(it.level_max());

    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        const int lsize = it.level_size(level);
        for(int z = 0; z < it.z_num(level); ++z) {
            const int zl = z * lsize;
            const int zh = std::min(zl + lsize, z_num);
            for(int x = 0; x < it.x_num(level); ++x) {
                const int xl = x * lsize;
                const int xh = std::min(xl + lsize, x_num);
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    if(labels[it] > 0) {
                        const int lab = labels[it];
                        const int yl = it.y() * lsize;
                        const int yh = std::min(yl + lsize, y_num);

                        minc(lab, 0) = std::min(minc(lab, 0), zl);
                        minc(lab, 1) = std::min(minc(lab, 1), xl);
                        minc(lab, 2) = std::min(minc(lab, 2), yl);

                        maxc(lab, 0) = std::max(maxc(lab, 0), zh);
                        maxc(lab, 1) = std::max(maxc(lab, 1), xh);
                        maxc(lab, 2) = std::max(maxc(lab, 2), yh);
                    }
                }
            }
        }
    }
}



template<typename T>
void find_label_centers(APR& apr, PyParticleData<T>& object_labels, py::array_t<double>& coords) {
    auto res = coords.mutable_unchecked<2>();
    auto it = apr.iterator();
    const int ndim = it.number_dimensions();
    std::vector<float> denominator(res.shape(0), 0.0f);

    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        const float psize = std::pow(it.level_size(level), ndim);
        for(int z = 0; z < it.z_num(level); ++z) {
            for(int x = 0; x < it.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    if(object_labels[it] > 0) {
                        res(object_labels[it], 0) += it.z_global(level, z) * psize;
                        res(object_labels[it], 1) += it.x_global(level, x) * psize;
                        res(object_labels[it], 2) += it.y_global(level, it.y()) * psize;
                        denominator[object_labels[it]] += psize;
                    }
                }
            }
        }
    }
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
    for(size_t i = 0; i < (size_t) res.shape(0); ++i) {
        if(denominator[i] > 1e-3) {
            res(i, 0) /= denominator[i];
            res(i, 1) /= denominator[i];
            res(i, 2) /= denominator[i];
        } else {
            res(i, 0) = -1;
            res(i, 1) = -1;
            res(i, 2) = -1;
        }
    }
}


template<typename T, typename S>
void find_label_centers_weighted(APR& apr, PyParticleData<T>& object_labels, py::array_t<double>& coords, ParticleData<S>& weights) {
    auto res = coords.mutable_unchecked<2>();
    auto it = apr.iterator();
    const int ndim = it.number_dimensions();
    std::vector<float> denominator(res.shape(0), 0.0f);

    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        const float psize = std::pow(it.level_size(level), ndim);
        for(int z = 0; z < it.z_num(level); ++z) {
            for(int x = 0; x < it.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    if(object_labels[it] > 0) {
                        res(object_labels[it], 0) += it.z_global(level, z) * psize * weights[it];
                        res(object_labels[it], 1) += it.x_global(level, x) * psize * weights[it];
                        res(object_labels[it], 2) += it.y_global(level, it.y()) * psize * weights[it];
                        denominator[object_labels[it]] += psize * weights[it];
                    }
                }
            }
        }
    }
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
    for(size_t i = 0; i < (size_t) res.shape(0); ++i) {
        if(denominator[i] > 1e-3) {
            res(i, 0) /= denominator[i];
            res(i, 1) /= denominator[i];
            res(i, 2) /= denominator[i];
        } else {
            res(i, 0) = -1;
            res(i, 1) = -1;
            res(i, 2) = -1;
        }
    }
}



template<typename T>
void find_label_volume(APR& apr, PyParticleData<T>& object_labels, py::array_t<uint64_t>& volume) {

    auto max_label = object_labels.max();
    auto res = volume.mutable_unchecked<1>();

    std::vector<uint64_t> bin_counts(max_label+1, 0);
    auto it = apr.iterator();
    const int ndim = it.number_dimensions();

    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        const int particle_volume = std::pow(2, ndim*(it.level_max() - level));

        for(uint64_t idx = it.particles_level_begin(level); idx < it.particles_level_end(level); ++idx) {
            res(object_labels[idx]) += particle_volume;
        }
    }
}


/// -----------------------------------------
/// APR connected component
/// -----------------------------------------

/* Find the equivalence class corresponding to a given label */
inline int uf_find(int x, std::vector<int>& labels) {
    while (x != labels[x]) {
        x = labels[x];
    }
    return x;
}


/* Merge each step from x to target. Assumes labels[target] = target */
inline void uf_merge_path(int x, int target, std::vector<int>& labels) {
    while(x != target) {
        int tmp = x;
        x = labels[x];
        labels[tmp] = target;
    }
}


/* Join two equivalence classes and return the minimum label */
inline int uf_union(int x, int y, std::vector<int>& labels) {
    const int orig_x = x;
    const int orig_y = y;

    // find mininum label
    int lx = uf_find(x, labels);
    int ly = uf_find(y, labels);
    const int minlabel = std::min(lx, ly);

    // merge roots
    labels[ly] = minlabel;
    labels[lx] = minlabel;

    // merge each step to minlevel
    uf_merge_path(orig_x, minlabel, labels);
    uf_merge_path(orig_y, minlabel, labels);

    return minlabel;
}


/*  Create a new equivalence class and returns its label */
inline int uf_make_set(std::vector<int>& labels) {
    labels[0]++;
    labels.push_back(labels[0]);
    return labels[0];
}


/**
 * Compute connected component labels from a binary mask, using face-side connectivity. That is, two segments are
 * considered connected if they share a common particle cell face. Diagonal neighbours are not considered connected.
 * @param apr
 * @param binary_mask
 * @param component_labels
 */
template<typename maskType, typename labelType>
void calc_connected_component(APR& apr, PyParticleData<maskType>& binary_mask, PyParticleData<labelType>& component_labels) {

    component_labels.init(apr);

    APRTimer timer(false);

    auto apr_it = apr.random_iterator();
    auto neigh_it = apr.random_iterator();

    std::vector<int> labels;
    labels.resize(1, 0);

    timer.start_timer("connected component first loop");

    // iterate over particles
    for(int level = apr_it.level_min(); level <= apr_it.level_max(); ++level) {
        for(int z = 0; z < apr_it.z_num(level); ++z) {
            for(int x = 0; x < apr_it.x_num(level); ++x) {
                for(apr_it.begin(level, z, x); apr_it < apr_it.end(); ++apr_it) {

                    if(binary_mask[apr_it] > 0) {

                        std::vector<int> neigh_labels;

                        // iterate over face-side neighbours in y, x, z directions
                        // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                        for (int direction = 0; direction < 6; direction++) {
                            apr_it.find_neighbours_in_direction(direction);

                            // For each face, there can be 0-4 neighbours
                            for (int index = 0; index < apr_it.number_neighbours_in_direction(direction); ++index) {
                                if (neigh_it.set_neighbour_iterator(apr_it, direction, index)) {

                                    if (component_labels[neigh_it] > 0) {
                                        neigh_labels.push_back(component_labels[neigh_it]);
                                    }
                                }
                            }
                        }

                        if(neigh_labels.size() == 0) {
                            // no neighbour labels, make new region
                            component_labels[apr_it] = uf_make_set(labels);

                        } else if (neigh_labels.size() == 1){
                            // one neighbour label, set mask to that label
                            component_labels[apr_it] = neigh_labels[0];

                        } else {
                            // multiple neighbour regions, resolve
                            int curr_label = neigh_labels[0];
                            for(int n = 0; n < ((int)neigh_labels.size()-1); ++n){
                                curr_label = uf_union(curr_label, neigh_labels[n+1], labels);
                            }

                            component_labels[apr_it] = curr_label;
                        }
                    }
                }
            }
        }
    }

    timer.stop_timer();

    std::vector<int> new_labels;
    new_labels.resize(labels.size(), 0);

    timer.start_timer("connected component second loop");

    // iterate over particles
    for(size_t idx = 0; idx < apr_it.total_number_particles(); ++idx) {

        if(component_labels[idx] > 0){

            int curr_label = uf_find(component_labels[idx], labels);
            if(new_labels[curr_label] == 0) {
                new_labels[curr_label] = (++new_labels[0]);
            }

            component_labels[idx] = new_labels[curr_label];
        }
    }

    timer.stop_timer();
}



template<typename T>
void bindFindObjects(py::module& m) {
    m.def("find_objects", &find_objects<T>, "find bounding boxes around each unique label",
          "apr"_a, "labels"_a, "min_coords"_a.noconvert(), "max_coords"_a.noconvert());
}

template<typename T>
void bindFindCenters(py::module& m) {
    m.def("find_label_centers", &find_label_centers<T>, "find volumetric center of each unique label",
          "apr"_a, "labels"_a, "coords"_a.noconvert());
}

template<typename labelType>
void bindFindCentersWeighted(py::module& m) {
    m.def("find_label_centers_weighted", &find_label_centers_weighted<labelType, uint16_t>,
          "find object centers by weighted average of label coordinates",
          "apr"_a, "labels"_a, "coords"_a.noconvert(), "weights"_a);

    m.def("find_label_centers_weighted", &find_label_centers_weighted<labelType, float>,
          "find object centers by weighted average of label coordinates",
          "apr"_a, "labels"_a, "coords"_a.noconvert(), "weights"_a);
}


template<typename T>
void bindFindVolume(py::module& m) {
    m.def("find_label_volume", &find_label_volume<T>, "find volume (in voxels) of each unique label",
          "apr"_a, "object_labels"_a, "volume"_a);
}


template<typename inputType>
void bindConnectedComponent(py::module &m) {
    m.def("connected_component", &calc_connected_component<inputType, uint8_t>, "Compute connected components from a binary particle mask",
          "apr"_a, "binary_mask"_a, "component_labels"_a);

    m.def("connected_component", &calc_connected_component<inputType, uint16_t>, "Compute connected components from a binary particle mask",
          "apr"_a, "binary_mask"_a, "component_labels"_a);

    m.def("connected_component", &calc_connected_component<inputType, uint64_t>, "Compute connected components from a binary particle mask",
          "apr"_a, "binary_mask"_a, "component_labels"_a);
}


void AddMeasure(py::module &m) {

    bindFindObjects<uint8_t>(m);
    bindFindObjects<uint16_t>(m);
    bindFindObjects<uint64_t>(m);

    bindFindCenters<uint8_t>(m);
    bindFindCenters<uint16_t>(m);
    bindFindCenters<uint64_t>(m);

    bindFindCentersWeighted<uint8_t>(m);
    bindFindCentersWeighted<uint16_t>(m);
    bindFindCentersWeighted<uint64_t>(m);

    bindFindVolume<uint8_t>(m);
    bindFindVolume<uint16_t>(m);
    bindFindVolume<uint64_t>(m);

    bindConnectedComponent<uint8_t>(m);
    bindConnectedComponent<uint16_t>(m);
    bindConnectedComponent<uint64_t>(m);
}

#endif //PYLIBAPR_BINDMEASURE_HPP
