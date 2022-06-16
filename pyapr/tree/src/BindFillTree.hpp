//
// Created by joeljonsson on 04.03.21.
//

#ifndef PYLIBAPR_BINDFILLTREE_HPP
#define PYLIBAPR_BINDFILLTREE_HPP

#include <pybind11/pybind11.h>
#include "data_structures/APR/APR.hpp"
#include "data_containers/src/BindParticleData.hpp"
#include "numerics/APRTreeNumerics.hpp"

namespace py = pybind11;
using namespace py::literals;


namespace PyAPRTreeNumerics {
    template<typename inputType, typename treeType>
    void fill_tree_mean(APR &apr, const PyParticleData<inputType>& particle_data, PyParticleData<treeType>& tree_data) {
        APRTreeNumerics::fill_tree_mean(apr, particle_data, tree_data);
    }

    template<typename inputType, typename treeType>
    void fill_tree_min(APR &apr, const PyParticleData<inputType>& particle_data, PyParticleData<treeType>& tree_data) {
        APRTreeNumerics::fill_tree_min(apr, particle_data, tree_data);
    }

    template<typename inputType, typename treeType>
    void fill_tree_max(APR &apr, const PyParticleData<inputType>& particle_data, PyParticleData<treeType>& tree_data) {
        APRTreeNumerics::fill_tree_max(apr, particle_data, tree_data);
    }

    template<typename inputType, typename treeType>
    void sample_from_tree(APR& apr,
                          PyParticleData<inputType>& particle_data,
                          PyParticleData<treeType>& tree_data,
                          const int num_levels) {
        APRTreeNumerics::push_down_tree(apr, tree_data, num_levels);
        APRTreeNumerics::push_to_leaves(apr, tree_data, particle_data);
    }
}


template<typename inputType>
void bindFillTreeMean(py::module &m) {
    m.def("fill_tree_mean", &PyAPRTreeNumerics::fill_tree_mean<inputType, inputType>,
          "Compute interior tree particle values by average downsampling",
          "apr"_a, "particle_data"_a, "tree_data"_a);

    if(!std::is_same<inputType, float>::value) {
        m.def("fill_tree_mean", &PyAPRTreeNumerics::fill_tree_mean<inputType, float>,
              "Compute interior tree particle values by average downsampling",
              "apr"_a, "particle_data"_a, "tree_data"_a);
    }
}


template<typename inputType>
void bindFillTreeMin(py::module &m) {
    m.def("fill_tree_min", &PyAPRTreeNumerics::fill_tree_min<inputType, inputType>,
          "Compute interior tree particle values by min downsampling",
          "apr"_a, "particle_data"_a, "tree_data"_a);

    if(!std::is_same<inputType, float>::value) {
        m.def("fill_tree_min", &PyAPRTreeNumerics::fill_tree_min<inputType, float>,
              "Compute interior tree particle values by min downsampling",
              "apr"_a, "particle_data"_a, "tree_data"_a);
    }
}


template<typename inputType>
void bindFillTreeMax(py::module &m) {
    m.def("fill_tree_max", &PyAPRTreeNumerics::fill_tree_max<inputType, inputType>,
          "Compute interior tree particle values by max downsampling",
          "apr"_a, "particle_data"_a, "tree_data"_a);

    if(!std::is_same<inputType, float>::value) {
        m.def("fill_tree_max", &PyAPRTreeNumerics::fill_tree_max<inputType, float>,
              "Compute interior tree particle values by max downsampling",
              "apr"_a, "particle_data"_a, "tree_data"_a);
    }
}


template<typename inputType>
void bindSampleFromTree(py::module &m) {
    m.def("sample_from_tree", &PyAPRTreeNumerics::sample_from_tree<inputType, inputType>,
          "Coarsen particle values by resampling from a certain level of the APR tree",
          "apr"_a, "particle_data"_a, "tree_data"_a, "num_levels"_a);

    if(!std::is_same<inputType, float>::value) {
        m.def("sample_from_tree", &PyAPRTreeNumerics::sample_from_tree<inputType, float>,
              "Coarsen particle values by resampling from a certain level of the APR tree",
              "apr"_a, "particle_data"_a, "tree_data"_a, "num_levels"_a);
    }
}


void AddFillTree(py::module &m) {

    bindFillTreeMean<uint8_t>(m);
    bindFillTreeMean<uint16_t>(m);
    bindFillTreeMean<uint64_t>(m);
    bindFillTreeMean<float>(m);

    bindFillTreeMin<uint8_t>(m);
    bindFillTreeMin<uint16_t>(m);
    bindFillTreeMin<uint64_t>(m);
    bindFillTreeMin<float>(m);

    bindFillTreeMax<uint8_t>(m);
    bindFillTreeMax<uint16_t>(m);
    bindFillTreeMax<uint64_t>(m);
    bindFillTreeMax<float>(m);

    bindSampleFromTree<uint8_t>(m);
    bindSampleFromTree<uint16_t>(m);
    bindSampleFromTree<uint64_t>(m);
    bindSampleFromTree<float>(m);
}

#endif //PYLIBAPR_BINDFILLTREE_HPP
