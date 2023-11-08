
#ifndef PYLIBAPR_PYAPRRECONSTRUCTION_HPP
#define PYLIBAPR_PYAPRRECONSTRUCTION_HPP

#include "data_structures/APR/APR.hpp"
#include "data_containers/src/BindParticleData.hpp"
#include "numerics/APRReconstruction.hpp"
#include "numerics/APRTreeNumerics.hpp"

namespace py = pybind11;
using namespace pybind11::literals;


namespace PyAPRReconstruction {

    template<typename T>
    void initialize_pixeldata_from_buffer(PixelData<T>& pd, py::buffer_info& buf) {
        auto* ptr = static_cast<T*>(buf.ptr);

        if(buf.ndim == 3) {
            pd.init_from_mesh(buf.shape[2], buf.shape[1], buf.shape[0], ptr);
        } else if(buf.ndim == 2) {
            pd.init_from_mesh(buf.shape[1], buf.shape[0], 1, ptr);
        } else if(buf.ndim == 1) {
            pd.init_from_mesh(buf.shape[0], 1, 1, ptr);
        } else {
            throw std::invalid_argument("input array must be 1-3 dimensional");
        }
    }

    /**
     * Constant reconstruction (full volume)
     */
    template<typename S, typename T>
    void reconstruct_constant_inplace(APR& apr, PyParticleData<S>& parts, py::array_t<T, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        PixelData<T> recon;
        initialize_pixeldata_from_buffer(recon, buf);

        APRReconstruction::reconstruct_constant(apr, recon, parts);
    }

    /**
     * Level reconstruction (full volume)
     */
    template<typename S>
    void reconstruct_level_inplace(APR& apr, py::array_t<S, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        PixelData<S> recon;
        initialize_pixeldata_from_buffer(recon, buf);

        APRReconstruction::reconstruct_level(apr, recon);
    }

    /**
     * Smooth reconstruction (full volume)
     */
    template<typename S, typename T>
    void reconstruct_smooth_inplace(APR& apr, PyParticleData<S>& parts, py::array_t<T, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        PixelData<T> recon;
        initialize_pixeldata_from_buffer(recon, buf);

        APRReconstruction::reconstruct_smooth(apr, recon, parts);
    }

    /**
     * Constant reconstruction (patch)
     */
    template<typename S, typename T>
    void
    reconstruct_constant_patch_inplace(APR& apr, PyParticleData<S>& parts, PyParticleData<T>& tree_parts,
                                       ReconPatch& patch, py::array_t<S, py::array::c_style>& arr) {
        auto buf = arr.request(true);
        PixelData<S> recon;
        initialize_pixeldata_from_buffer(recon, buf);

        if(recon.mesh.size() != patch.size()) {
            throw std::invalid_argument("input array must agree with patch size!");
        }

        APRReconstruction::reconstruct_constant(apr, recon, parts, tree_parts, patch);
    }


    /**
     * Level reconstruction (patch)
     */
    template<typename S>
    void
    reconstruct_level_patch_inplace(APR& apr, ReconPatch& patch, py::array_t<S, py::array::c_style>& arr) {
        auto buf = arr.request(true);
        PixelData<S> recon;
        initialize_pixeldata_from_buffer(recon, buf);

        if(recon.mesh.size() != patch.size()) {
            throw std::invalid_argument("input array must agree with patch size!");
        }

        APRReconstruction::reconstruct_level(apr, recon, patch);
    }


    /**
     * smooth reconstruction (patch)
     */
    template<typename S, typename T>
    void
    reconstruct_smooth_patch_inplace(APR& apr, PyParticleData<S>& parts, PyParticleData<T>& tree_parts,
                                     ReconPatch& patch, py::array_t<S, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        PixelData<S> recon;
        initialize_pixeldata_from_buffer(recon, buf);

        if(recon.mesh.size() != patch.size()) {
            throw std::invalid_argument("input array must agree with patch size!");
        }

        APRReconstruction::reconstruct_smooth(apr, recon, parts, tree_parts, patch);
    }



    /**
     * Lazy patch reconstruction (constant)
     */
    template<typename S, typename T>
    void reconstruct_constant_lazy_inplace(LazyIterator& apr_it, LazyIterator& tree_it,
                                           LazyData<S>& parts, LazyData<T>& tree_parts,
                                           ReconPatch& patch, py::array_t<S, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        PixelData<S> recon;
        initialize_pixeldata_from_buffer(recon, buf);

        if(recon.mesh.size() != patch.size()) {
            throw std::invalid_argument("input array must agree with patch size!");
        }

        APRReconstruction::reconstruct_constant_lazy(apr_it, tree_it, recon, parts, tree_parts, patch);
    }

    /**
     * Lazy patch reconstruction (level)
     */
    template<typename S>
    void reconstruct_level_lazy_inplace(LazyIterator& apr_it,
                                        LazyIterator& tree_it,
                                        ReconPatch& patch,
                                        py::array_t<S, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        PixelData<S> recon;
        initialize_pixeldata_from_buffer(recon, buf);

        if(recon.mesh.size() != patch.size()) {
            throw std::invalid_argument("input array must agree with patch size!");
        }

        APRReconstruction::reconstruct_level_lazy(apr_it, tree_it, recon, patch);
    }

    /**
     * Lazy patch reconstruction (smooth)
     */
    template<typename S, typename T>
    void reconstruct_smooth_lazy_inplace(LazyIterator& apr_it, LazyIterator& tree_it,
                                         LazyData<S>& parts, LazyData<T>& tree_parts,
                                         ReconPatch& patch, py::array_t<S, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        PixelData<S> recon;
        initialize_pixeldata_from_buffer(recon, buf);

        if(recon.mesh.size() != patch.size()) {
            throw std::invalid_argument("input array size does not agree with APR dimensions");
        }

        APRReconstruction::reconstruct_smooth_lazy(apr_it, tree_it, recon, parts, tree_parts, patch);
    }

}


/**
 * Bind full volume reconstruction methods
 */
template<typename partsType>
void bindReconstruct(py::module &m) {
    /// constant
    m.def("reconstruct_constant_inplace", &PyAPRReconstruction::reconstruct_constant_inplace<partsType, partsType>,
          "Piecewise constant reconstruction", "apr"_a, "parts"_a, "arr"_a);

    /// level
    m.def("reconstruct_level_inplace", &PyAPRReconstruction::reconstruct_level_inplace<partsType>,
          "Particle level reconstruction", "apr"_a, "arr"_a);

    /// smooth
    m.def("reconstruct_smooth_inplace", &PyAPRReconstruction::reconstruct_smooth_inplace<partsType, partsType>,
          "Smooth reconstruction", "apr"_a, "parts"_a, "arr"_a);
}


/**
 * Bind partial volume reconstruction methods
 */
template<typename partsType>
void bindReconstructPatch(py::module &m) {
    /// constant
    m.def("reconstruct_constant_patch_inplace", &PyAPRReconstruction::reconstruct_constant_patch_inplace<partsType, partsType>,
          "Piecewise constant patch reconstruction", "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);

    /// level
    m.def("reconstruct_level_patch_inplace", &PyAPRReconstruction::reconstruct_level_patch_inplace<partsType>,
          "Particle level patch reconstruction", "apr"_a, "patch"_a, "arr"_a);

    /// smooth
    m.def("reconstruct_smooth_patch_inplace", &PyAPRReconstruction::reconstruct_smooth_patch_inplace<partsType, partsType>,
          "Smooth patch reconstruction", "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);

    /// constant and smooth reconstruction with tree_parts of float type
    if(!std::is_same<partsType, float>::value) {
        m.def("reconstruct_constant_patch_inplace",
              &PyAPRReconstruction::reconstruct_constant_patch_inplace<partsType, float>,
              "Piecewise constant patch reconstruction", "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);

        m.def("reconstruct_smooth_patch_inplace",
              &PyAPRReconstruction::reconstruct_smooth_patch_inplace<partsType, float>,
              "Smooth patch reconstruction", "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);
    }
}


/**
 * Bind lazy reconstruction methods
 */
template<typename partsType>
void bindReconstructLazy(py::module &m) {
    /// constant
    m.def("reconstruct_constant_lazy_inplace", &PyAPRReconstruction::reconstruct_constant_lazy_inplace<partsType, partsType>,
          "Piecewise constant lazy patch reconstruction",
          "apr_it"_a, "tree_it"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);

    /// level
    m.def("reconstruct_level_lazy_inplace", &PyAPRReconstruction::reconstruct_level_lazy_inplace<partsType>,
          "Particle level lazy patch reconstruction", "apr_it"_a, "tree_it"_a, "patch"_a, "arr"_a);

    /// smooth
    m.def("reconstruct_smooth_lazy_inplace", &PyAPRReconstruction::reconstruct_smooth_lazy_inplace<partsType, partsType>,
          "Smooth lazy patch reconstruction",
          "apr_it"_a, "tree_it"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);

    /// constant and smooth reconstruction with tree_parts of float type
    if(!std::is_same<partsType, float>::value) {
        m.def("reconstruct_constant_lazy_inplace",
              &PyAPRReconstruction::reconstruct_constant_lazy_inplace<partsType, float>,
              "Piecewise constant lazy patch reconstruction",
              "apr_it"_a, "tree_it"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);

        m.def("reconstruct_smooth_lazy_inplace",
              &PyAPRReconstruction::reconstruct_smooth_lazy_inplace<partsType, float>,
              "Smooth lazy patch reconstruction",
              "apr_it"_a, "tree_it"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);
    }
}


void AddReconstruction(py::module &m) {

    bindReconstruct<uint8_t>(m);
    bindReconstruct<uint16_t>(m);
    bindReconstruct<uint64_t>(m);
    bindReconstruct<float>(m);
    bindReconstruct<int32_t>(m);

    bindReconstructPatch<uint8_t>(m);
    bindReconstructPatch<uint16_t>(m);
    bindReconstructPatch<uint64_t>(m);
    bindReconstructPatch<float>(m);
    bindReconstructPatch<int32_t>(m);

    bindReconstructLazy<uint8_t>(m);
    bindReconstructLazy<uint16_t>(m);
    bindReconstructLazy<uint64_t>(m);
    bindReconstructLazy<float>(m);
}


#endif //PYLIBAPR_PYAPRRECONSTRUCTION_HPP
