
#ifndef PYLIBAPR_PYAPRRECONSTRUCTION_HPP
#define PYLIBAPR_PYAPRRECONSTRUCTION_HPP

#include "data_containers/PyPixelData.hpp"
#include "data_containers/PyAPR.hpp"
#include "numerics/APRReconstruction.hpp"
#include "numerics/APRTreeNumerics.hpp"

namespace py = pybind11;


namespace PyAPRReconstruction {

    /**
     * Constant reconstruction (full volume)
     */
    template<typename S, typename T>
    void reconstruct_constant_inplace(APR& apr, PyParticleData<S>& parts, py::array_t<T, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        auto* ptr = static_cast<T*>(buf.ptr);
        PixelData<T> recon;
        recon.init_from_mesh(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2), ptr);

        const size_t size_alloc = std::accumulate(buf.shape.begin(), buf.shape.end(), (size_t) 1, std::multiplies<size_t>());
        if(recon.mesh.size() != size_alloc) {
            throw std::invalid_argument("input array size does not agree with APR dimensions");
        }

        APRReconstruction::reconstruct_constant(apr, recon, parts);
    }

    /**
     * Level reconstruction (full volume)
     */
    template<typename S>
    void reconstruct_level_inplace(APR& apr, py::array_t<S, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        auto* ptr = static_cast<S*>(buf.ptr);
        PixelData<S> recon;
        recon.init_from_mesh(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2), ptr);

        const size_t size_alloc = std::accumulate(buf.shape.begin(), buf.shape.end(), 1, std::multiplies<size_t>());
        if(recon.mesh.size() != size_alloc) {
            throw std::invalid_argument("input array size does not agree with APR dimensions");
        }
        APRReconstruction::reconstruct_level(apr, recon);
    }

    /**
     * Smooth reconstruction (full volume)
     */
    template<typename S, typename T>
    void reconstruct_smooth_inplace(APR& apr, PyParticleData<S>& parts, py::array_t<T, py::array::c_style>& arr) {

        auto buf = arr.request(true);
        auto* ptr = static_cast<T*>(buf.ptr);
        PixelData<T> recon;
        recon.init_from_mesh(apr.org_dims(0), apr.org_dims(1), apr.org_dims(2), ptr);

        const size_t size_alloc = std::accumulate(buf.shape.begin(), buf.shape.end(), 1, std::multiplies<size_t>());
        if(recon.mesh.size() != size_alloc) {
            throw std::invalid_argument("input array size does not agree with APR dimensions");
        }

        APRReconstruction::reconstruct_smooth(apr, recon, parts);
    }

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
    void reconstruct_constant_lazy_inplace(LazyIterator& apr_it, LazyIterator& tree_it, py::array_t<S, py::array::c_style>& arr,
                                           LazyData<T>& parts, LazyData<T>& tree_parts, ReconPatch& patch) {

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
                                        py::array_t<S, py::array::c_style>& arr,
                                        ReconPatch& patch) {

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
    void reconstruct_smooth_lazy_inplace(LazyIterator& apr_it, LazyIterator& tree_it, py::array_t<S, py::array::c_style>& arr,
                                         LazyData<T>& parts, LazyData<T>& tree_parts, ReconPatch& patch) {

        auto buf = arr.request(true);
        PixelData<S> recon;
        initialize_pixeldata_from_buffer(recon, buf);

        if(recon.mesh.size() != patch.size()) {
            throw std::invalid_argument("input array size does not agree with APR dimensions");
        }

        APRReconstruction::reconstruct_smooth_lazy(apr_it, tree_it, recon, parts, tree_parts, patch);
    }

}

void AddPyAPRReconstruction(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());

    using namespace pybind11::literals;

        /// constant reconstruction (full volume) into preallocated numpy array
    m2.def("reconstruct_constant_inplace", &PyAPRReconstruction::reconstruct_constant_inplace<uint16_t, uint16_t>, "Piecewise constant reconstruction",
           "apr"_a, "parts"_a, "arr"_a);
    m2.def("reconstruct_constant_inplace", &PyAPRReconstruction::reconstruct_constant_inplace<uint64_t, uint64_t>, "Piecewise constant reconstruction",
           "apr"_a, "parts"_a, "arr"_a);
    m2.def("reconstruct_constant_inplace", &PyAPRReconstruction::reconstruct_constant_inplace<uint64_t, float>, "Piecewise constant reconstruction",
           "apr"_a, "parts"_a, "arr"_a);
    m2.def("reconstruct_constant_inplace", &PyAPRReconstruction::reconstruct_constant_inplace<uint16_t, float>, "Piecewise constant reconstruction",
           "apr"_a, "parts"_a, "arr"_a);
    m2.def("reconstruct_constant_inplace", &PyAPRReconstruction::reconstruct_constant_inplace<float, float>, "Piecewise constant reconstruction",
           "apr"_a, "parts"_a, "arr"_a);

    /// level reconstruction (full volume) into preallocated numpy array
    m2.def("reconstruct_level_inplace", &PyAPRReconstruction::reconstruct_level_inplace<uint8_t>, "Particle level reconstruction",
           "apr"_a, "arr"_a);
    m2.def("reconstruct_level_inplace", &PyAPRReconstruction::reconstruct_level_inplace<uint16_t>, "Particle level reconstruction",
           "apr"_a, "arr"_a);
    m2.def("reconstruct_level_inplace", &PyAPRReconstruction::reconstruct_level_inplace<float>, "Particle level reconstruction",
           "apr"_a, "arr"_a);

    /// smooth reconstruction (full volume) into preallocated numpy array
    m2.def("reconstruct_smooth_inplace", &PyAPRReconstruction::reconstruct_smooth_inplace<uint16_t, uint16_t>, "Smooth reconstruction",
           "apr"_a, "parts"_a, "arr"_a);
    m2.def("reconstruct_smooth_inplace", &PyAPRReconstruction::reconstruct_smooth_inplace<uint64_t, uint64_t>, "Smooth reconstruction",
           "apr"_a, "parts"_a, "arr"_a);
    m2.def("reconstruct_smooth_inplace", &PyAPRReconstruction::reconstruct_smooth_inplace<uint64_t, float>, "Smooth reconstruction",
           "apr"_a, "parts"_a, "arr"_a);
    m2.def("reconstruct_smooth_inplace", &PyAPRReconstruction::reconstruct_smooth_inplace<uint16_t, float>, "Smooth reconstruction",
           "apr"_a, "parts"_a, "arr"_a);
    m2.def("reconstruct_smooth_inplace", &PyAPRReconstruction::reconstruct_smooth_inplace<float, float>, "Smooth reconstruction",
           "apr"_a, "parts"_a, "arr"_a);

    /// constant reconstruction (patch) into preallocated numpy array
    m2.def("reconstruct_constant_patch_inplace", &PyAPRReconstruction::reconstruct_constant_patch_inplace<uint16_t, uint16_t>,
           "Piecewise constant patch reconstruction",
           "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);
    m2.def("reconstruct_constant_patch_inplace", &PyAPRReconstruction::reconstruct_constant_patch_inplace<uint64_t, uint64_t>,
           "Piecewise constant patch reconstruction",
           "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);
    m2.def("reconstruct_constant_patch_inplace", &PyAPRReconstruction::reconstruct_constant_patch_inplace<uint64_t, float>,
           "Piecewise constant patch reconstruction",
           "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);
    m2.def("reconstruct_constant_patch_inplace", &PyAPRReconstruction::reconstruct_constant_patch_inplace<uint16_t, float>,
           "Piecewise constant patch reconstruction",
           "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);
    m2.def("reconstruct_constant_patch_inplace", &PyAPRReconstruction::reconstruct_constant_patch_inplace<float, float>,
           "Piecewise constant patch reconstruction",
           "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);

    /// level reconstruction (patch) into preallocated numpy array
    m2.def("reconstruct_level_patch_inplace", &PyAPRReconstruction::reconstruct_level_patch_inplace<uint8_t>,
           "Particle level patch reconstruction", "apr"_a, "patch"_a, "arr"_a);
    m2.def("reconstruct_level_patch_inplace", &PyAPRReconstruction::reconstruct_level_patch_inplace<uint16_t>,
           "Particle level patch reconstruction", "apr"_a, "patch"_a, "arr"_a);
    m2.def("reconstruct_level_patch_inplace", &PyAPRReconstruction::reconstruct_level_patch_inplace<float>,
           "Particle level patch reconstruction", "apr"_a, "patch"_a, "arr"_a);

    /// smooth reconstruction (patch) into preallocated numpy array
    m2.def("reconstruct_smooth_patch_inplace", &PyAPRReconstruction::reconstruct_smooth_patch_inplace<uint16_t, uint16_t>,
           "Smooth patch reconstruction",
           "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);
    m2.def("reconstruct_smooth_patch_inplace", &PyAPRReconstruction::reconstruct_smooth_patch_inplace<uint64_t, uint64_t>,
           "Smooth patch reconstruction",
           "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);
    m2.def("reconstruct_smooth_patch_inplace", &PyAPRReconstruction::reconstruct_smooth_patch_inplace<uint64_t, float>,
           "Smooth patch reconstruction",
           "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);
    m2.def("reconstruct_smooth_patch_inplace", &PyAPRReconstruction::reconstruct_smooth_patch_inplace<uint16_t, float>,
           "Smooth patch reconstruction",
           "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);
    m2.def("reconstruct_smooth_patch_inplace", &PyAPRReconstruction::reconstruct_smooth_patch_inplace<float, float>,
           "Smooth patch reconstruction",
           "apr"_a, "parts"_a, "tree_parts"_a, "patch"_a, "arr"_a);

    /// lazy constant reconstruction (patch)
    m2.def("reconstruct_constant_lazy_inplace", &PyAPRReconstruction::reconstruct_constant_lazy_inplace<uint16_t, uint16_t>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "parts"_a, "tree_parts"_a, "patch"_a);
    m2.def("reconstruct_constant_lazy_inplace", &PyAPRReconstruction::reconstruct_constant_lazy_inplace<uint16_t, float>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "parts"_a, "tree_parts"_a, "patch"_a);
    m2.def("reconstruct_constant_lazy_inplace", &PyAPRReconstruction::reconstruct_constant_lazy_inplace<float, float>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "parts"_a, "tree_parts"_a, "patch"_a);
    m2.def("reconstruct_constant_lazy_inplace", &PyAPRReconstruction::reconstruct_constant_lazy_inplace<uint64_t, uint64_t>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "parts"_a, "tree_parts"_a, "patch"_a);
    m2.def("reconstruct_constant_lazy_inplace", &PyAPRReconstruction::reconstruct_constant_lazy_inplace<uint64_t, float>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "parts"_a, "tree_parts"_a, "patch"_a);

    /// lazy level reconstruction (patch)
    m2.def("reconstruct_level_lazy_inplace", &PyAPRReconstruction::reconstruct_level_lazy_inplace<uint8_t>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "patch"_a);
    m2.def("reconstruct_level_lazy_inplace", &PyAPRReconstruction::reconstruct_level_lazy_inplace<uint16_t>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "patch"_a);
    m2.def("reconstruct_level_lazy_inplace", &PyAPRReconstruction::reconstruct_level_lazy_inplace<uint64_t>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "patch"_a);
    m2.def("reconstruct_level_lazy_inplace", &PyAPRReconstruction::reconstruct_level_lazy_inplace<float>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "patch"_a);

    /// lazy smooth reconstruction (patch)
    m2.def("reconstruct_smooth_lazy_inplace", &PyAPRReconstruction::reconstruct_smooth_lazy_inplace<uint16_t, uint16_t>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "parts"_a, "tree_parts"_a, "patch"_a);
    m2.def("reconstruct_smooth_lazy_inplace", &PyAPRReconstruction::reconstruct_smooth_lazy_inplace<uint16_t, float>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "parts"_a, "tree_parts"_a, "patch"_a);
    m2.def("reconstruct_smooth_lazy_inplace", &PyAPRReconstruction::reconstruct_smooth_lazy_inplace<float, float>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "parts"_a, "tree_parts"_a, "patch"_a);
    m2.def("reconstruct_smooth_lazy_inplace", &PyAPRReconstruction::reconstruct_smooth_lazy_inplace<uint64_t, uint64_t>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "parts"_a, "tree_parts"_a, "patch"_a);
    m2.def("reconstruct_smooth_lazy_inplace", &PyAPRReconstruction::reconstruct_smooth_lazy_inplace<uint64_t, float>, "lazy reconstruction",
           "apr_it"_a, "tree_it"_a, "arr"_a, "parts"_a, "tree_parts"_a, "patch"_a);
}


#endif //PYLIBAPR_PYAPRRECONSTRUCTION_HPP
