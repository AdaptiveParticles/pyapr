//
// Created by Krzysztof Gonciarz on 5/7/18.
// Modified by Joel Jonsson on 2/5/19.
//

#include <ConfigAPR.h>
#include <pybind11/pybind11.h>

#include "converter/src/BindConverter.hpp"
#include "converter/src/BindConverterBatch.hpp"

#include "data_containers/src/BindPixelData.hpp"
#include "data_containers/src/BindAPR.hpp"
#include "data_containers/src/BindParameters.hpp"
#include "data_containers/src/BindParticleData.hpp"
#include "data_containers/src/BindReconPatch.hpp"
#include "data_containers/src/BindLinearIterator.hpp"
#include "data_containers/src/BindLazyAccess.hpp"
#include "data_containers/src/BindLazyData.hpp"
#include "data_containers/src/BindLazyIterator.hpp"

#include "filter/src/BindFilter.hpp"
#include "io/src/BindAPRFile.hpp"
#include "measure/src/BindMeasure.hpp"
#include "morphology/src/BindMorphology.hpp"
#include "reconstruction/src/BindReconstruction.hpp"
#include "restoration/src/BindRichardsonLucy.hpp"
#include "segmentation/src/BindGraphCut.hpp"
#include "transform/src/BindProjection.hpp"
#include "tree/src/BindFillTree.hpp"

#include "viewer/src/BindRaycaster.hpp"
#include "viewer/src/BindViewerHelpers.hpp"


#ifdef PYAPR_USE_CUDA
#define BUILT_WITH_CUDA true
#else
#define BUILT_WITH_CUDA false
#endif

namespace py = pybind11;

// -------- Check if properly configured in CMAKE -----------------------------
#ifndef APR_PYTHON_MODULE_NAME
#error "Name of APR module (python binding) is not defined!"
#endif

// -------- Definition of python module ---------------------------------------
PYBIND11_MODULE(APR_PYTHON_MODULE_NAME, m) {
    m.doc() = "python binding for LibAPR library";
    m.attr("__version__") = py::str(ConfigAPR::APR_VERSION);
    m.attr("__cuda_build__") = BUILT_WITH_CUDA;

    py::module data_containers = m.def_submodule("data_containers");

    AddAPR(data_containers, "APR");
    AddAPRParameters(data_containers);
    AddLinearIterator(data_containers);
    AddReconPatch(data_containers);

    //wrap PyPixelData class for different data types
    AddPyPixelData<uint8_t>(data_containers, "Byte");
    AddPyPixelData<uint16_t>(data_containers, "Short");
    AddPyPixelData<float>(data_containers, "Float");

    // wrap PyParticleData class for different data types
    AddPyParticleData<uint8_t>(data_containers, "Byte");
    AddPyParticleData<float>(data_containers, "Float");
    AddPyParticleData<uint16_t>(data_containers, "Short");
    AddPyParticleData<uint64_t>(data_containers, "Long");

    // wrap lazy classes
    AddLazyAccess(data_containers, "LazyAccess");
    AddLazyIterator(data_containers);
    AddLazyData<uint8_t>(data_containers, "Byte");
    AddLazyData<uint16_t>(data_containers, "Short");
    AddLazyData<uint64_t>(data_containers, "Long");
    AddLazyData<float>(data_containers, "Float");


    py::module converter = m.def_submodule("converter");

    // wrap APRConverter
    AddPyAPRConverter<uint8_t>(converter, "Byte");
    AddPyAPRConverter<uint16_t>(converter, "Short");
    AddPyAPRConverter<float>(converter, "Float");

    // wrap APRConverterBatch (tiled conversion)
    AddPyAPRConverterBatch<uint8_t>(converter, "Byte");
    AddPyAPRConverterBatch<uint16_t>(converter, "Short");
    AddPyAPRConverterBatch<float>(converter, "Float");


    py::module filter = m.def_submodule("filter");
    AddFilter(filter);


    py::module io = m.def_submodule("io");
    AddAPRFile(io, "APRFile");


    py::module measure = m.def_submodule("measure");
    AddMeasure(measure);


    py::module morphology = m.def_submodule("morphology");
    AddMorphology(morphology);


    py::module reconstruction = m.def_submodule("reconstruction");
    AddReconstruction(reconstruction);


    py::module restoration = m.def_submodule("restoration");
    AddRichardsonLucy(restoration);


    py::module segmentation = m.def_submodule("segmentation");
    AddGraphcut(segmentation, "graphcut");


    py::module transform = m.def_submodule("transform");
    AddProjection(transform);


    py::module tree = m.def_submodule("tree");
    AddFillTree(tree);


    py::module viewer = m.def_submodule("viewer");
    AddViewerHelpers(viewer);
    AddRaycaster(viewer, "APRRaycaster");
}
