//
// Created by Joel Jonsson on 29.06.18.
//

#ifndef PYLIBAPR_PYAPR_HPP
#define PYLIBAPR_PYAPR_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

//#include "ConfigAPR.h"
#include "data_structures/APR/APR.hpp"

#include "PyPixelData.hpp"
#include "PyAPRFiltering.hpp"

namespace py = pybind11;

// -------- Utility classes to be wrapped in python ----------------------------
template <typename T>
class PyAPR {
    template<typename> friend class PyPixelData;

public:

    APR <T> apr;

    PyAPR () {}

    /**
     * Reads in the given HDF5 APR file.
     *
     * @param aAprFileName
     */
    void read_apr(const std::string &aAprFileName) {
        apr.read_apr(aAprFileName);
    }

    // TODO: add more versions of write_apr, with compression options etc?
    /**
     * Writes the APR to a HDF5 file without(?) compression.
     *
     * @param aOutputFile
     */
    void write_apr(const std::string &aOutputFile) {
        apr.write_apr("", aOutputFile);
    }

    /**
     * Returns the piecewise constant reconstruction from the APR instance as a PyPixelData object. This can be cast
     * into a numpy array without copy using 'arr = numpy.array(obj, copy=False)'.
     *
     * @return PyPixelData holding the reconstructed image
     */
    PyPixelData<T> pc_recon() {

        PixelData<T> reconstructedImage;

        APRReconstruction().interp_img(apr, reconstructedImage, apr.particles_intensities);

        /*
        // this creates a copy...
        return py::array_t<T>({reconstructedImage.x_num, reconstructedImage.y_num, reconstructedImage.z_num},
                         {sizeof(T) * reconstructedImage.y_num * reconstructedImage.x_num, sizeof(T), sizeof(T) * reconstructedImage.y_num},
                         reconstructedImage.mesh.get());
        */

        //this does not copy, and can be cast to numpy.array on python side without copy (set copy=False)
        return PyPixelData<T>(reconstructedImage);
    }

    /**
     * Returns the smooth reconstruction from the APR instance as a PyPixelData object. This can be cast into a numpy
     * array without copy using 'arr = numpy.array(obj, copy=False)'.
     *
     * @return PyPixelData holding the reconstructed image
     */
    PyPixelData<T> smooth_recon() {

        PixelData<T> reconstructedImage;

        APRReconstruction().interp_parts_smooth(apr, reconstructedImage, apr.particles_intensities);

        return PyPixelData<T>(reconstructedImage);
    }


    /**
     * Sets the parameters for the APR conversion.
     *
     * @param par pyApr.APRParameters object
     */
    void set_parameters(const py::object &par) {

        if( py::isinstance<APRParameters>(par) ) {
            apr.parameters = par.cast<APRParameters>();
        } else {
            throw std::invalid_argument("Input has to be a pyApr.APRParameters object.");
        }
    }


    /**
     * Computes the APR from the input python array.
     *
     * @param input image as python (numpy) array
     */
    void get_apr_from_array(py::array &input) {

        auto buf = input.request();


        // Some checks, may need some polishing
        if( buf.ptr == nullptr ) {
            std::cerr << "Could not pass buffer in call to apr_from_array" << std::endl;
        }

        if ( !input.writeable() ) {
            std::cerr << "Input array must be writeable" << std::endl;
        }

        if( !py::isinstance<py::array_t<T>>(input) ) {
            throw std::invalid_argument("Conflicting types. Make sure the input array is of the same type as the AprType instance.");
        }

        auto *ptr = static_cast<T*>(buf.ptr);

        PixelData<T> input_img;

        //TODO: fix memory/ownership passing or just revert to copying?
        input_img.init_from_mesh(buf.shape[1], buf.shape[0], buf.shape[2], ptr); // may lead to memory issues

        apr.get_apr(input_img);
    }


    /**
     * Reads in the provided tiff file and computes its APR. Note: parameters for the APR conversion should be set
     * before by using set_parameters.
     *
     * @param aInputFile path to the tiff image file
     */
    void get_apr_from_file(const std::string &aInputFile) {
        const TiffUtils::TiffInfo aTiffFile(aInputFile);

        apr.parameters.input_dir = "";
        apr.parameters.input_image_name = aInputFile;
        apr.get_apr();
    }


    uint64_t total_num_particles() {
        return apr.total_number_particles();
    }


    /**
     * //TODO: think of copy-free options
     *
     * @return particle intensities as numpy array
     */
    py::array get_intensities() {

        auto v = new std::vector<T>(apr.particles_intensities.data);

        //auto v = &apr.particles_intensities.data;

        auto capsule = py::capsule(v, [](void *v) { delete reinterpret_cast<std::vector<float>*>(v); });
        return py::array(v->size(), v->data(), capsule);


    }

    py::array get_levels() {
        auto levels = new std::vector<float>;
        levels->resize(apr.total_number_particles());

        auto apr_iterator = apr.iterator();

        for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
            int z = 0;
            int x = 0;

#ifdef PYAPR_HAVE_OPENMP
            const bool parallel_z = apr_iterator.spatial_index_z_max(level) > 1;
            const bool parallel_x = !parallel_z && apr_iterator.spatial_index_x_max(level) > 1;
#endif

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator) if(parallel_z)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator) if(parallel_x)
#endif
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        levels->data()[apr_iterator] = (float)apr_iterator.level();

                    }
                }
            }
        }

        auto capsule = py::capsule(levels, [](void *levels) { delete reinterpret_cast<std::vector<float>*>(levels); });
        return py::array(levels->size(), levels->data(), capsule);

    }



    uint64_t compute_particles_after_maxpool(int level_delta) {

        PyAPRFiltering filter_fns;

        /// Find the current maximum level using the shape of the input
        unsigned int current_max_level = std::max(apr.level_max()-level_delta, apr.level_min());
        apr.apr_tree.init(apr);
        /// Find the number of particles of the output, which is the number of particles up to current_max_level-1 plus the
        /// number of new particles at current_max_level-1
        uint64_t number_parts_out = filter_fns.number_parts_at_level(apr, current_max_level-1);

        return number_parts_out;
    }


    void sample_multichannel_image(py::array &image) {

        py::buffer_info image_buf = image.request();

        auto image_ptr = (float *) image_buf.ptr;

        /// assuming input array is x-contiguous of shape (channels, height, width)
        size_t number_channels = image_buf.shape[0];
        size_t x_num = image_buf.shape[1];
        size_t y_num = image_buf.shape[2];

        /// expand the particle data vector
        apr.particles_intensities.data.resize(number_channels * apr.total_number_particles());

        for(size_t channel = 0; channel < number_channels; ++channel) {
            PixelData<float> image_temp(y_num, x_num, 1);

            size_t offset = channel * x_num * y_num;

            /// copy the current channel to a PixelData object. note that it is read in y -> x

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
            for(size_t idx = 0; idx < x_num * y_num; ++idx) {
                image_temp.mesh[idx] = image_ptr[idx + offset];
            }

            std::vector<PixelData<float>> img_by_level;
            downsamplePyramid(image_temp, img_by_level, apr.level_max(), apr.level_min());

            auto apr_iterator = apr.iterator();

            uint64_t parts_offset = channel * apr.total_number_particles();

            for (unsigned int level = apr_iterator.level_min(); level <= apr_iterator.level_max(); ++level) {
                int z = 0;
                int x = 0;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(x) firstprivate(apr_iterator)
#endif
                //for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        apr.particles_intensities.data[apr_iterator.global_index() + parts_offset] =
                                img_by_level[level].at(apr_iterator.y(), x, 0);

                    }
                }
                //}
            }
        }
    }


    /**
     * compute a piecewise constant reconstruction using the provided particle intensities and return the image as a
     * PyPixelData object
     *
     * @param intensities   (numpy) array
     * @return              PyPixelData reconstruction (can be cast to numpy in python w/o copy)
     */
    PyPixelData<T> recon_newints(py::array &intensities, int level_delta) {

        PyAPRFiltering filter_fns;

        PixelData<T> recon;
        filter_fns.interp_img_new_intensities(apr, recon, intensities, level_delta);

        //TiffUtils::saveMeshAsTiff("/Users/joeljonsson/Documents/STUFF/recon_new_intensities.tif", recon);

        return PyPixelData<T>(recon);

    }

    template<typename S, typename R, typename C>
    void downsample(const PixelData<float> &aInput, PixelData<S> &aOutput, R reduce, C constant_operator, bool aInitializeOutput = false) {
        const size_t z_num = aInput.z_num;
        const size_t x_num = aInput.x_num;
        const size_t y_num = aInput.y_num;

        // downsampled dimensions twice smaller (rounded up)
        const size_t z_num_ds = ceil(z_num/2.0);
        const size_t x_num_ds = ceil(x_num/2.0);
        const size_t y_num_ds = ceil(y_num/2.0);

        APRTimer timer;
        timer.verbose_flag = false;

        if (aInitializeOutput) {
            timer.start_timer("downsample_initalize");
            aOutput.init(y_num_ds, x_num_ds, z_num_ds);
            timer.stop_timer();
        }

        timer.start_timer("downsample_loop");
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (size_t z = 0; z < z_num_ds; ++z) {
            for (size_t x = 0; x < x_num_ds; ++x) {

                // shifted +1 in original inMesh space
                const int64_t shx = std::min(2*x + 1, x_num - 1);
                const int64_t shz = std::min(2*z + 1, z_num - 1);

                const ArrayWrapper<float> &inMesh = aInput.mesh;
                ArrayWrapper<S> &outMesh = aOutput.mesh;

                for (size_t y = 0; y < y_num_ds; ++y) {
                    const int64_t shy = std::min(2*y + 1, y_num - 1);
                    const int64_t idx = z * x_num_ds * y_num_ds + x * y_num_ds + y;
                    outMesh[idx] =  constant_operator(
                            reduce(reduce(reduce(reduce(reduce(reduce(reduce(        // inMesh coordinates
                                    inMesh[2*z * x_num * y_num + 2*x * y_num + 2*y],  // z,   x,   y
                                    inMesh[2*z * x_num * y_num + 2*x * y_num + shy]), // z,   x,   y+1
                                                                      inMesh[2*z * x_num * y_num + shx * y_num + 2*y]), // z,   x+1, y
                                                               inMesh[2*z * x_num * y_num + shx * y_num + shy]), // z,   x+1, y+1
                                                        inMesh[shz * x_num * y_num + 2*x * y_num + 2*y]), // z+1, x,   y
                                                 inMesh[shz * x_num * y_num + 2*x * y_num + shy]), // z+1, x,   y+1
                                          inMesh[shz * x_num * y_num + shx * y_num + 2*y]), // z+1, x+1, y
                                   inMesh[shz * x_num * y_num + shx * y_num + shy])  // z+1, x+1, y+1
                    );
                }
            }
        }
        timer.stop_timer();
    }

    void downsamplePyramid(PixelData<float> &original_image, std::vector<PixelData<float>> &downsampled, size_t l_max, size_t l_min) {
        downsampled.resize(l_max + 1); // each level is kept at same index
        downsampled.back().swap(original_image); // put original image at l_max index

        // calculate downsampled in range (l_max, l_min]
        auto sum = [](const float x, const float y) -> float { return x + y; };
        auto divide_by_8 = [](const float x) -> float { return x/8.0; };
        for (size_t level = l_max; level > l_min; --level) {
            downsample(downsampled[level], downsampled[level - 1], sum, divide_by_8, true);
        }
    }


    void get_mask(py::array &mask, int level_delta, int bn) {

        auto mask_buf = mask.request();
        auto mask_ptr = (float *) mask_buf.ptr;

        const uint64_t offset = bn * mask_buf.shape[1] * mask_buf.shape[2] * mask_buf.shape[3];

        auto apr_iterator = apr.iterator();

        unsigned int z = 0;
        unsigned int x;

        int level = std::max(apr.level_max()-level_delta, apr.level_min());

        for(x = 0; x < apr.spatial_index_x_max(level); ++x) {
            for(apr_iterator.set_new_lzx(level, z, x);
                apr_iterator.global_index() < apr_iterator.end_index;
                apr_iterator.set_iterator_to_particle_next_particle()) {

                mask_ptr[x * apr.spatial_index_y_max(level) + apr_iterator.y() + offset] = 1.0f;
            }
        }
    }


    uint64_t min_level() {
        return apr.level_min();
    }

    uint64_t max_level() {
        return apr.level_max();
    }


};

// -------- Templated wrapper -------------------------------------------------
template <typename DataType>
void AddPyAPR(pybind11::module &m, const std::string &aTypeString) {
    using AprType = PyAPR<DataType>;
    std::string typeStr = "Apr" + aTypeString;
    py::class_<AprType>(m, typeStr.c_str())
            .def(py::init())
            .def("read_apr", &AprType::read_apr, "Method to read HDF5 APR files")
            .def("write_apr", &AprType::write_apr, "Writes the APR instance to a HDF5 file")
            .def("reconstruct", &AprType::pc_recon, py::return_value_policy::move, "returns the piecewise constant image reconstruction as a python array")
            .def("reconstruct_smooth", &AprType::smooth_recon, py::return_value_policy::move, "returns a smooth image reconstruction as a python array")
            .def("set_parameters", &AprType::set_parameters, "Set parameters for APR conversion")
            .def("get_apr_from_array", &AprType::get_apr_from_array, "Construct APR from input array (no copy)")
            .def("get_apr_from_file", &AprType::get_apr_from_file, "Construct APR from input .tif image")
            .def("get_intensities", &AprType::get_intensities, "return the particle intensities as a python array")
            .def("get_levels", &AprType::get_levels, "return the particle levels as a python array")
            .def("recon", &AprType::recon_newints, "recon with given intensities")
            .def("nparticles", &AprType::total_num_particles, "return number of particles")
            .def("number_particles_after_maxpool", &AprType::compute_particles_after_maxpool, "computes the number of particles in the output of max_pool")
            .def("sample_channels", &AprType::sample_multichannel_image, "samples each channel of the input image separately")
            .def("get_mask", &AprType::get_mask, "compute mask for a given level (1 where particles exist, 0 elsewhere)")
            .def("min_level", &AprType::min_level, "return the minimum resolution level")
            .def("max_level", &AprType::max_level, "return the maximum resolution level");
}

#endif //PYLIBAPR_PYAPR_HPP
