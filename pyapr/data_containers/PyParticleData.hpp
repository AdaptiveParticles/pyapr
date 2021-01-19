#ifndef PYLIBAPR_PYPARTICLEDATA_HPP
#define PYLIBAPR_PYPARTICLEDATA_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "data_containers/PyAPR.hpp"
#include "data_structures/APR/particles/ParticleData.hpp"
#include "data_containers/iterators/PyLinearIterator.hpp"


namespace py = pybind11;

/**
 * @tparam T data type of particle values
 */
template<typename T>
class PyParticleData {

public:

    ParticleData<T> parts; //#TODO: should this be called something else?

    PyParticleData() {}

    PyParticleData(uint64_t num_particles){ parts.init(num_particles); }

    PyParticleData(PyAPR &aPyAPR) { parts.init(aPyAPR.apr); }

    PyParticleData(ParticleData<T> &aInput) {
        parts.swap(aInput);
    }

    PyParticleData(py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
        auto buf = arr.request();
        this->parts.init(buf.size);
        auto ptr = static_cast<T*>(buf.ptr);
        std::copy(ptr, ptr+size(), this->data());
    }

    bool contains(T val) const {
        for(size_t i = 0; i < this->size(); ++i) {
            if(parts[i] == val) {
                return true;
            }
        }
        return false;
    }

    void resize(uint64_t num_particles) {
        parts.init(num_particles);
    }


    void fill(T val) {
        parts.fill(val);
    }


    T max() {
        T max_val = parts[0];
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) reduction(max : max_val)
#endif
        for(uint64_t i = 1; i < parts.size(); ++i) {
            max_val = std::max(max_val, parts[i]);
        }
        return max_val;
    }

    T min() {
        T min_val = parts[0];
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) reduction(min : min_val)
#endif
        for(uint64_t i = 1; i < parts.size(); ++i) {
            min_val = std::min(min_val, parts[i]);
        }
        return min_val;
    }


    inline T& operator[](size_t aGlobalIndex){ return parts.data[aGlobalIndex]; }
    inline const T& operator[](size_t aGlobalIndex) const { return parts.data[aGlobalIndex]; }


    void copy_float(PyParticleData<float>& partsToCopy) {
        parts.copy(partsToCopy.parts);
    }

    void copy_short(PyParticleData<uint16_t>& partsToCopy) {
        parts.copy(partsToCopy.parts);
    }

    void copy_parts_float(PyAPR &aPyAPR, PyParticleData<float>& partsToCopy, int level=0) {
        parts.copy_parts(aPyAPR.apr, partsToCopy.parts, level);
    }

    void copy_parts_short(PyAPR &aPyAPR, PyParticleData<uint16_t>& partsToCopy, int level=0) {
        parts.copy_parts(aPyAPR.apr, partsToCopy.parts, level);
    }

    /**
     *  return a copy of this
     */
    PyParticleData copy() {
        PyParticleData output(this->size());
        output.parts.copy(this->parts);
        return output;
    }


    /**
     * Check if all elements are equal between two PyParticleData objects
     */
    bool operator==(const PyParticleData<T>& other) const {
        if(this->size() != other.size()) return false;
        for(size_t i=0; i < this->size(); ++i) {
            if(parts[i] != other[i]) return false;
        }
        return true;
    }


    /**
     * Check if not all elements are equal between two PyParticleData objects
     */
    bool operator!=(const PyParticleData<T>& other) const { return !operator==(other); }


    /**
     *  in-place addition with another PyParticleData
     */
    template<typename S>
    PyParticleData& operator+=(const PyParticleData<S>& other) {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot add PyParticleData of different sizes");
        }
        this->parts.binary_map(other.parts, this->parts, [](const T& a, const S& b){return a+b;});
        return *this;
    }


    /**
     *  in-place subtraction of another PyParticleData
     */
    template<typename S>
    PyParticleData& operator-=(const PyParticleData<S>& other) {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot subtract PyParticleData of different sizes");
        }
        this->parts.binary_map(other.parts, this->parts, [](const T& a, const S& b){return a-b;});
        return *this;
    }


    /**
     *   in-place elementwise multiplication with another PyParticleData
     */
    template<typename S>
    PyParticleData& operator*=(const PyParticleData<S> &other) {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot multiply PyParticleData of different sizes");
        }
        this->parts.binary_map(other.parts, this->parts, [](const T& a, const S& b){return a*b;});
        return *this;
    }


    /**
     *  in-place multiplication with a constant
     */
    PyParticleData& operator*=(float v) {
        this->parts.unary_map(this->parts, [v](const T& a){return a*v;});
        return *this;
    }


    /**
     *  in-place addition of a constant
     */
    PyParticleData& operator+=(float v) {
        this->parts.unary_map(this->parts, [v](const T& a){return a+v;});
        return *this;
    }


    /**
     *  in-place subtraction of a constant
     */
    PyParticleData& operator-=(float v) {
        this->parts.unary_map(this->parts, [v](const T& a){return a-v;});
        return *this;
    }


    /**
     *  multiplication with a constant
     */
    PyParticleData operator*(float v) const {
        PyParticleData output(this->size());
        this->parts.unary_map(output.parts, [v](const T& a){return a*v;});
        return output;
    }


    /**
     *  multiplication with a constant
     */
    friend PyParticleData operator*(float v, const PyParticleData& p) {
        PyParticleData output(p.size());
        p.parts.unary_map(output.parts, [v](const T& a){return a*v;});
        return output;
    }


    /**
     *  addition of a constant
     */
    PyParticleData operator+(float v) const {
        PyParticleData output(this->size());
        this->parts.unary_map(output.parts, [v](const T& a){return a+v;});
        return output;
    }


    /**
     *  subtraction of a constant
     */
    PyParticleData operator-(float v) const {
        PyParticleData output(this->size());
        this->parts.unary_map(output.parts, [v](const T& a){return a-v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData operator==(float v) const {
        PyParticleData output(this->size());
        this->parts.unary_map(output.parts, [v](const T& a){return a==v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData operator!=(float v) const {
        PyParticleData output(this->size());
        this->parts.unary_map(output.parts, [v](const T& a){return a!=v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData operator<(float v) const {
        PyParticleData output(this->size());
        this->parts.unary_map(output.parts, [v](const T& a){return a<v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData operator<=(float v) const {
        PyParticleData output(this->size());
        this->parts.unary_map(output.parts, [v](const T& a){return a<=v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData operator>(float v) const {
        PyParticleData output(this->size());
        this->parts.unary_map(output.parts, [v](const T& a){return a>v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData operator>=(float v) const {
        PyParticleData output(this->size());
        this->parts.unary_map(output.parts, [v](const T& a){return a>=v;});
        return output;
    }


    /**
     *  addition of two PyParticleData, where at least one is float -> return float
     */
    template<typename S, std::enable_if_t<(std::is_floating_point<T>::value || std::is_floating_point<S>::value), bool> = true>
    PyParticleData<float> operator+(const PyParticleData<S> &other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot add PyParticleData of different sizes");
        }
        PyParticleData<float> output(this->size());
        this->parts.binary_map(other.parts, output.parts, [](const T& a, const S& b){return a+b;});
        return output;
    }


    /**
     *  addition of two PyParticleData, where neither is float -> return own type
     */
    template<typename S, std::enable_if_t<!std::is_floating_point<T>::value && !std::is_floating_point<S>::value, bool> = false>
    PyParticleData operator+(const PyParticleData<S> &other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot add PyParticleData of different sizes");
        }
        PyParticleData output(this->size());
        this->parts.binary_map(other.parts, output.parts, [](const T& a, const S& b){return a+b;});
        return output;
    }



    /**
     *  subtraction of two PyParticleData, where at least one is float -> return float
     */
    template<typename S, std::enable_if_t<std::is_floating_point<T>::value || std::is_floating_point<S>::value, bool> = true>
    PyParticleData<float> operator-(const PyParticleData<S> &other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot subtract PyParticleData of different sizes");
        }
        PyParticleData<float> output(this->size());
        this->parts.binary_map(other.parts, output.parts, [](const T& a, const S& b){return a-b;});
        return output;
    }

    /**
     *  subtraction of two PyParticleData, where neither is float -> return own type
     */
    template<typename S, std::enable_if_t<!std::is_floating_point<T>::value && !std::is_floating_point<S>::value, bool> = false>
    PyParticleData operator-(const PyParticleData<S> &other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot subtract PyParticleData of different sizes");
        }
        PyParticleData output(this->size());
        this->parts.binary_map(other.parts, output.parts, [](const T& a, const S& b){return a-b;});
        return output;
    }


    /**
     *  elementwise multiplication of two PyParticleData, where at least one is float -> return float
     */
    template<typename S, std::enable_if_t<std::is_floating_point<T>::value || std::is_floating_point<S>::value, bool> = true>
    PyParticleData<float> operator*(const PyParticleData<S> &other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot multiply PyParticleData of different sizes");
        }
        PyParticleData<float> output(this->size());
        this->parts.binary_map(other.parts, output.parts, [](const T& a, const S& b){return a*b;});
        return output;
    }


    /**
     *  elementwise multiplication of two PyParticleData, where at least one is float -> return own type
     */
    template<typename S, std::enable_if_t<!std::is_floating_point<T>::value && !std::is_floating_point<S>::value, bool> = false>
    PyParticleData operator*(const PyParticleData<S> &other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot multiply PyParticleData of different sizes");
        }
        PyParticleData output(this->size());
        this->parts.binary_map(other.parts, output.parts, [](const T& a, const S& b){return a*b;});
        return output;
    }


    /**
     * @return pointer to the data
     */
    T* data() {return parts.data.data();}

    /**
     * @return size of the data (number of particle values)
     */
    uint64_t size() const { return parts.data.size(); }


    //
    //  Particle compression settings
    //

    /**
     * Set the quantization factor for lossy compression step
     *
     * @param q  float compression quantization factor, larger -> more loss
     */
    void set_quantization_factor(float q){
        parts.compressor.set_quantization_factor(q);
    }

    /**
     * Set the background offset and threshold for lossy compression transform (values below this will be ignored
     * and set to the value of bkgrd.
     *
     * @param bkgrd  lower truncation value of the lossy sqrt transform
     */
    void set_background(float bkgrd){
        parts.compressor.set_background(bkgrd);
    }

    /**
     * Turns compression on and off.
     *
     * @param type  int (1 turns lossy compresison on, 0 should be off)
     */
    void set_compression_type(bool type){
        parts.compressor.set_compression_type(type);
    }


    /**
     * Sample the particle values from an image (numpy array) for a given APR (computed from the same image).
     *
     * @param aPyAPR  PyAPR object
     * @param img     numpy array representing the image
     */
    void sample_image(PyAPR &aPyAPR, py::array_t<T> &img) {
        auto buf = img.request(false);
        unsigned int y_num, x_num, z_num;

        std::vector<ssize_t> shape = buf.shape;

        if(buf.ndim == 3) {
            y_num = shape[2];
            x_num = shape[1];
            z_num = shape[0];
        } else if (buf.ndim == 2) {
            y_num = shape[1];
            x_num = shape[0];
            z_num = 1;
        } else if (buf.ndim == 1) {
            y_num = shape[0];
            x_num = 1;
            z_num = 1;
        } else {
            throw std::invalid_argument("input array must be of dimension at most 3");
        }

        auto ptr = static_cast<T*>(buf.ptr);

        PixelData<T> pd;
        pd.init_from_mesh(y_num, x_num, z_num, ptr);

        parts.sample_parts_from_img_downsampled(aPyAPR.apr, pd);
    }


    /**
     * Sample the particle values from a TIFF file, that is read in blocks of z-slices to reduce memory usage.
     *
     * @param aPyAPR     PyAPR object
     * @param aFileName  Path to the TIFF file
     * @param blockSize  Number of z-slices to process in each tile
     * @param ghostSize  Number of ghost slices on each side of the block (maximum slices held in memory at any given time is blockSize + 2*ghostSize)
     */
    void sample_image_blocked(PyAPR &aPyAPR,
                              const std::string &aFileName,
                              const int blockSize,
                              const int ghostSize) {
        parts.sample_parts_from_img_blocked(aPyAPR.apr, aFileName, blockSize, ghostSize);
    }


    /**
     * Fill the particle values with their corresponding resolution levels.
     *
     * @param aPyAPR  a PyAPR object
     */
    void fill_with_levels(PyAPR &aPyAPR) {

        auto it = aPyAPR.apr.iterator();

        for (int level = it.level_min(); level <= it.level_max(); ++level) {

            T lvl = static_cast<T>(level);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
            for (int z = 0; z < it.z_num(level); ++z) {
                for (int x = 0; x < it.x_num(level); ++x) {
                    for (it.begin(level, z, x);it <it.end();it++) {

                        parts[it] = lvl;
                    }
                }
            }
        }
    }
};


template<typename DataType>
void AddPyParticleData(pybind11::module &m, const std::string &aTypeString) {
    using TypeParticles = PyParticleData<DataType>;
    std::string typeStr = aTypeString + "Particles";
    py::class_<TypeParticles>(m, typeStr.c_str(), py::buffer_protocol())
            .def(py::init())
            .def(py::init([](uint64_t num_particles) { return new TypeParticles(num_particles); }))
            .def(py::init([](PyAPR& aPyAPR) { return new TypeParticles(aPyAPR); }))
            .def(py::init([](py::array_t<DataType, py::array::c_style | py::array::forcecast>& arr){ return new TypeParticles(arr); }))
            .def("__len__", [](const TypeParticles &p){ return p.size(); })
            .def("__contains__", [](const TypeParticles &p, DataType v) { return p.contains(v); })
            .def("resize", &TypeParticles::resize, "resize the data array to a specified number of elements")
            .def("fill", &TypeParticles::fill, "fill all elements with a given value")
            .def("min", &TypeParticles::min, "return the minimum value")
            .def("max", &TypeParticles::max, "return the maximum value")
            .def("copy", &TypeParticles::copy_float, "copy particles from another PyParticleData object",
                 py::arg("partsToCopy"))
            .def("copy", &TypeParticles::copy_short, "copy particles from another PyParticleData object",
                 py::arg("partsToCopy"))
            .def("copy", &TypeParticles::copy_parts_float, "copy particles from another PyParticleData object",
                 py::arg("aPyAPR"), py::arg("partsToCopy"), py::arg("level")=0)
            .def("copy", &TypeParticles::copy_parts_short, "copy particles from another PyParticleData object",
                 py::arg("aPyAPR"), py::arg("partsToCopy"), py::arg("level")=0)
            .def("copy", &TypeParticles::copy, "return a copy of self")
            .def("sample_image", &TypeParticles::sample_image, "sample particle values from an image (numpy array)")
            .def("sample_image_blocked", &TypeParticles::sample_image_blocked,
                 "sample particle values from a file in z-blocks to reduce memory usage")
            .def("fill_with_levels", &TypeParticles::fill_with_levels, "fill particle values with levels",
                 py::arg("aPyAPR"))
            .def("set_quantization_factor", &TypeParticles::set_quantization_factor, "set lossy quantization factor")
            .def("set_background", &TypeParticles::set_background, "set lossy background cut off")
            .def("set_compression_type", &TypeParticles::set_compression_type, "turn lossy compression on and off")
            .def(py::self += PyParticleData<uint16_t>())
            .def(py::self += PyParticleData<float>())
            .def(py::self -= PyParticleData<uint16_t>())
            .def(py::self -= PyParticleData<float>())
            .def(py::self *= PyParticleData<uint16_t>())
            .def(py::self *= PyParticleData<float>())
            .def(py::self *= float())
            .def(py::self += float())
            .def(py::self -= float())
            .def(py::self + PyParticleData<uint16_t>())
            .def(py::self + PyParticleData<float>())
            .def(py::self - PyParticleData<uint16_t>())
            .def(py::self - PyParticleData<float>())
            .def(py::self * PyParticleData<uint16_t>())
            .def(py::self * PyParticleData<float>())
            .def(py::self * float())
            .def(py::self + float())
            .def(py::self - float())
            .def(float() * py::self)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::self == float())
            .def(py::self != float())
            .def(py::self < float())
            .def(py::self <= float())
            .def(py::self > float())
            .def(py::self >= float())
            // access single element
            .def("__getitem__", [](TypeParticles &s, size_t i) {
                if (i >= s.size()) { throw py::index_error(); }
                return s[i];
            })
            .def("__setitem__", [](TypeParticles &s, size_t i, DataType v) {
                if (i >= s.size()) { throw py::index_error(); }
                s[i] = v;
            })
            /// Slicing protocol (via NumPy array)
            .def("__getitem__", [](const TypeParticles &p, py::slice slice) -> py::array_t<DataType> {
                size_t start, stop, step, slicelength;
                if (!slice.compute(p.size(), &start, &stop, &step, &slicelength))
                    throw py::error_already_set();

                auto result = py::array_t<DataType>(slicelength);
                auto buf = result.request();
                auto ptr = static_cast<DataType *>(buf.ptr);

                for (size_t i = 0; i < slicelength; ++i) {
                    ptr[i] = p[start];
                    start += step;
                }
                return result;
            })
            .def("__setitem__", [](TypeParticles &p, py::slice slice, const py::array_t<DataType> &value) {
                size_t start, stop, step, slicelength;
                if (!slice.compute(p.size(), &start, &stop, &step, &slicelength))
                    throw py::error_already_set();

                auto buf = value.request();
                if (slicelength != (size_t)buf.size)
                    throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");

                auto ptr = static_cast<DataType *>(buf.ptr);
                for (size_t i = 0; i < slicelength; ++i) {
                    p[start] = ptr[i];
                    start += step;
                }
            })

            /// set slice to single value
            .def("__setitem__", [](TypeParticles &p, py::slice slice, const DataType value) {
                size_t start, stop, step, slicelength;
                if (!slice.compute(p.size(), &start, &stop, &step, &slicelength))
                    throw py::error_already_set();

                for (size_t i = 0; i < slicelength; ++i) {
                    p[start] = value;
                    start += step;
                }
            })

            .def("__iter__", [](const TypeParticles &s) { return py::make_iterator(s.parts.data.begin(), s.parts.data.end()); },
                         py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */)
            .def_buffer([](TypeParticles &p) -> py::buffer_info{
                return py::buffer_info(
                        p.data(),
                        sizeof(DataType),
                        py::format_descriptor<DataType>::format(),
                        1,
                        {p.size()},
                        {sizeof(DataType)}
                );
            });
}

#endif //PYLIBAPR_PYPARTICLEDATA_HPP