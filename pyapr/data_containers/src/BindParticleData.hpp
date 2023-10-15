#ifndef PYLIBAPR_PYPARTICLEDATA_HPP
#define PYLIBAPR_PYPARTICLEDATA_HPP

#include <iostream>
#include <string>
#include <typeinfo>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <data_structures/APR/particles/ParticleData.hpp>

template<typename T>
struct TypeParseTraits{
    static const char* name() { return typeid(T).name(); }
};

#define REGISTER_PARSE_TYPE(X) template <> struct TypeParseTraits<X> { static const char* name() { return #X; }; }

REGISTER_PARSE_TYPE(uint8);
REGISTER_PARSE_TYPE(uint16);
REGISTER_PARSE_TYPE(int32);
REGISTER_PARSE_TYPE(uint64);
REGISTER_PARSE_TYPE(float);

namespace py = pybind11;

/**
 * @tparam T data type of particle values
 */
template<typename T>
class PyParticleData : public ParticleData<T> {

public:

    PyParticleData() : ParticleData<T>() {}
    PyParticleData(uint64_t num_particles) : ParticleData<T>(num_particles) {};
    PyParticleData(APR &apr) : ParticleData<T>(apr.total_number_particles()) {};

    PyParticleData(py::array_t<T, py::array::c_style | py::array::forcecast>& arr) {
        auto buf = arr.request();
        this->init(buf.size);
        auto ptr = static_cast<T*>(buf.ptr);
        std::copy(ptr, ptr+buf.size, this->begin());
    }

    template<typename S>
    PyParticleData(PyParticleData<S>& other) {
        this->copy(other);
    }

    bool contains(T val) const {
        for(size_t i = 0; i < this->size(); ++i) {
            if(this->data[i] == val) {
                return true;
            }
        }
        return false;
    }

    void resize(uint64_t num_particles) {
        this->init(num_particles);
    }

    T max() const {
        T max_val = this->data[0];
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) reduction(max : max_val)
#endif
        for(uint64_t i = 1; i < this->size(); ++i) {
            max_val = std::max(max_val, this->data[i]);
        }
        return max_val;
    }

    T min() const {
        T min_val = this->data[0];
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) reduction(min : min_val)
#endif
        for(uint64_t i = 1; i < this->size(); ++i) {
            min_val = std::min(min_val, this->data[i]);
        }
        return min_val;
    }


    /**
     *  return a copy of this
     */
    PyParticleData ret_copy() const {
        PyParticleData output(this->size());
        output.copy(*this);
        return output;
    }


    /**
     * Check if all elements are equal between two PyParticleData objects
     */
    bool operator==(const PyParticleData<T>& other) const {
        if(this->size() != other.size()) return false;
        for(size_t i=0; i < this->size(); ++i) {
            if(this->data[i] != other[i]) return false;
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
        this->binary_map(other, *this, [](const T a, const S b){return a+b;});
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
        this->binary_map(other, *this, [](const T a, const S b){return a-b;});
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
        this->binary_map(other, *this, [](const T a, const S b){return a*b;});
        return *this;
    }


    /**
     *  in-place multiplication with a constant
     */
    PyParticleData& operator*=(float v) {
        this->unary_map(*this, [v](const T a){return a*v;});
        return *this;
    }


    /**
     *  in-place addition of a constant
     */
    PyParticleData& operator+=(float v) {
        this->unary_map(*this, [v](const T a){return a+v;});
        return *this;
    }


    /**
     *  in-place subtraction of a constant
     */
    PyParticleData& operator-=(float v) {
        this->unary_map(*this, [v](const T a){return a-v;});
        return *this;
    }


    /**
     *  multiplication with a constant
     */
    PyParticleData operator*(float v) const {
        PyParticleData output(this->size());
        this->unary_map(output, [v](const T a){return a*v;});
        return output;
    }


    /**
     *  multiplication with a constant
     */
    friend PyParticleData operator*(float v, const PyParticleData& p) {
        PyParticleData output(p.size());
        p.unary_map(output, [v](const T a){return a*v;});
        return output;
    }


    /**
     *  addition of a constant
     */
    PyParticleData operator+(float v) const {
        PyParticleData output(this->size());
        this->unary_map(output, [v](const T a){return a+v;});
        return output;
    }


    /**
     *  subtraction of a constant
     */
    PyParticleData operator-(float v) const {
        PyParticleData output(this->size());
        this->unary_map(output, [v](const T a){return a-v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData<uint8_t> operator==(float v) const {
        PyParticleData<uint8_t> output(this->size());
        this->unary_map(output, [v](const T a){return a==v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData<uint8_t> operator!=(float v) const {
        PyParticleData<uint8_t> output(this->size());
        this->unary_map(output, [v](const T a){return a!=v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData<uint8_t> operator<(float v) const {
        PyParticleData<uint8_t> output(this->size());
        this->unary_map(output, [v](const T a){return a<v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData<uint8_t> operator<=(float v) const {
        PyParticleData<uint8_t> output(this->size());
        this->unary_map(output, [v](const T a){return a<=v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData<uint8_t> operator>(float v) const {
        PyParticleData<uint8_t> output(this->size());
        this->unary_map(output, [v](const T a){return a>v;});
        return output;
    }


    /**
     * compare each value to a constant.
     * @returns a new PyParticleData object with 1's and 0's indicating where the condition is true or false
     */
    PyParticleData<uint8_t> operator>=(float v) const {
        PyParticleData<uint8_t> output(this->size());
        this->unary_map(output, [v](const T a){return a>=v;});
        return output;
    }


    /**
     *  addition of two PyParticleData, where at least one is float -> return float
     */
    template<typename S, std::enable_if_t<(std::is_floating_point<T>::value || std::is_floating_point<S>::value), bool> = true>
    PyParticleData<float> operator+(const PyParticleData<S>& other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot add PyParticleData of different sizes");
        }
        PyParticleData<float> output(this->size());
        this->binary_map(other, output, [](const T a, const S b){return a+b;});
        return output;
    }


    /**
     *  addition of two PyParticleData, where neither is float -> return largest type (case S > T)
     */
    template<typename S, std::enable_if_t<!std::is_floating_point<T>::value && \
                                          !std::is_floating_point<S>::value && \
                                          (sizeof(S) > sizeof(T)), bool> = false>
    PyParticleData<S> operator+(const PyParticleData<S>& other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot add PyParticleData of different sizes");
        }
        PyParticleData<S> output(this->size());
        this->binary_map(other, output, [](const T a, const S b){return a+b;});
        return output;
    }


    /**
     *  addition of two PyParticleData, where neither is float -> return largest type (case S <= T)
     */
    template<typename S, std::enable_if_t<!std::is_floating_point<T>::value && \
                                          !std::is_floating_point<S>::value && \
                                          (sizeof(S) <= sizeof(T)), bool> = false>
    PyParticleData<T> operator+(const PyParticleData<S>& other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot add PyParticleData of different sizes");
        }
        PyParticleData<T> output(this->size());
        this->binary_map(other, output, [](const T a, const S b){return a+b;});
        return output;
    }


    /**
     *  subtraction of two PyParticleData, where at least one is float -> return float
     */
    template<typename S, std::enable_if_t<std::is_floating_point<T>::value || std::is_floating_point<S>::value, bool> = true>
    PyParticleData<float> operator-(const PyParticleData<S>& other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot subtract PyParticleData of different sizes");
        }
        PyParticleData<float> output(this->size());
        this->binary_map(other, output, [](const T a, const S b){return a-b;});
        return output;
    }

    /**
     *  subtraction of two PyParticleData, where neither is float -> return largest type (case S > T)
     */
    template<typename S, std::enable_if_t<!std::is_floating_point<T>::value && \
                                          !std::is_floating_point<S>::value && \
                                          (sizeof(S) > sizeof(T)), bool> = false>
    PyParticleData<S> operator-(const PyParticleData<S>& other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot subtract PyParticleData of different sizes");
        }
        PyParticleData<S> output(this->size());
        this->binary_map(other, output, [](const T a, const S b){return a-b;});
        return output;
    }

    /**
     *  subtraction of two PyParticleData, where neither is float -> return largest type (case S <= T)
     */
    template<typename S, std::enable_if_t<!std::is_floating_point<T>::value && \
                                          !std::is_floating_point<S>::value && \
                                          (sizeof(S) <= sizeof(T)), bool> = false>
    PyParticleData<T> operator-(const PyParticleData<S>& other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot subtract PyParticleData of different sizes");
        }
        PyParticleData<T> output(this->size());
        this->binary_map(other, output, [](const T a, const S b){return a-b;});
        return output;
    }


    /**
     *  elementwise multiplication of two PyParticleData, where at least one is float -> return float
     */
    template<typename S, std::enable_if_t<std::is_floating_point<T>::value || std::is_floating_point<S>::value, bool> = true>
    PyParticleData<float> operator*(const PyParticleData<S>& other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot multiply PyParticleData of different sizes");
        }
        PyParticleData<float> output(this->size());
        this->binary_map(other, output, [](const T a, const S b){return a*b;});
        return output;
    }


    /**
     *  elementwise multiplication of two PyParticleData, where none is float -> return largest type (case S > T)
     */
    template<typename S, std::enable_if_t<!std::is_floating_point<T>::value && \
                                          !std::is_floating_point<S>::value && \
                                          (sizeof(S) > sizeof(T)), bool> = false>
    PyParticleData<S> operator*(const PyParticleData<S>& other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot multiply PyParticleData of different sizes");
        }
        PyParticleData<S> output(this->size());
        this->binary_map(other, output, [](const T a, const S b){return a*b;});
        return output;
    }

    /**
     *  elementwise multiplication of two PyParticleData, where none is float -> return largest type (case S <= T)
     */
    template<typename S, std::enable_if_t<!std::is_floating_point<T>::value && \
                                          !std::is_floating_point<S>::value && \
                                          (sizeof(S) <= sizeof(T)), bool> = false>
    PyParticleData<T> operator*(const PyParticleData<S>& other) const {
        if(this->size() != other.size()) {
            throw std::invalid_argument("cannot multiply PyParticleData of different sizes");
        }
        PyParticleData<T> output(this->size());
        this->binary_map(other, output, [](const T a, const S b){return a*b;});
        return output;
    }


    /**
     * Set the quantization factor for lossy compression step
     *
     * @param q  float compression quantization factor, larger -> more loss
     */
    void set_quantization_factor(float q){ this->compressor.set_quantization_factor(q); }

    /**
     * Set the background offset and threshold for lossy compression transform (values below this will be ignored
     * and set to the value of bkgrd.
     *
     * @param bkgrd  lower truncation value of the lossy sqrt transform
     */
    void set_background(float bkgrd){ this->compressor.set_background(bkgrd); }

    /**
     * Turns compression on and off.
     *
     * @param type  int (1 turns lossy compresison on, 0 should be off)
     */
    void set_compression_type(bool type){ this->compressor.set_compression_type(type); }


    /**
     * Sample the particle values from an image (numpy array) for a given APR (computed from the same image).
     *
     * @param apr     APR object
     * @param img     numpy array representing the image
     */
    template<typename ImageType>
    void sample_image_py(APR &apr, py::array_t<ImageType, py::array::c_style> &img) {

        if(std::is_integral<T>::value && (std::is_floating_point<ImageType>::value || sizeof(T) < sizeof(ImageType))) {
            std::cerr << "Warning: narrowing conversion from " << TypeParseTraits<ImageType>::name() << " to " <<
                         TypeParseTraits<T>::name() << " in PyParticleData::sample_image" << std::endl;
        }
        auto buf = img.request(false);
        int y_num, x_num, z_num;

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
            throw std::invalid_argument("sample_image: input array must be of dimension 1-3");
        }

        auto ptr = static_cast<ImageType*>(buf.ptr);

        PixelData<ImageType> pd;
        pd.init_from_mesh(y_num, x_num, z_num, ptr);

        this->sample_image(apr, pd);
    }


    /**
     * Sample the particle values from a TIFF file, that is read in blocks of z-slices to reduce memory usage.
     *
     * @param apr        APR object
     * @param aFileName  Path to the TIFF file
     * @param blockSize  Number of z-slices to process in each tile
     * @param ghostSize  Number of ghost slices on each side of the block (maximum slices held in memory at any given time is blockSize + 2*ghostSize)
     */
    void sample_image_blocked(APR &apr,
                              const std::string &aFileName,
                              const int blockSize,
                              const int ghostSize) {
        this->sample_parts_from_img_blocked(apr, aFileName, blockSize, ghostSize);
    }


    /**
     * Fill the particle values with their corresponding resolution levels.
     *
     * @param apr  APR object
     */
    void fill_with_levels(APR &apr) {

        this->resize(apr.total_number_particles());
        auto it = apr.iterator();

        for (int level = it.level_min(); level <= it.level_max(); ++level) {

            T lvl = static_cast<T>(level);

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it) collapse(2)
#endif
            for (int z = 0; z < it.z_num(level); ++z) {
                for (int x = 0; x < it.x_num(level); ++x) {
                    for (it.begin(level, z, x); it < it.end(); it++) {

                        this->data[it] = lvl;
                    }
                }
            }
        }
    }
};


template<typename DataType>
void AddPyParticleData(pybind11::module &m, const std::string &aTypeString) {

    using namespace py::literals;

    /// wrap base class (ParticleData)
    using BaseParticles = ParticleData<DataType>;
    std::string baseName = aTypeString + "Particles_CPP";
    py::class_<BaseParticles>(m, baseName.c_str());

    /// wrap derived class (PyParticleData)
    using TypeParticles = PyParticleData<DataType>;
    std::string typeStr = aTypeString + "Particles";
    py::class_<TypeParticles, BaseParticles>(m, typeStr.c_str(), py::buffer_protocol())
            .def(py::init())
            .def(py::init([](uint64_t num_particles) { return TypeParticles(num_particles); }))
            .def(py::init([](APR& apr) { return TypeParticles(apr); }))
            .def(py::init([](py::array_t<DataType, py::array::c_style | py::array::forcecast>& arr){ return TypeParticles(arr); }))
            .def(py::init([](PyParticleData<float>& other){ return TypeParticles(other); }))
            .def(py::init([](PyParticleData<uint8_t>& other){ return TypeParticles(other); }))
            .def(py::init([](PyParticleData<uint16_t>& other){ return TypeParticles(other); }))
            .def(py::init([](PyParticleData<uint64_t>& other){ return TypeParticles(other); }))
            .def("__repr__", [typeStr](const TypeParticles &p) { return typeStr + "(size " + std::to_string(p.size()) + ")"; })
            .def("__len__", [](const TypeParticles &p){ return p.size(); })
            .def("__contains__", [](const TypeParticles &p, DataType v) { return p.contains(v); })
            .def("resize", &TypeParticles::resize, "resize the data array to a specified number of elements")
            .def("fill", &TypeParticles::fill, "fill all elements with a given value", "value"_a)
            .def("min", &TypeParticles::min, "return the minimum value")
            .def("max", &TypeParticles::max, "return the maximum value")
            .def("copy", &TypeParticles::template copy<uint8_t>, "copy data from ByteParticles object",
                 "partsToCopy"_a)
            .def("copy", &TypeParticles::template copy<uint16_t>, "copy data from ShortParticles object",
                 "partsToCopy"_a)
            .def("copy", &TypeParticles::template copy<uint64_t>, "copy data from LongParticles object",
                 "partsToCopy"_a)
            .def("copy", &TypeParticles::template copy<float>, "copy data from FloatParticles object",
                 "partsToCopy"_a)
            .def("copy", &TypeParticles::template copy_parts<uint8_t>,
                 "partially copy data from ByteParticles object, up to a certain level (default 0 -> all levels)",
                 "apr"_a, "partsToCopy"_a, "level"_a=0, "numBlocks"_a=10)
            .def("copy", &TypeParticles::template copy_parts<uint16_t>,
                 "partially copy data from ShortParticles object, up to a certain level (default 0 -> all levels)",
                 "apr"_a, "partsToCopy"_a, "level"_a=0, "numBlocks"_a=10)
            .def("copy", &TypeParticles::template copy_parts<uint64_t>,
                 "partially copy data from LongParticles object, up to a certain level (default 0 -> all levels)",
                 "apr"_a, "partsToCopy"_a, "level"_a=0, "numBlocks"_a=10)
            .def("copy", &TypeParticles::template copy_parts<float>,
                 "partially copy data from FloatParticles object, up to a certain level (default 0 -> all levels)",
                 "apr"_a, "partsToCopy"_a, "level"_a=0, "numBlocks"_a=10)
            .def("copy", &TypeParticles::ret_copy, "return a copy of self")
            .def("sample_image", &TypeParticles::template sample_image_py<uint8_t>, "sample particle values from an image (numpy array)",
                 "apr"_a, "img"_a.noconvert())
            .def("sample_image", &TypeParticles::template sample_image_py<uint16_t>, "sample particle values from an image (numpy array)",
                 "apr"_a, "img"_a.noconvert())
            .def("sample_image", &TypeParticles::template sample_image_py<uint64_t>, "sample particle values from an image (numpy array)",
                 "apr"_a, "img"_a.noconvert())
            .def("sample_image", &TypeParticles::template sample_image_py<float>, "sample particle values from an image (numpy array)",
                 "apr"_a, "img"_a.noconvert())
            .def("sample_image_blocked", &TypeParticles::sample_image_blocked,
                 "sample particle values from a file in z-blocks to reduce memory usage",
                 "apr"_a, "fileName"_a, "blockSize"_a, "ghostSize"_a)
            .def("fill_with_levels", &TypeParticles::fill_with_levels, "fill particle values with levels",
                 "apr"_a)
            .def("set_quantization_factor", &TypeParticles::set_quantization_factor, "set lossy quantization factor",
                 "value"_a)
            .def("set_background", &TypeParticles::set_background, "set lossy background cut off",
                 "value"_a)
            .def("set_compression_type", &TypeParticles::set_compression_type, "turn lossy compression on (1) or off (0)",
                 "value"_a)
            .def(py::self += PyParticleData<uint8_t>())
            .def(py::self += PyParticleData<uint16_t>())
            .def(py::self += PyParticleData<uint64_t>())
            .def(py::self += PyParticleData<float>())
            .def(py::self -= PyParticleData<uint8_t>())
            .def(py::self -= PyParticleData<uint16_t>())
            .def(py::self -= PyParticleData<uint64_t>())
            .def(py::self -= PyParticleData<float>())
            .def(py::self *= PyParticleData<uint8_t>())
            .def(py::self *= PyParticleData<uint16_t>())
            .def(py::self *= PyParticleData<uint64_t>())
            .def(py::self *= PyParticleData<float>())
            .def(py::self *= float())
            .def(py::self += float())
            .def(py::self -= float())
            .def(py::self + PyParticleData<uint8_t>())
            .def(py::self + PyParticleData<uint16_t>())
            .def(py::self + PyParticleData<uint64_t>())
            .def(py::self + PyParticleData<float>())
            .def(py::self - PyParticleData<uint8_t>())
            .def(py::self - PyParticleData<uint16_t>())
            .def(py::self - PyParticleData<uint64_t>())
            .def(py::self - PyParticleData<float>())
            .def(py::self * PyParticleData<uint8_t>())
            .def(py::self * PyParticleData<uint16_t>())
            .def(py::self * PyParticleData<uint64_t>())
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
            /// access single element
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
            .def("__setitem__", [](TypeParticles &p, py::slice slice, py::array_t<DataType> &value) {
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

            .def("__iter__", [](const TypeParticles &s) { return py::make_iterator(s.begin(), s.end()); },
                         py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */)
            .def_buffer([](TypeParticles &p) -> py::buffer_info{
                return py::buffer_info(
                        p.begin(),
                        sizeof(DataType),
                        py::format_descriptor<DataType>::format(),
                        1,
                        {p.size()},
                        {sizeof(DataType)}
                );
            });
}

#endif //PYLIBAPR_PYPARTICLEDATA_HPP