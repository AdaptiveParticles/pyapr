
#ifndef PYLIBAPR_BINDPROJECTION_HPP
#define PYLIBAPR_BINDPROJECTION_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#include "data_structures/APR/APR.hpp"
#include "numerics/APRReconstruction.hpp"
#include "data_containers/src/BindParticleData.hpp"

#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <set>

namespace py = pybind11;
using namespace py::literals;


template<typename T>
PixelData<T> maximum_projection_y(APR& apr, PyParticleData<T>& parts) {
    auto it = apr.iterator();

    const int z_num = it.z_num(it.level_max());
    const int x_num = it.x_num(it.level_max());
    const T minval = std::numeric_limits<T>::min();
    PixelData<T> mip(x_num, z_num, 1, minval);

    for(int level = it.level_max(); level >= it.level_min(); --level) {
        const int level_size = it.level_size(level);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = 0; z < it.z_num(level); ++z) {
            for(int x = 0; x < it.x_num(level); ++x) {

                const size_t offset_begin = it.begin(level, z, x);
                const size_t offset_end = it.end();

                if(offset_begin < offset_end) {
                    const T row_max = *std::max_element(parts.begin()+offset_begin, parts.begin()+offset_end);

                    const int z_m = z * level_size;
                    const int x_m = x * level_size;

                    for (int i = z_m; i < std::min(z_m + level_size, z_num); ++i) {
                        for (int j = x_m; j < std::min(x_m + level_size, x_num); ++j) {
                            mip.at(j, i, 0) = std::max(mip.at(j, i, 0), row_max);
                        }
                    }
                }
            }
        }
    }
    return mip;
}


template<typename T>
PixelData<T> maximum_projection_x(APR& apr, PyParticleData<T>& parts) {
    auto it = apr.iterator();

    const int z_num = it.z_num(it.level_max());
    const int y_num = it.y_num(it.level_max());
    const T minval = std::numeric_limits<T>::min();
    PixelData<T> mip(y_num, z_num, 1, minval);

    for(int level = it.level_max(); level >= it.level_min(); --level) {
        const int level_size = it.level_size(level);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = 0; z < it.z_num(level); ++z) {
            const int z_m = z * level_size;
            for(int x = 0; x < it.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {

                    const int y_m = it.y() * level_size;
                    for (int i = z_m; i < std::min(z_m + level_size, z_num); ++i) {
                        for (int j = y_m; j < std::min(y_m + level_size, y_num); ++j) {
                            mip.at(j, i, 0) = std::max(mip.at(j, i, 0), parts[it]);
                        }
                    }
                }
            }
        }
    }
    return mip;
}


template<typename T>
PixelData<T> maximum_projection_z(APR& apr, PyParticleData<T>& parts) {
    auto it = apr.iterator();

    const int x_num = it.x_num(it.level_max());
    const int y_num = it.y_num(it.level_max());
    const T minval = std::numeric_limits<T>::min();
    PixelData<T> mip(y_num, x_num, 1, minval);

    for(int level = it.level_max(); level >= it.level_min(); --level) {
        const int level_size = it.level_size(level);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int x = 0; x < it.x_num(level); ++x) {  // swapped x and z loops to avoid race conditions
            const int x_m = x * level_size;
            for(int z = 0; z < it.z_num(level); ++z) {
                for(it.begin(level, z, x); it < it.end(); ++it) {

                    const int y_m = it.y() * level_size;
                    for (int i = x_m; i < std::min(x_m + level_size, x_num); ++i) {
                        for (int j = y_m; j < std::min(y_m + level_size, y_num); ++j) {
                            mip.at(j, i, 0) = std::max(mip.at(j, i, 0), parts[it]);
                        }
                    }
                }
            }
        }
    }
    return mip;
}


/// maximum projections in a subregion of the image, specified by a ReconPatch struct

template<typename T>
PixelData<T> maximum_projection_y_patch(APR& apr, PyParticleData<T>& parts, ReconPatch& patch) {
    auto it = apr.iterator();

    int tmp = patch.level_delta;
    patch.level_delta = 0;
    patch.check_limits(apr);
    patch.level_delta = tmp;

    const T minval = std::numeric_limits<T>::min();
    PixelData<T> res(patch.x_end-patch.x_begin, patch.z_end-patch.z_begin, 1, minval);

    for(int level = it.level_max(); level > it.level_min(); --level) {
        const int level_size = it.level_size(level);

        const int z_begin_l = patch.z_begin / level_size;
        const int x_begin_l = patch.x_begin / level_size;
        const int y_begin_l = patch.y_begin / level_size;
        const int z_end_l = (patch.z_end + level_size - 1) / level_size;
        const int x_end_l = (patch.x_end + level_size - 1) / level_size;
        const int y_end_l = (patch.y_end + level_size - 1) / level_size;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = z_begin_l; z < z_end_l; ++z) {
            const int zp_begin = std::max(z * level_size, patch.z_begin) - patch.z_begin;
            const int zp_end = std::min((z+1) * level_size, patch.z_end) - patch.z_begin;

            for(int x = x_begin_l; x < x_end_l; ++x) {

                const int xp_begin = std::max(x * level_size, patch.x_begin) - patch.x_begin;
                const int xp_end = std::min((x+1) * level_size, patch.x_end) - patch.x_begin;

                // find start of y region
                it.begin(level, z, x);
                while(it.y() < y_begin_l && it < it.end()) { ++it; }

                // find max in y region
                T row_max = minval;
                for(; (it < it.end()) && (it.y() < y_end_l); ++it) {
                    row_max = std::max(row_max, parts[it]);
                }

                // compare to projection
                for (int i = zp_begin; i < zp_end; ++i) {
                    for (int j = xp_begin; j < xp_end; ++j) {
                        res.at(j, i, 0) = std::max(res.at(j, i, 0), row_max);
                    }
                }
            }
        }
    }
    return res;
}


template<typename T>
PixelData<T> maximum_projection_x_patch(APR& apr, PyParticleData<T>& parts, ReconPatch& patch) {
    auto it = apr.iterator();

    int tmp = patch.level_delta;
    patch.level_delta = 0;
    patch.check_limits(apr);
    patch.level_delta = tmp;

    const T minval = std::numeric_limits<T>::min();
    PixelData<T> res(patch.y_end-patch.y_begin, patch.z_end-patch.z_begin, 1, minval);

    for(int level = it.level_max(); level > it.level_min(); --level) {
        const int level_size = it.level_size(level);

        const int z_begin_l = patch.z_begin / level_size;
        const int x_begin_l = patch.x_begin / level_size;
        const int y_begin_l = patch.y_begin / level_size;
        const int z_end_l = (patch.z_end + level_size - 1) / level_size;
        const int x_end_l = (patch.x_end + level_size - 1) / level_size;
        const int y_end_l = (patch.y_end + level_size - 1) / level_size;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = z_begin_l; z < z_end_l; ++z) {
            const int zp_begin = std::max(z * level_size, patch.z_begin) - patch.z_begin;
            const int zp_end = std::min((z+1) * level_size, patch.z_end) - patch.z_begin;

            for(int x = x_begin_l; x < x_end_l; ++x) {

                // find start of y region
                it.begin(level, z, x);
                while(it.y() < y_begin_l && it < it.end()) { ++it; }

                // iterate over y region and project
                for(; (it < it.end()) && (it.y() < y_end_l); ++it) {
                    const int yp_begin = std::max(it.y() * level_size, patch.y_begin) - patch.y_begin;
                    const int yp_end = std::min((it.y()+1) * level_size, patch.y_end) - patch.y_begin;

                    for (int i = zp_begin; i < zp_end; ++i) {
                        for (int j = yp_begin; j < yp_end; ++j) {
                            res.at(j, i, 0) = std::max(res.at(j, i, 0), parts[it]);
                        }
                    }
                }
            }
        }
    }
    return res;
}


template<typename T>
PixelData<T> maximum_projection_z_patch(APR& apr, PyParticleData<T>& parts, ReconPatch& patch) {
    auto it = apr.iterator();

    int tmp = patch.level_delta;
    patch.level_delta = 0;
    patch.check_limits(apr);
    patch.level_delta = tmp;

    const T minval = std::numeric_limits<T>::min();
    PixelData<T> res(patch.y_end-patch.y_begin, patch.x_end-patch.x_begin, 1, minval);

    for(int level = it.level_max(); level > it.level_min(); --level) {
        const int level_size = it.level_size(level);

        const int z_begin_l = patch.z_begin / level_size;
        const int x_begin_l = patch.x_begin / level_size;
        const int y_begin_l = patch.y_begin / level_size;
        const int z_end_l = (patch.z_end + level_size - 1) / level_size;
        const int x_end_l = (patch.x_end + level_size - 1) / level_size;
        const int y_end_l = (patch.y_end + level_size - 1) / level_size;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int x = x_begin_l; x < x_end_l; ++x) {  // swapped x and z loops to avoid race conditions
            const int xp_begin = std::max(x * level_size, patch.x_begin) - patch.x_begin;
            const int xp_end = std::min((x+1) * level_size, patch.x_end) - patch.x_begin;

            for(int z = z_begin_l; z < z_end_l; ++z) {

                // find start of y region
                it.begin(level, z, x);
                while(it.y() < y_begin_l && it < it.end()) { ++it; }

                // iterate over y region and project
                for(; (it < it.end()) && (it.y() < y_end_l); ++it) {
                    const int yp_begin = std::max(it.y() * level_size, patch.y_begin) - patch.y_begin;
                    const int yp_end = std::min((it.y()+1) * level_size, patch.y_end) - patch.y_begin;

                    for (int i = xp_begin; i < xp_end; ++i) {
                        for (int j = yp_begin; j < yp_end; ++j) {
                            res.at(j, i, 0) = std::max(res.at(j, i, 0), parts[it]);
                        }
                    }
                }
            }
        }
    }
    return res;
}


template<typename T>
PixelData<T> maximum_projection_y_alt(APR& apr, PyParticleData<T>& parts) {
    auto it = apr.iterator();

    std::vector<PixelData<T>> image_vec;
    image_vec.resize(it.level_max()+1);
    const T minval = std::numeric_limits<T>::min();
    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        image_vec[level].initWithValue(it.x_num(level), it.z_num(level), 1, minval);
    }

    for(int level = it.level_max(); level >= it.level_min(); --level) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = 0; z < it.z_num(level); ++z) {
            for (int x = 0; x < it.x_num(level); ++x) {
                const size_t offset_begin = it.begin(level, z, x);
                const size_t offset_end = it.end();
                if(offset_begin < offset_end) {
                    const T row_max = *std::max_element(parts.begin()+offset_begin, parts.begin()+offset_end);
                    image_vec[level].at(x, z, 0) = row_max;
                }
            }
        }
    }

    const int l_max = it.level_max();
    const int z_num = it.z_num(l_max);
    const int x_num = it.x_num(l_max);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int z = 0; z < z_num; ++z) {
        for(int level = it.level_max()-1; level >= it.level_min(); --level) {
            const int level_size = it.level_size(level);
            const int z_l = z / level_size;
            for (int x = 0; x < x_num; ++x) {
                const int x_l = x / level_size;
                image_vec[l_max].at(x, z, 0) = std::max(image_vec[l_max].at(x, z, 0), image_vec[level].at(x_l, z_l, 0));
            }
        }
    }
    PixelData<T> res;
    res.swap(image_vec[l_max]);
    return res;
}


template<typename T>
PixelData<T> maximum_projection_x_alt(APR& apr, PyParticleData<T>& parts) {
    auto it = apr.iterator();

    std::vector<PixelData<T>> image_vec;
    image_vec.resize(it.level_max()+1);
    const T minval = std::numeric_limits<T>::min();
    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        image_vec[level].initWithValue(it.y_num(level), it.z_num(level), 1, minval);
    }

    for(int level = it.level_max(); level >= it.level_min(); --level) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = 0; z < it.z_num(level); ++z) {
            for (int x = 0; x < it.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    image_vec[level].at(it.y(), z, 0) = std::max(image_vec[level].at(it.y(), z, 0), parts[it]);
                }
            }
        }
    }

    const int l_max = it.level_max();
    const int z_num = it.z_num(l_max);
    const int y_num = it.y_num(l_max);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int z = 0; z < z_num; ++z) {

        for(int level = it.level_max()-1; level >= it.level_min(); --level) {
            const int level_size = it.level_size(level);
            const int z_l = z / level_size;
            for (int y = 0; y < y_num; ++y) {
                const int y_l = y / level_size;
                image_vec[l_max].at(y, z, 0) = std::max(image_vec[l_max].at(y, z, 0), image_vec[level].at(y_l, z_l, 0));
            }
        }
    }
    PixelData<T> res;
    res.swap(image_vec[l_max]);
    return res;
}


template<typename T>
PixelData<T> maximum_projection_z_alt(APR& apr, PyParticleData<T>& parts) {
    auto it = apr.iterator();

    std::vector<PixelData<T>> image_vec;
    image_vec.resize(it.level_max()+1);
    const T minval = std::numeric_limits<T>::min();
    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        image_vec[level].initWithValue(it.y_num(level), it.x_num(level), 1, minval);
    }

    for(int level = it.level_max(); level >= it.level_min(); --level) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for (int x = 0; x < it.x_num(level); ++x) {
            for (int z = 0; z < it.z_num(level); ++z) {
                for(it.begin(level, z, x); it < it.end(); ++it) {
                    image_vec[level].at(it.y(), x, 0) = std::max(image_vec[level].at(it.y(), x, 0), parts[it]);
                }
            }
        }
    }

    const int l_max = it.level_max();
    const int x_num = it.x_num(l_max);
    const int y_num = it.y_num(l_max);
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int x = 0; x < x_num; ++x) {

        for(int level = it.level_max()-1; level >= it.level_min(); --level) {
            const int level_size = it.level_size(level);
            const int x_l = x / level_size;
            for (int y = 0; y < y_num; ++y) {
                const int y_l = y / level_size;
                image_vec[l_max].at(y, x, 0) = std::max(image_vec[l_max].at(y, x, 0), image_vec[level].at(y_l, x_l, 0));
            }
        }
    }
    PixelData<T> res;
    res.swap(image_vec[l_max]);
    return res;
}


/// maximum projections in a subregion of the image, specified by a ReconPatch struct
template<typename T>
PixelData<T> maximum_projection_y_alt_patch(APR& apr, PyParticleData<T>& parts, ReconPatch& patch) {
    auto it = apr.iterator();
    std::vector<PixelData<T>> image_vec;
    image_vec.resize(it.level_max()+1);
    const T minval = std::numeric_limits<T>::min();

    // Instantiate a vector which will contain the max proj for each level
    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        const int level_size = it.level_size(level);
        const int z_begin_l = patch.z_begin / level_size;
        const int x_begin_l = patch.x_begin / level_size;
        const int z_end_l = (patch.z_end + level_size - 1) / level_size;
        const int x_end_l = (patch.x_end + level_size - 1) / level_size;

        image_vec[level].initWithValue(x_end_l-x_begin_l, z_end_l-z_begin_l, 1, minval);
    }

    // Compute the max proj for each level
    for(int level = it.level_max(); level >= it.level_min(); --level) {

        const int level_size = it.level_size(level);
        const int z_begin_l = patch.z_begin / level_size;
        const int x_begin_l = patch.x_begin / level_size;
        const int y_begin_l = patch.y_begin / level_size;
        const int z_end_l = (patch.z_end + level_size - 1) / level_size;
        const int x_end_l = (patch.x_end + level_size - 1) / level_size;
        const int y_end_l = (patch.y_end + level_size - 1) / level_size;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = z_begin_l; z < z_end_l; ++z) {
            for (int x = x_begin_l; x < x_end_l; ++x) {

                // find start of y region
                it.begin(level, z, x);
                while(it.y() < y_begin_l && it < it.end()) { ++it; }

                // find max in (x, z) column
                T row_max = minval;
                for(; it.y() < y_end_l && it < it.end(); ++it) {
                    row_max = std::max(row_max, parts[it]);
                }
                image_vec[level].at(x-x_begin_l, z-z_begin_l, 0) = row_max;
            }
        }
    }

    const int l_max = it.level_max();

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    // Max project on the levels
    for(int z = patch.z_begin; z < patch.z_end; ++z) {
        for(int level = it.level_max()-1; level >= it.level_min(); --level) {
            const int level_size = it.level_size(level);
            const int z_begin_l = patch.z_begin / level_size;
            const int x_begin_l = patch.x_begin / level_size;
            const int z_l = z / level_size;
            for (int x = patch.x_begin; x < patch.x_end; ++x) {
                const int x_l = x / level_size;
                image_vec[l_max].at(x-patch.x_begin, z-patch.z_begin, 0) = std::max(image_vec[l_max].at(x-patch.x_begin, z-patch.z_begin, 0), image_vec[level].at(x_l-x_begin_l, z_l-z_begin_l, 0));
            }
        }
    }
    PixelData<T> res;
    res.swap(image_vec[l_max]);
    return res;
}


template<typename T>
PixelData<T> maximum_projection_x_alt_patch(APR& apr, PyParticleData<T>& parts, ReconPatch& patch) {
    auto it = apr.iterator();
    std::vector<PixelData<T>> image_vec;
    image_vec.resize(it.level_max()+1);
    const T minval = std::numeric_limits<T>::min();

    // Instantiate a vector which will contain the max proj for each level
    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        const int level_size = it.level_size(level);
        const int z_begin_l = patch.z_begin / level_size;
        const int y_begin_l = patch.y_begin / level_size;
        const int z_end_l = (patch.z_end + level_size - 1) / level_size;
        const int y_end_l = (patch.y_end + level_size - 1) / level_size;


        image_vec[level].initWithValue(y_end_l-y_begin_l, z_end_l-z_begin_l, 1, minval);
    }

    // Compute the max proj for each level
    for(int level = it.level_max(); level >= it.level_min(); --level) {

        const int level_size = it.level_size(level);
        const int z_begin_l = patch.z_begin / level_size;
        const int x_begin_l = patch.x_begin / level_size;
        const int y_begin_l = patch.y_begin / level_size;
        const int z_end_l = (patch.z_end + level_size - 1) / level_size;
        const int x_end_l = (patch.x_end + level_size - 1) / level_size;
        const int y_end_l = (patch.y_end + level_size - 1) / level_size;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int z = z_begin_l; z < z_end_l; ++z) {
            for (int x = x_begin_l; x < x_end_l; ++x) {

                // find start of y region
                it.begin(level, z, x);
                while(it.y() < y_begin_l && it < it.end()) { ++it; }

                // find max in (x, z) column
                for(; it.y() < y_end_l && it < it.end(); ++it) {
                    image_vec[level].at(it.y()-y_begin_l, z-z_begin_l, 0) = std::max(image_vec[level].at(it.y()-y_begin_l, z-z_begin_l, 0), parts[it]);
                }
            }
        }
    }

    const int l_max = it.level_max();

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    // Max project on the levels
    for(int z = patch.z_begin; z < patch.z_end; ++z) {
        for(int level = it.level_max()-1; level >= it.level_min(); --level) {
            const int level_size = it.level_size(level);
            const int z_begin_l = patch.z_begin / level_size;
            const int y_begin_l = patch.y_begin / level_size;
            const int z_l = z / level_size;
            for (int y = patch.y_begin; y < patch.y_end; ++y) {
                const int y_l = y / level_size;
                image_vec[l_max].at(y-patch.y_begin, z-patch.z_begin, 0) = std::max(image_vec[l_max].at(y-patch.y_begin, z-patch.z_begin, 0), image_vec[level].at(y_l-y_begin_l, z_l-z_begin_l, 0));
            }
        }
    }
    PixelData<T> res;
    res.swap(image_vec[l_max]);
    return res;
}


template<typename T>
PixelData<T> maximum_projection_z_alt_patch(APR& apr, PyParticleData<T>& parts, ReconPatch& patch) {
    auto it = apr.iterator();
    std::vector<PixelData<T>> image_vec;
    image_vec.resize(it.level_max()+1);
    const T minval = std::numeric_limits<T>::min();

    // Instantiate a vector which will contain the max proj for each level
    for(int level = it.level_min(); level <= it.level_max(); ++level) {
        const int level_size = it.level_size(level);
        const int x_begin_l = patch.x_begin / level_size;
        const int y_begin_l = patch.y_begin / level_size;
        const int x_end_l = (patch.x_end + level_size - 1) / level_size;
        const int y_end_l = (patch.y_end + level_size - 1) / level_size;


        image_vec[level].initWithValue(y_end_l-y_begin_l, x_end_l-x_begin_l, 1, minval);
    }

    // Compute the max proj for each level
    for(int level = it.level_max(); level >= it.level_min(); --level) {

        const int level_size = it.level_size(level);
        const int z_begin_l = patch.z_begin / level_size;
        const int x_begin_l = patch.x_begin / level_size;
        const int y_begin_l = patch.y_begin / level_size;
        const int z_end_l = (patch.z_end + level_size - 1) / level_size;
        const int x_end_l = (patch.x_end + level_size - 1) / level_size;
        const int y_end_l = (patch.y_end + level_size - 1) / level_size;

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) firstprivate(it)
#endif
        for(int x = x_begin_l; x < x_end_l; ++x) {
            for (int z = z_begin_l; z < z_end_l; ++z) {

                // find start of y region
                it.begin(level, z, x);
                while(it.y() < y_begin_l && it < it.end()) { ++it; }

                // find max in (x, z) column
                for(; it.y() < y_end_l && it < it.end(); ++it) {
                    image_vec[level].at(it.y()-y_begin_l, x-x_begin_l, 0) = std::max(image_vec[level].at(it.y()-y_begin_l, x-x_begin_l, 0), parts[it]);
                }
            }
        }
    }

    const int l_max = it.level_max();

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    // Max project on the levels
    for(int x = patch.x_begin; x < patch.x_end; ++x) {
        for(int level = it.level_max()-1; level >= it.level_min(); --level) {
            const int level_size = it.level_size(level);
            const int x_begin_l = patch.x_begin / level_size;
            const int y_begin_l = patch.y_begin / level_size;
            const int x_l = x / level_size;
            for (int y = patch.y_begin; y < patch.y_end; ++y) {
                const int y_l = y / level_size;
                image_vec[l_max].at(y-patch.y_begin, x-patch.x_begin, 0) = std::max(image_vec[l_max].at(y-patch.y_begin, x-patch.x_begin, 0), image_vec[level].at(y_l-y_begin_l, x_l-x_begin_l, 0));
            }
        }
    }
    PixelData<T> res;
    res.swap(image_vec[l_max]);
    return res;
}


template<typename T>
void bindProjectionY(py::module& m) {
    m.def("max_projection_y", &maximum_projection_y<T>, "maximum projection along y axis",
          "apr"_a, "parts"_a);

    m.def("max_projection_y_alt", &maximum_projection_y_alt<T>, "maximum projection along y axis",
          "apr"_a, "parts"_a);

    m.def("max_projection_y", &maximum_projection_y_patch<T>, "maximum projection along y axis",
          "apr"_a, "parts"_a, "patch"_a);

    m.def("max_projection_y_alt", &maximum_projection_y_alt_patch<T>, "maximum projection along y axis",
          "apr"_a, "parts"_a, "patch"_a);
}

template<typename T>
void bindProjectionX(py::module& m) {
    m.def("max_projection_x", &maximum_projection_x<T>, "maximum projection along x axis",
          "apr"_a, "parts"_a);

    m.def("max_projection_x_alt", &maximum_projection_x_alt<T>, "maximum projection along x axis",
          "apr"_a, "parts"_a);

    m.def("max_projection_x", &maximum_projection_x_patch<T>, "maximum projection along x axis",
          "apr"_a, "parts"_a, "patch"_a);

    m.def("max_projection_x_alt", &maximum_projection_x_alt_patch<T>, "maximum projection along x axis",
          "apr"_a, "parts"_a, "patch"_a);
}


template<typename T>
void bindProjectionZ(py::module& m) {
    m.def("max_projection_z", &maximum_projection_z<T>, "maximum projection along z axis",
          "apr"_a, "parts"_a);

    m.def("max_projection_z_alt", &maximum_projection_z_alt<T>, "maximum projection along z axis",
          "apr"_a, "parts"_a);

    m.def("max_projection_z", &maximum_projection_z_patch<T>, "maximum projection along z axis",
          "apr"_a, "parts"_a, "patch"_a);

    m.def("max_projection_z_alt", &maximum_projection_z_alt_patch<T>, "maximum projection along z axis",
          "apr"_a, "parts"_a, "patch"_a);
}


void AddProjection(py::module &m) {

    bindProjectionY<uint8_t>(m);
    bindProjectionY<uint16_t>(m);
    bindProjectionY<uint64_t>(m);
    bindProjectionY<float>(m);

    bindProjectionX<uint8_t>(m);
    bindProjectionX<uint16_t>(m);
    bindProjectionX<uint64_t>(m);
    bindProjectionX<float>(m);

    bindProjectionZ<uint8_t>(m);
    bindProjectionZ<uint16_t>(m);
    bindProjectionZ<uint64_t>(m);
    bindProjectionZ<float>(m);
}

#endif //PYLIBAPR_BINDPROJECTION_HPP
