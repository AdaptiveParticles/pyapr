//
// Created by Joel Jonsson on 09.05.19.
//

#ifndef PYLIBAPR_PYAPRFILTERING_HPP
#define PYLIBAPR_PYAPRFILTERING_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "data_structures/APR/APRTree.hpp"
#include "data_structures/APR/APRTreeIterator.hpp"
#include "data_structures/APR/APRIterator.hpp"

#include "PyAPR.hpp"
#include "PyPixelData.hpp"


namespace py = pybind11;


class PyAPRFiltering {

public:


    //*****************************************************************************************************************
    //                              Forward convolution arbitrary filter size
    //*****************************************************************************************************************

    template<typename ImageType>
    void convolve_batchparallel(APR<ImageType> &apr, float *input_intensities,
                                const std::vector<PixelData<float>> &stencil_vec, float bias, int out_channel,
                                int in_channel, int current_max_level, const uint64_t in_offset,
                                ExtraParticleData<float> &tree_data, APRIterator &apr_iterator,
                                APRTreeIterator &tree_iterator, const size_t number_in_channels,
                                const uint64_t out_offset, float *output_ptr) {

        APRTimer timer(false);

        int stencil_counter = 0;

        const float b = in_channel == (number_in_channels - 1) ? bias : 0.0f;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

            const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                    (int) stencil_vec[stencil_counter].x_num,
                                                    (int) stencil_vec[stencil_counter].z_num};

            const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2,
                                                   (stencil_shape[1] - 1) / 2,
                                                   (stencil_shape[2] - 1) / 2};

            unsigned int z = 0;
            unsigned int x = 0;

            const unsigned int z_num = apr_iterator.spatial_index_z_max(level);

            const unsigned int y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_iterator.spatial_index_y_max(level) +
                                                                            stencil_shape[0] - 1 : 1;
            const unsigned int x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_iterator.spatial_index_x_max(level) +
                                                                            stencil_shape[1] - 1 : 1;

            timer.start_timer("init temp vec");

            PixelData<float> temp_vec;
            temp_vec.init(y_num_m,
                          x_num_m,
                          stencil_shape[2],
                          0); //zero padded boundaries

            timer.stop_timer();

            timer.start_timer("update temp vec first ('pad')");
            //initial condition
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array_ptr(level, padd, apr, apr_iterator, tree_iterator, tree_data, temp_vec,
                                       input_intensities, stencil_shape, stencil_half, in_offset);
            }
            timer.stop_timer();

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - stencil_half[2])) {
                    //update the next z plane for the access
                    timer.start_timer("update_dense_array");
                    update_dense_array_ptr(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                           temp_vec, input_intensities, stencil_shape, stencil_half, in_offset);
                    timer.stop_timer();
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
                    timer.start_timer("padding");

                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                    timer.stop_timer();
                }

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec);

                /// Compute convolution output at apr particles
                timer.start_timer("convolve apr particles");

                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float neigh_sum = 0;

                        for (size_t ix = 0; ix < stencil_shape[1]; ++ix) {
                            for (size_t iy = 0; iy < stencil_shape[0]; ++iy) {
                                neigh_sum += temp_vec.at(iy + apr_iterator.y(), ix + x, 0) *
                                             stencil_vec[stencil_counter].mesh[ix * stencil_shape[0] + iy];
                            }
                        }

                        output_ptr[out_offset + apr_iterator.global_index()] += neigh_sum + b;

                    }//y, pixels/columns (apr)
                }//x , rows (apr)

                timer.stop_timer();

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if (level == current_max_level && current_max_level < apr.level_max()) {

                    timer.start_timer("convolve tree particles");

                    const int64_t tree_offset = compute_tree_offset(apr, level, false);

                    for (x = 0; x < tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            float neigh_sum = 0;

                            for (size_t ix = 0; ix < stencil_shape[1]; ++ix) {
                                for (size_t iy = 0; iy < stencil_shape[0]; ++iy) {
                                    neigh_sum += temp_vec.at(iy + tree_iterator.y(), ix + x, 0) *
                                                 stencil_vec[stencil_counter].mesh[ix * stencil_shape[0] + iy];
                                }
                            }

                            output_ptr[out_offset + tree_iterator.global_index() + tree_offset] += neigh_sum + b;

                        }//y, pixels/columns (tree)
                    }//x, rows (tree)
                    timer.stop_timer();
                } //if
            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);
        }//levels
    }


    //*****************************************************************************************************************
    //                              Forward convolution 3x3 filter
    //*****************************************************************************************************************

    template<typename ImageType>
    void convolve3x3_batchparallel(APR<ImageType> &apr, float *input_intensities,
                                   const std::vector<PixelData<float>> &stencil_vec, float bias, int out_channel,
                                   int in_channel, int current_max_level, const uint64_t in_offset,
                                   ExtraParticleData<float> &tree_data, APRIterator &apr_iterator,
                                   APRTreeIterator &tree_iterator, const size_t number_in_channels,
                                   const uint64_t out_offset, float *output_ptr) {

        APRTimer timer(false);

        int stencil_counter = 0;

        const float b = in_channel==(number_in_channels-1) ? bias : 0.0f;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

            const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                    (int) stencil_vec[stencil_counter].x_num,
                                                    (int) stencil_vec[stencil_counter].z_num};
            const std::vector<int> stencil_half = {(stencil_shape[0] - 1) / 2,
                                                   (stencil_shape[1] - 1) / 2,
                                                   (stencil_shape[2] - 1) / 2};

            // assert stencil_shape compatible with apr org_dims?

            unsigned int z = 0;
            unsigned int x = 0;

            const int z_num = apr_iterator.spatial_index_z_max(level);

            const int y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_iterator.spatial_index_y_max(level) +
                                                                   stencil_shape[0] - 1 : 1;
            const int x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_iterator.spatial_index_x_max(level) +
                                                                   stencil_shape[1] - 1 : 1;

            timer.start_timer("init temp vec");

            PixelData<float> temp_vec;
            temp_vec.init(y_num_m,
                          x_num_m,
                          stencil_shape[2],
                          0); //zero padded boundaries

            timer.stop_timer();

            timer.start_timer("update temp vec first ('pad')");
            //initial condition
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array_ptr(level, padd, apr, apr_iterator, tree_iterator, tree_data, temp_vec,
                                       input_intensities, stencil_shape, stencil_half, in_offset);
            }
            timer.stop_timer();

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - stencil_half[2])) {
                    //update the next z plane for the access
                    timer.start_timer("update_dense_array");
                    update_dense_array_ptr(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                           temp_vec, input_intensities, stencil_shape, stencil_half, in_offset);
                    timer.stop_timer();
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);
                    timer.start_timer("padding");

                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                    timer.stop_timer();
                }

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec);

                /// Compute convolution output at apr particles
                timer.start_timer("convolve apr particles");

                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        float neigh_sum = temp_vec.at(apr_iterator.y(),   x, 0)   * stencil_vec[stencil_counter].mesh[0] +
                                          temp_vec.at(apr_iterator.y()+1, x, 0)   * stencil_vec[stencil_counter].mesh[1] +
                                          temp_vec.at(apr_iterator.y()+2, x, 0)   * stencil_vec[stencil_counter].mesh[2] +
                                          temp_vec.at(apr_iterator.y(),   x+1, 0) * stencil_vec[stencil_counter].mesh[3] +
                                          temp_vec.at(apr_iterator.y()+1, x+1, 0) * stencil_vec[stencil_counter].mesh[4] +
                                          temp_vec.at(apr_iterator.y()+2, x+1, 0) * stencil_vec[stencil_counter].mesh[5] +
                                          temp_vec.at(apr_iterator.y(),   x+2, 0) * stencil_vec[stencil_counter].mesh[6] +
                                          temp_vec.at(apr_iterator.y()+1, x+2, 0) * stencil_vec[stencil_counter].mesh[7] +
                                          temp_vec.at(apr_iterator.y()+2, x+2, 0) * stencil_vec[stencil_counter].mesh[8];

                        output_ptr[out_offset + apr_iterator.global_index()] += neigh_sum + b;

                    }//y, pixels/columns (apr)
                }//x , rows (apr)

                timer.stop_timer();

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    timer.start_timer("convolve tree particles");

                    const int64_t tree_offset = compute_tree_offset(apr, level, false);

                    for (x = 0; x < tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            float neigh_sum = temp_vec.at(tree_iterator.y(),   x, 0)   * stencil_vec[stencil_counter].mesh[0] +
                                              temp_vec.at(tree_iterator.y()+1, x, 0)   * stencil_vec[stencil_counter].mesh[1] +
                                              temp_vec.at(tree_iterator.y()+2, x, 0)   * stencil_vec[stencil_counter].mesh[2] +
                                              temp_vec.at(tree_iterator.y(),   x+1, 0) * stencil_vec[stencil_counter].mesh[3] +
                                              temp_vec.at(tree_iterator.y()+1, x+1, 0) * stencil_vec[stencil_counter].mesh[4] +
                                              temp_vec.at(tree_iterator.y()+2, x+1, 0) * stencil_vec[stencil_counter].mesh[5] +
                                              temp_vec.at(tree_iterator.y(),   x+2, 0) * stencil_vec[stencil_counter].mesh[6] +
                                              temp_vec.at(tree_iterator.y()+1, x+2, 0) * stencil_vec[stencil_counter].mesh[7] +
                                              temp_vec.at(tree_iterator.y()+2, x+2, 0) * stencil_vec[stencil_counter].mesh[8];

                            output_ptr[out_offset + tree_iterator.global_index() + tree_offset] += neigh_sum + b;

                        }//y, pixels/columns (tree)
                    }//x, rows (tree)
                    timer.stop_timer();
                } //if
            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);
        }//levels
    }


    //*****************************************************************************************************************
    //                              Forward convolution 1x1 filter
    //*****************************************************************************************************************

    template<typename ImageType>
    void convolve1x1_batchparallel(APR<ImageType> &apr, float * input_ptr, const std::vector<float> &stencil_vec,
                                   float bias, int out_channel, int in_channel, int current_max_level,
                                   const uint64_t in_offset, APRIterator &apr_iterator, APRTreeIterator &tree_iterator,
                                   const size_t number_in_channels, const uint64_t out_offset, float * output_ptr) {

        APRTimer timer(false);

        const float b = in_channel==(number_in_channels-1) ? bias : 0.0f;

        unsigned int z;
        unsigned int x;

        int stencil_counter = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

            float w = stencil_vec[stencil_counter];

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                /// Compute convolution output at apr particles
                timer.start_timer("convolve 1x1 apr particles");

                for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        output_ptr[out_offset + apr_iterator.global_index()] +=
                                w * input_ptr[in_offset + apr_iterator.global_index()] + b;

                    }//y, pixels/columns (apr)
                }//x , rows (apr)

                timer.stop_timer();

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    timer.start_timer("convolve 1x1 tree particles");

                    const int64_t tree_offset = compute_tree_offset(apr, level, false);

                    for (x = 0; x < tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            const uint64_t idx = tree_iterator.global_index() + tree_offset;

                            output_ptr[out_offset + idx] += w * input_ptr[in_offset + idx] + b;

                        }//y, pixels/columns (tree)
                    }//x, rows (tree)
                    timer.stop_timer();
                } //if

            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);
        }//levels

    }//convolve1x1_batchparallel


    //*****************************************************************************************************************
    //                              Temporary isotropic patch reconstruction
    //*****************************************************************************************************************

    template<typename ImageType>
    void update_dense_array_ptr(const uint64_t level,
                                const uint64_t z,
                                APR<ImageType> &apr,
                                APRIterator &apr_iterator,
                                APRTreeIterator &treeIterator,
                                ExtraParticleData<float> &tree_data,
                                PixelData<float> &temp_vec,
                                float * part_int,
                                const std::vector<int> &stencil_shape,
                                const std::vector<int> &stencil_half,
                                uint64_t in_offset) {

        //py::buffer_info particleData = particle_intensities.request(); // pybind11::buffer_info to access data
        //auto part_int = (float *) particleData.ptr;

        uint64_t x;

        const uint64_t x_num_m = temp_vec.x_num;
        const uint64_t y_num_m = temp_vec.y_num;

        for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

            //
            //  This loop recreates particles at the current level, using a simple copy
            //

            uint64_t mesh_offset = (x + stencil_half[1]) * y_num_m + x_num_m * y_num_m * (z % stencil_shape[2]);


            //std::cout << "stencil_shape = {" << stencil_shape[0] << ", " << stencil_shape[1] << ", " << stencil_shape[2] << "}" << std::endl;

            for (apr_iterator.set_new_lzx(level, z, x);
                 apr_iterator.global_index() < apr_iterator.end_index;
                 apr_iterator.set_iterator_to_particle_next_particle()) {

                temp_vec.mesh[apr_iterator.y() + stencil_half[0] + mesh_offset] = part_int[apr_iterator.global_index() + in_offset];//particleData.data[apr_iterator];
            }
        }

        if (level > apr_iterator.level_min()) {
            const int y_num = apr_iterator.spatial_index_y_max(level);

            //
            //  This loop interpolates particles at a lower level (Larger Particle Cell or resolution), by simple uploading
            //

            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {

                for (apr_iterator.set_new_lzx(level - 1, z / 2, x / 2);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    int y_m = std::min(2 * apr_iterator.y() + 1, y_num - 1);    // 2y+1+offset

                    temp_vec.at(2 * apr_iterator.y() + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = part_int[apr_iterator.global_index() + in_offset];//particleData[apr_iterator];
                    temp_vec.at(y_m + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = part_int[apr_iterator.global_index() + in_offset];//particleData[apr_iterator];

                }
            }
        }

        /******** start of using the tree iterator for downsampling ************/

        if (level < apr_iterator.level_max()) {

            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                for (treeIterator.set_new_lzx(level, z, x);
                     treeIterator.global_index() < treeIterator.end_index;
                     treeIterator.set_iterator_to_particle_next_particle()) {

                    temp_vec.at(treeIterator.y() + stencil_half[0], x + stencil_half[1],
                                z % stencil_shape[2]) = tree_data[treeIterator];
                }
            }
        }
    }


    //*****************************************************************************************************************
    //                              Backward convolution arbitrary filter size
    //*****************************************************************************************************************

    template<typename ImageType>
    void convolve_batchparallel_backward(APR<ImageType> &apr, APRIterator &apr_iterator, APRTreeIterator &tree_iterator,
                                         float * input_ptr, const uint64_t in_offset, const uint64_t out_offset,
                                         const std::vector<PixelData<float>>& stencil_vec,
                                         float * grad_output_ptr, float * grad_input_ptr, std::vector<float> &temp_vec_db,
                                         std::vector<float> &temp_vec_dw, const uint64_t grad_weight_offset, const size_t db_offset,
                                         ExtraParticleData<float> &grad_tree_temp, ExtraParticleData<float> &tree_data,
                                         const int out_channel, const int in_channel, const unsigned int current_max_level) {

        int stencil_counter = 0;
        float d_bias = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

            //PixelData<float> stencil(stencil_vec[stencil_counter], true);

            const std::vector<int> stencil_shape = {(int) stencil_vec[stencil_counter].y_num,
                                                    (int) stencil_vec[stencil_counter].x_num,
                                                    (int) stencil_vec[stencil_counter].z_num};

            const std::vector<int> stencil_half = {(stencil_shape[0]-1)/2,
                                                   (stencil_shape[1]-1)/2,
                                                   (stencil_shape[2]-1)/2};

            const uint64_t dw_offset = grad_weight_offset + stencil_counter * stencil_shape[0] * stencil_shape[1];

            // assert stencil_shape compatible with apr org_dims?

            unsigned int z = 0;
            unsigned int x = 0;

            const int z_num = 1; //apr_iterator.spatial_index_z_max(level);

            const int y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_iterator.spatial_index_y_max(level) +
                                                                   stencil_shape[0] - 1 : 1;
            const int x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_iterator.spatial_index_x_max(level) +
                                                                   stencil_shape[1] - 1 : 1;


            PixelData<float> temp_vec;
            temp_vec.init(y_num_m, x_num_m, stencil_shape[2], 0); //zero padded boundaries

            PixelData<float> temp_vec_di;
            temp_vec_di.init(y_num_m, x_num_m, stencil_shape[2], 0);

            /*
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array2(level,
                                    padd,
                                    apr,
                                    apr_iterator,
                                    tree_iterator,
                                    tree_data,
                                    temp_vec,
                                    input_intensities,
                                    stencil_shape,
                                    stencil_half,
                                    in_offset);
            }
            */

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - stencil_half[2])) {
                    //update the next z plane for the access
                    update_dense_array_ptr(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                           temp_vec, input_ptr, stencil_shape, stencil_half, in_offset);
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);

                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                }

                for(x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        const float dO = grad_output_ptr[out_offset + apr_iterator.global_index()];
                        d_bias += dO;

                        for(size_t ix=0; ix < stencil_shape[1]; ++ix) {
                            for(size_t iy=0; iy < stencil_shape[0]; ++iy) {
                                temp_vec_di.at(apr_iterator.y() + iy, x + ix, 0) += dO * stencil_vec[stencil_counter].mesh[ix*stencil_shape[0] + iy];
                            }
                        }

                        for(size_t ix=0; ix < stencil_shape[1]; ++ix) {
                            for(size_t iy=0; iy < stencil_shape[0]; ++iy) {
                                temp_vec_dw[dw_offset + ix*stencil_shape[0] + iy] += dO * temp_vec.at(apr_iterator.y() + iy, x + ix, 0);
                            }
                        }

                    } // y pixels/columns (apr)
                } // x

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    int64_t tree_offset = compute_tree_offset(apr, level, false);

                    for(x = 0; x<tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            //const int k = tree_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                            //const int i = x + stencil_half[1];

                            float dO = grad_output_ptr[out_offset + tree_iterator.global_index() + tree_offset];
                            d_bias += dO;

                            for(size_t ix=0; ix < stencil_shape[1]; ++ix) {
                                for(size_t iy=0; iy < stencil_shape[0]; ++iy) {
                                    temp_vec_di.at(tree_iterator.y() + iy, x + ix, 0) += dO * stencil_vec[stencil_counter].mesh[ix*stencil_shape[0] + iy];
                                }
                            }

                            for(size_t ix=0; ix < stencil_shape[1]; ++ix) {
                                for(size_t iy=0; iy < stencil_shape[0]; ++iy) {
                                    temp_vec_dw[dw_offset + ix*stencil_shape[0] + iy] += dO * temp_vec.at(tree_iterator.y() + iy, x + ix, 0);
                                }
                            }

                        }//y, pixels/columns (tree)
                    }//x
                } //if

                //TODO: this works for 2D images, but for 3D the updating needs to change
                /// push temp_vec_di to grad_input and grad_tree_temp
                update_dense_array_ptr_backward(level, z, apr, apr_iterator, tree_iterator, grad_tree_temp, temp_vec_di,
                                                grad_input_ptr, stencil_shape, stencil_half, in_offset);

            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels

        /// push d_bias to temp_vec_db
        temp_vec_db[out_channel + db_offset] += d_bias;
    }


    //*****************************************************************************************************************
    //                              Backward convolution 3x3 filter
    //*****************************************************************************************************************

    template<typename ImageType>
    void convolve3x3_batchparallel_backward(APR<ImageType> &apr, APRIterator &apr_iterator, APRTreeIterator &tree_iterator,
                                            float * input_ptr, const uint64_t in_offset, const uint64_t out_offset,
                                            const std::vector<PixelData<float>>& stencil_vec,
                                            float * grad_output_ptr, float * grad_input_ptr, std::vector<float> &temp_vec_db,
                                            std::vector<float> &temp_vec_dw, const uint64_t grad_weight_offset, const size_t db_offset,
                                            ExtraParticleData<float> &grad_tree_temp, ExtraParticleData<float> &tree_data,
                                            const int out_channel, const int in_channel, const unsigned int current_max_level) {

        int stencil_counter = 0;
        float d_bias = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

            //PixelData<float> stencil(stencil_vec[stencil_counter], true);

            const uint64_t dw_offset = grad_weight_offset + stencil_counter * 9;

            const std::vector<int> stencil_shape = {3, 3, 1};
            const std::vector<int> stencil_half = {1, 1, 0};

            // assert stencil_shape compatible with apr org_dims?

            unsigned int z = 0;
            unsigned int x = 0;

            const int z_num = 1; //apr_iterator.spatial_index_z_max(level);

            const int y_num_m = (apr.apr_access.org_dims[0] > 1) ? apr_iterator.spatial_index_y_max(level) +
                                                                   stencil_shape[0] - 1 : 1;
            const int x_num_m = (apr.apr_access.org_dims[1] > 1) ? apr_iterator.spatial_index_x_max(level) +
                                                                   stencil_shape[1] - 1 : 1;


            PixelData<float> temp_vec;
            temp_vec.init(y_num_m, x_num_m, stencil_shape[2], 0); //zero padded boundaries

            PixelData<float> temp_vec_di;
            temp_vec_di.init(y_num_m, x_num_m, stencil_shape[2], 0);

            //initial condition
            /*
            for (int padd = 0; padd < stencil_half[2]; ++padd) {
                update_dense_array2(level,
                                    padd,
                                    apr,
                                    apr_iterator,
                                    tree_iterator,
                                    tree_data,
                                    temp_vec,
                                    input_intensities,
                                    stencil_shape,
                                    stencil_half,
                                    in_offset);
            }
            */

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                if (z < (z_num - stencil_half[2])) {
                    //update the next z plane for the access
                    update_dense_array_ptr(level, z + stencil_half[2], apr, apr_iterator, tree_iterator, tree_data,
                                           temp_vec, input_ptr, stencil_shape, stencil_half, in_offset);
                } else {
                    //padding
                    uint64_t index = temp_vec.x_num * temp_vec.y_num * ((z + stencil_half[2]) % stencil_shape[2]);

                    for (x = 0; x < temp_vec.x_num; ++x) {
                        std::fill(temp_vec.mesh.begin() + index + (x + 0) * temp_vec.y_num,
                                  temp_vec.mesh.begin() + index + (x + 1) * temp_vec.y_num, 0);
                    }
                }

                //std::string fileName = "/Users/joeljonsson/Documents/STUFF/temp_vec_bw_lvl" + std::to_string(level) + ".tif";
                //TiffUtils::saveMeshAsTiff(fileName, temp_vec);

                for(x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        const float dO = grad_output_ptr[out_offset + apr_iterator.global_index()];
                        d_bias += dO;

                        const size_t y_num = temp_vec.y_num;
                        const uint64_t idx0 = x * y_num + apr_iterator.y();
                        const uint64_t idx1 = idx0 + y_num;
                        const uint64_t idx2 = idx1 + y_num;

                        temp_vec_di.mesh[idx0]   += dO * stencil_vec[stencil_counter].mesh[0];
                        temp_vec_di.mesh[idx0+1] += dO * stencil_vec[stencil_counter].mesh[1];
                        temp_vec_di.mesh[idx0+2] += dO * stencil_vec[stencil_counter].mesh[2];

                        temp_vec_di.mesh[idx1]   += dO * stencil_vec[stencil_counter].mesh[3];
                        temp_vec_di.mesh[idx1+1] += dO * stencil_vec[stencil_counter].mesh[4];
                        temp_vec_di.mesh[idx1+2] += dO * stencil_vec[stencil_counter].mesh[5];

                        temp_vec_di.mesh[idx2]   += dO * stencil_vec[stencil_counter].mesh[6];
                        temp_vec_di.mesh[idx2+1] += dO * stencil_vec[stencil_counter].mesh[7];
                        temp_vec_di.mesh[idx2+2] += dO * stencil_vec[stencil_counter].mesh[8];


                        temp_vec_dw[dw_offset]   += dO * temp_vec.mesh[idx0];
                        temp_vec_dw[dw_offset+1] += dO * temp_vec.mesh[idx0+1];
                        temp_vec_dw[dw_offset+2] += dO * temp_vec.mesh[idx0+2];

                        temp_vec_dw[dw_offset+3] += dO * temp_vec.mesh[idx1];
                        temp_vec_dw[dw_offset+4] += dO * temp_vec.mesh[idx1+1];
                        temp_vec_dw[dw_offset+5] += dO * temp_vec.mesh[idx1+2];

                        temp_vec_dw[dw_offset+6] += dO * temp_vec.mesh[idx2];
                        temp_vec_dw[dw_offset+7] += dO * temp_vec.mesh[idx2+1];
                        temp_vec_dw[dw_offset+8] += dO * temp_vec.mesh[idx2+2];

                    }
                }

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    int64_t tree_offset = compute_tree_offset(apr, level, false);

                    for(x = 0; x<tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            //const int k = tree_iterator.y() + stencil_half[0]; // offset to allow for boundary padding
                            //const int i = x + stencil_half[1];

                            float dO = grad_output_ptr[out_offset + tree_iterator.global_index() + tree_offset];
                            d_bias += dO;

                            const size_t y_num = temp_vec.y_num;
                            const uint64_t idx0 = x * y_num + tree_iterator.y();
                            const uint64_t idx1 = idx0 + y_num;
                            const uint64_t idx2 = idx1 + y_num;

                            temp_vec_di.mesh[idx0]   += dO * stencil_vec[stencil_counter].mesh[0];
                            temp_vec_di.mesh[idx0+1] += dO * stencil_vec[stencil_counter].mesh[1];
                            temp_vec_di.mesh[idx0+2] += dO * stencil_vec[stencil_counter].mesh[2];

                            temp_vec_di.mesh[idx1]   += dO * stencil_vec[stencil_counter].mesh[3];
                            temp_vec_di.mesh[idx1+1] += dO * stencil_vec[stencil_counter].mesh[4];
                            temp_vec_di.mesh[idx1+2] += dO * stencil_vec[stencil_counter].mesh[5];

                            temp_vec_di.mesh[idx2]   += dO * stencil_vec[stencil_counter].mesh[6];
                            temp_vec_di.mesh[idx2+1] += dO * stencil_vec[stencil_counter].mesh[7];
                            temp_vec_di.mesh[idx2+2] += dO * stencil_vec[stencil_counter].mesh[8];


                            temp_vec_dw[dw_offset]   += dO * temp_vec.mesh[idx0];
                            temp_vec_dw[dw_offset+1] += dO * temp_vec.mesh[idx0+1];
                            temp_vec_dw[dw_offset+2] += dO * temp_vec.mesh[idx0+2];

                            temp_vec_dw[dw_offset+3] += dO * temp_vec.mesh[idx1];
                            temp_vec_dw[dw_offset+4] += dO * temp_vec.mesh[idx1+1];
                            temp_vec_dw[dw_offset+5] += dO * temp_vec.mesh[idx1+2];

                            temp_vec_dw[dw_offset+6] += dO * temp_vec.mesh[idx2];
                            temp_vec_dw[dw_offset+7] += dO * temp_vec.mesh[idx2+1];
                            temp_vec_dw[dw_offset+8] += dO * temp_vec.mesh[idx2+2];
                        }//y, pixels/columns (tree)
                    }//x
                } //if

                //TODO: this works for 2D images, but for 3D the updating needs to change
                /// push temp_vec_di to grad_input and grad_tree_temp
                update_dense_array_ptr_backward(level, z, apr, apr_iterator, tree_iterator, grad_tree_temp, temp_vec_di,
                                                grad_input_ptr, stencil_shape, stencil_half, in_offset);

            }//z

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels

        /// push d_bias to temp_vec_db
        temp_vec_db[out_channel + db_offset] += d_bias;
    }


    //*****************************************************************************************************************
    //                              Backward convolution 1x1 filter
    //*****************************************************************************************************************

    template<typename ImageType>
    void convolve1x1_batchparallel_backward(APR<ImageType> &apr, APRIterator &apr_iterator, APRTreeIterator &tree_iterator,
                                            float * input_ptr, const uint64_t in_offset, const uint64_t out_offset,
                                            const std::vector<float>& stencil_vec, float * grad_output_ptr, float * grad_input_ptr,
                                            std::vector<float> &temp_vec_db, std::vector<float> &temp_vec_dw,
                                            const uint64_t grad_weight_offset, const size_t db_offset, const int out_channel,
                                            const int in_channel, const unsigned int current_max_level) {

        float d_bias = 0;
        int stencil_counter = 0;

        for (int level = current_max_level; level >= apr_iterator.level_min(); --level) {

            float d_weight = 0;
            const float w = stencil_vec[stencil_counter];

            unsigned int z = 0;
            unsigned int x = 0;

            for (z = 0; z < apr.spatial_index_z_max(level); ++z) {

                for(x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        const float dO = grad_output_ptr[out_offset + apr_iterator.global_index()];

                        d_bias += dO;
                        d_weight += dO * input_ptr[in_offset + apr_iterator.global_index()];
                        grad_input_ptr[in_offset + apr_iterator.global_index()] += dO * w;

                    }//y, pixels/columns
                } //x

                /// if there are downsampled values, we need to use the tree iterator for those outputs
                if(level == current_max_level && current_max_level < apr.level_max()) {

                    const int64_t tree_offset = compute_tree_offset(apr, level, false);

                    for(x = 0; x < tree_iterator.spatial_index_x_max(level); ++x) {
                        for (tree_iterator.set_new_lzx(level, z, x);
                             tree_iterator.global_index() < tree_iterator.end_index;
                             tree_iterator.set_iterator_to_particle_next_particle()) {

                            const uint64_t idx = tree_iterator.global_index() + tree_offset;
                            const float dO = grad_output_ptr[out_offset + idx];

                            d_bias += dO;
                            d_weight += dO * input_ptr[in_offset + idx];
                            grad_input_ptr[in_offset + idx] += dO * w;

                        }//y, pixels/columns (tree)
                    } //x
                } //if
            }//z

            temp_vec_dw[grad_weight_offset + stencil_counter] += d_weight;

            // Use the next stencil (if available). The last supplied stencil will be used for all remaining levels.
            stencil_counter = std::min(stencil_counter + 1, (int) stencil_vec.size() - 1);

        }//levels

        temp_vec_db[db_offset + out_channel] += d_bias;

    }


    //*****************************************************************************************************************
    //                              Backward temporary isotropic patch reconstruction
    //*****************************************************************************************************************

    template<typename ImageType>
    void update_dense_array_ptr_backward(const uint64_t level,
                                         const uint64_t z,
                                         APR<ImageType> &apr,
                                         APRIterator &apr_iterator,
                                         APRTreeIterator &treeIterator,
                                         ExtraParticleData<float> &grad_tree_data,
                                         PixelData<float> &temp_vec_di,
                                         float * grad_input_ptr,
                                         const std::vector<int> &stencil_shape,
                                         const std::vector<int> &stencil_half,
                                         const uint64_t in_offset) {


        size_t x;

        const size_t x_num_m = temp_vec_di.x_num;
        const size_t y_num_m = temp_vec_di.y_num;


        /// insert values at matching apr particle locations

        for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {

            uint64_t mesh_offset = x_num_m * y_num_m * (z % stencil_shape[2]) + (x + stencil_half[1]) * y_num_m + stencil_half[0];

            for (apr_iterator.set_new_lzx(level, z, x);
                 apr_iterator.global_index() < apr_iterator.end_index;
                 apr_iterator.set_iterator_to_particle_next_particle()) {

                //temp_vec.mesh[apr_iterator.y() + stencil_half[0] + mesh_offset] = part_int[apr_iterator.global_index() + in_offset];//particleData.data[apr_iterator];

                grad_input_ptr[in_offset + apr_iterator.global_index()] += temp_vec_di.mesh[apr_iterator.y() + mesh_offset];
            }
        }

        if (level > apr_iterator.level_min()) {
            const int y_num = apr_iterator.spatial_index_y_max(level);

            /// insert values at lower level particles

            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {

                for (apr_iterator.set_new_lzx(level - 1, z / 2, x / 2);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    int y_m = std::min(2 * apr_iterator.y() + 1, y_num - 1);    // 2y+1+offset

                    //temp_vec.at(2 * apr_iterator.y() + stencil_half[0], x + stencil_half[1],
                    //            z % stencil_shape[2]) = part_int[apr_iterator.global_index() + in_offset];//particleData[apr_iterator];
                    //temp_vec.at(y_m + stencil_half[0], x + stencil_half[1],
                    //            z % stencil_shape[2]) = part_int[apr_iterator.global_index() + in_offset];//particleData[apr_iterator];

                    grad_input_ptr[in_offset + apr_iterator.global_index()] +=
                            temp_vec_di.at(2 * apr_iterator.y() + stencil_half[0], x + stencil_half[1], z % stencil_shape[2]) +
                            temp_vec_di.at(y_m + stencil_half[0], x + stencil_half[1], z % stencil_shape[2]);

                }
            }
        }

        /******** start of using the tree iterator for downsampling ************/

        if (level < apr_iterator.level_max()) {

            for (x = 0; x < apr.spatial_index_x_max(level); ++x) {
                for (treeIterator.set_new_lzx(level, z, x);
                     treeIterator.global_index() < treeIterator.end_index;
                     treeIterator.set_iterator_to_particle_next_particle()) {

                    grad_tree_data[treeIterator] += temp_vec_di.at(treeIterator.y() + stencil_half[0], x + stencil_half[1], z % stencil_shape[2]);

                }
            }
        }
    }


    //*****************************************************************************************************************
    //                              Fill APRTree by average-downsampling APR particles
    //*****************************************************************************************************************

    template<typename T,typename U>
    void fill_tree_mean_py_ptr(APR<T>& apr, APRTree<T>& apr_tree, float * input_ptr, ExtraParticleData<U>& tree_data,
                               APRIterator &apr_iterator, APRTreeIterator &tree_iterator, uint64_t in_offset,
                               unsigned int current_max_level) {

        APRTimer timer(false);

        timer.start_timer("ds-init");

        if( current_max_level < apr.level_max() ) {

            tree_data.init(apr_tree.tree_access.global_index_by_level_and_zx_end[current_max_level].back());

        } else {

            tree_data.init(apr_tree.total_number_parent_cells());
        }

        //auto tree_iterator = apr_tree.tree_iterator();
        auto parent_iterator = apr_tree.tree_iterator();
        //auto apr_iterator = apr.iterator();

        //int z_d;
        int x_d;
        timer.stop_timer();
        timer.start_timer("ds-1l");

        //py::buffer_info input_buf = particle_data.request();
        //auto input_ptr = (float *) input_buf.ptr;

        /// if downsampling has been performed, insert the downsampled values directly
        if(current_max_level < apr.level_max()) {

            int z=0;
            int x;

            const int64_t tree_offset_in  = compute_tree_offset(apr, current_max_level, false);

            //for (z = 0; z < treeIterator.spatial_index_z_max(current_max_level); z++) {
            for (x = 0; x < tree_iterator.spatial_index_x_max(current_max_level); ++x) {
                for (tree_iterator.set_new_lzx(current_max_level, z, x);
                     tree_iterator.global_index() < tree_iterator.end_index;
                     tree_iterator.set_iterator_to_particle_next_particle()) {

                    tree_data[tree_iterator] = input_ptr[in_offset + tree_iterator.global_index() + tree_offset_in];

                }
            }//x
            //}//z
        }//if

        /// now fill in parent nodes of APR particles
        for (unsigned int level = current_max_level; level >= apr_iterator.level_min(); --level) {
            //z_d = 0;
            int z = 0;

            //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(level)-1); ++z) {
            //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
            for (x_d = 0; x_d < parent_iterator.spatial_index_x_max(level-1); ++x_d) {
                for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level)-1); ++x) {

                    parent_iterator.set_new_lzx(level - 1, z / 2, x / 2);

                    //dealing with boundary conditions
                    float scale_factor_xz =
                            (((2 * parent_iterator.spatial_index_x_max(level - 1) != apr.spatial_index_x_max(level)) &&
                              ((x / 2) == (parent_iterator.spatial_index_x_max(level - 1) - 1))) +
                             ((2 * parent_iterator.spatial_index_z_max(level - 1) != apr.spatial_index_z_max(level)) &&
                              (z / 2) == (parent_iterator.spatial_index_z_max(level - 1) - 1))) * 2;

                    if (scale_factor_xz == 0) {
                        scale_factor_xz = 1;
                    }

                    float scale_factor_yxz = scale_factor_xz;

                    if ((2 * parent_iterator.spatial_index_y_max(level - 1) != apr.spatial_index_y_max(level))) {
                        scale_factor_yxz = scale_factor_xz * 2;
                    }


                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() <
                         apr_iterator.end_index; apr_iterator.set_iterator_to_particle_next_particle()) {

                        while (parent_iterator.y() != (apr_iterator.y() / 2)) {
                            parent_iterator.set_iterator_to_particle_next_particle();
                        }


                        if (parent_iterator.y() == (parent_iterator.spatial_index_y_max(level - 1) - 1)) {
                            tree_data[parent_iterator] +=
                                    scale_factor_yxz * input_ptr[in_offset + apr_iterator.global_index()] / 8.0f;
                        } else {

                            tree_data[parent_iterator] +=
                                    scale_factor_xz * input_ptr[in_offset + apr_iterator.global_index()] / 8.0f;
                        }

                    }
                }
            }
            //}
            //}
        }

        timer.stop_timer();
        timer.start_timer("ds-2l");

        ///then do the rest of the tree where order matters
        for (unsigned int level = std::min(current_max_level, (unsigned int)tree_iterator.level_max()); level > tree_iterator.level_min(); --level) {
            //z_d = 0;
            int z = 0;

            //for (z_d = 0; z_d < treeIterator.spatial_index_z_max(level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(level)-1); ++z) {
            //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
            for (x_d = 0; x_d < tree_iterator.spatial_index_x_max(level-1); ++x_d) {
                for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) tree_iterator.spatial_index_x_max(level)-1); ++x) {

                    parent_iterator.set_new_lzx(level - 1, z/2, x/2);

                    float scale_factor_xz =
                            (((2 * parent_iterator.spatial_index_x_max(level - 1) != parent_iterator.spatial_index_x_max(level)) &&
                              ((x / 2) == (parent_iterator.spatial_index_x_max(level - 1) - 1))) +
                             ((2 * parent_iterator.spatial_index_z_max(level - 1) != parent_iterator.spatial_index_z_max(level)) &&
                              ((z / 2) == (parent_iterator.spatial_index_z_max(level - 1) - 1)))) * 2;

                    if (scale_factor_xz == 0) {
                        scale_factor_xz = 1;
                    }

                    float scale_factor_yxz = scale_factor_xz;

                    if ((2 * parent_iterator.spatial_index_y_max(level - 1) != parent_iterator.spatial_index_y_max(level))) {
                        scale_factor_yxz = scale_factor_xz * 2;
                    }

                    for (tree_iterator.set_new_lzx(level, z, x);
                         tree_iterator.global_index() < tree_iterator.end_index;
                         tree_iterator.set_iterator_to_particle_next_particle()) {

                        while (parent_iterator.y() != tree_iterator.y() / 2) {
                            parent_iterator.set_iterator_to_particle_next_particle();
                        }

                        if (parent_iterator.y() == (parent_iterator.spatial_index_y_max(level - 1) - 1)) {
                            tree_data[parent_iterator] += scale_factor_yxz * tree_data[tree_iterator] / 8.0f;
                        } else {
                            tree_data[parent_iterator] += scale_factor_xz * tree_data[tree_iterator] / 8.0f;
                        }

                    }
                }
            }
            //}
            //}
        }
        timer.stop_timer();
    }


    //*****************************************************************************************************************
    //                              Backward fill APRTree (push APRTree gradient values to APR particles)
    //*****************************************************************************************************************

    template<typename T,typename U>
    void fill_tree_mean_py_backward_ptr(APR<T>& apr, APRTree<T>& apr_tree, APRIterator &apr_iterator, APRTreeIterator &tree_iterator,
                                        float * grad_input_ptr, ExtraParticleData<U> &grad_tree_temp, uint64_t in_offset,
                                        unsigned int current_max_level) {

        APRTimer timer(false);
        APRTimer t2(false);

        auto parent_iterator = apr_tree.tree_iterator();

        //int z_d;
        int x_d;
        timer.start_timer("fill_tree_mean_backwards");

        /// go through the tree from top (low level) to bottom (high level) and push values downwards
        for (unsigned int level = tree_iterator.level_min()+1;
             level <= std::min(current_max_level, (unsigned int)tree_iterator.level_max());
             ++level) {

            //z_d = 0;
            int z = 0;

            //for (z_d = 0; z_d < treeIterator.spatial_index_z_max(level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)treeIterator.spatial_index_z_max(level)-1); ++z) {
            //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
            t2.start_timer("fill tree backward, first loop");

            for (x_d = 0; x_d < tree_iterator.spatial_index_x_max(level-1); ++x_d) {
                for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) tree_iterator.spatial_index_x_max(level)-1); ++x) {

                    parent_iterator.set_new_lzx(level - 1, z/2, x/2);

                    float scale_factor_xz =
                            (((2 * parent_iterator.spatial_index_x_max(level - 1) != parent_iterator.spatial_index_x_max(level)) &&
                              ((x / 2) == (parent_iterator.spatial_index_x_max(level - 1) - 1))) +
                             ((2 * parent_iterator.spatial_index_z_max(level - 1) != parent_iterator.spatial_index_z_max(level)) &&
                              ((z / 2) == (parent_iterator.spatial_index_z_max(level - 1) - 1)))) * 2;

                    if (scale_factor_xz == 0) {
                        scale_factor_xz = 1;
                    }

                    float scale_factor_yxz = scale_factor_xz;

                    if ((2 * parent_iterator.spatial_index_y_max(level - 1) != parent_iterator.spatial_index_y_max(level))) {
                        scale_factor_yxz = scale_factor_xz * 2;
                    }

                    for (tree_iterator.set_new_lzx(level, z, x);
                         tree_iterator.global_index() < tree_iterator.end_index;
                         tree_iterator.set_iterator_to_particle_next_particle()) {

                        while (parent_iterator.y() != tree_iterator.y() / 2) {
                            parent_iterator.set_iterator_to_particle_next_particle();
                        }

                        if (parent_iterator.y() == (parent_iterator.spatial_index_y_max(level - 1) - 1)) {
                            //tree_data[parentIterator] = scale_factor_yxz * tree_data[treeIterator] / 8.0f +
                            //                            tree_data[parentIterator];
                            grad_tree_temp[tree_iterator] += grad_tree_temp[parent_iterator] * scale_factor_yxz / 8.0f;

                        } else {
                            //tree_data[parentIterator] = scale_factor_xz * tree_data[treeIterator] / 8.0f +
                            //                            tree_data[parentIterator];
                            grad_tree_temp[tree_iterator] += grad_tree_temp[parent_iterator] * scale_factor_xz / 8.0f;
                        }

                    }
                }
            }
            t2.stop_timer();
            //}
            //}
        }

        /// go through apr particles and transfer values from grad_tree_temp to grad_input
        for (unsigned int level = current_max_level; level >= apr_iterator.level_min(); --level) {
            //int z_d = 0;
            int z = 0;

            //for (z_d = 0; z_d < parentIterator.spatial_index_z_max(level-1); z_d++) {
            //for (int z = 2*z_d; z <= std::min(2*z_d+1,(int)apr.spatial_index_z_max(level)-1); ++z) {
            //the loop is bundled into blocks of 2, this prevents race conditions with OpenMP parents
            t2.start_timer("fill tree backward, second loop");

            for (x_d = 0; x_d < parent_iterator.spatial_index_x_max(level-1); ++x_d) {
                for (int x = 2 * x_d; x <= std::min(2 * x_d + 1, (int) apr.spatial_index_x_max(level)-1); ++x) {

                    parent_iterator.set_new_lzx(level - 1, z / 2, x / 2);

                    //dealing with boundary conditions
                    float scale_factor_xz =
                            (((2 * parent_iterator.spatial_index_x_max(level - 1) != apr.spatial_index_x_max(level)) &&
                              ((x / 2) == (parent_iterator.spatial_index_x_max(level - 1) - 1))) +
                             ((2 * parent_iterator.spatial_index_z_max(level - 1) != apr.spatial_index_z_max(level)) &&
                              (z / 2) == (parent_iterator.spatial_index_z_max(level - 1) - 1))) * 2;

                    if (scale_factor_xz == 0) {
                        scale_factor_xz = 1;
                    }

                    float scale_factor_yxz = scale_factor_xz;

                    if ((2 * parent_iterator.spatial_index_y_max(level - 1) != apr.spatial_index_y_max(level))) {
                        scale_factor_yxz = scale_factor_xz * 2;
                    }


                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {

                        while (parent_iterator.y() != (apr_iterator.y() / 2)) {
                            parent_iterator.set_iterator_to_particle_next_particle();
                        }


                        if (parent_iterator.y() == (parent_iterator.spatial_index_y_max(level - 1) - 1)) {
                            //tree_data[parentIterator] =
                            //        scale_factor_yxz * input_ptr[in_offset + apr_iterator.global_index()] / 8.0f +
                            //        tree_data[parentIterator];

                            grad_input_ptr[in_offset + apr_iterator.global_index()] +=
                                    scale_factor_yxz * grad_tree_temp[parent_iterator] / 8.0f;
                        } else {

                            //tree_data[parentIterator] =
                            //        scale_factor_xz * input_ptr[in_offset + apr_iterator.global_index()] / 8.0f +
                            //        tree_data[parentIterator];

                            grad_input_ptr[in_offset + apr_iterator.global_index()] +=
                                    scale_factor_xz * grad_tree_temp[parent_iterator] / 8.0f;
                        }

                    }
                }
            }
            t2.stop_timer();
            //}
            //}
        }

        /// if downsampling has been performed, the downsampled values have to be accessed via the tree iterator
        if(current_max_level < apr.level_max()) {

            int z=0;
            int x;

            int64_t tree_offset_in  = compute_tree_offset(apr, current_max_level, false);

            //for (z = 0; z < treeIterator.spatial_index_z_max(current_max_level); z++) {
            t2.start_timer("fill tree backward, third loop");

            for (x = 0; x < tree_iterator.spatial_index_x_max(current_max_level); ++x) {
                for (tree_iterator.set_new_lzx(current_max_level, z, x);
                     tree_iterator.global_index() < tree_iterator.end_index;
                     tree_iterator.set_iterator_to_particle_next_particle()) {

                    //tree_data[treeIterator] = input_ptr[in_offset + treeIterator.global_index() + tree_offset_in];

                    grad_input_ptr[in_offset + tree_iterator.global_index() + tree_offset_in] += grad_tree_temp[tree_iterator];

                }
            }//x
            t2.stop_timer();
            //}//z
        }//if
        timer.stop_timer();

    }


    //*****************************************************************************************************************
    //                              Forward 2x2 maxpooling
    //*****************************************************************************************************************

    template<typename ImageType>
    void max_pool_batchparallel(APR<ImageType> &apr, float * input_ptr, float * output_ptr, int64_t * idx_ptr,
                                const uint64_t in_offset, const uint64_t out_offset, const int64_t tree_offset_in,
                                const int64_t tree_offset_out, APRIterator &apr_iterator, APRTreeIterator &treeIterator,
                                APRTreeIterator &parentIterator, const unsigned int current_max_level) {

        /// Start by filling in the existing values up to and including current_max_level - 1, as they are unchanged
        for (unsigned int level = apr_iterator.level_min(); level < current_max_level; ++level) {
            int z = 0;
            int x = 0;

            for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                for (apr_iterator.set_new_lzx(level, z, x);
                     apr_iterator.global_index() < apr_iterator.end_index;
                     apr_iterator.set_iterator_to_particle_next_particle()) {

                    output_ptr[apr_iterator.global_index() + out_offset] = input_ptr[apr_iterator.global_index() + in_offset];
                    idx_ptr[apr_iterator.global_index() + out_offset] = apr_iterator.global_index() + in_offset;
                }
            }
        }

        /// Downsample the APR particles
        //int z_d = 0;
        int z = 0;
        unsigned int x;

        for(x = 0; x < apr_iterator.spatial_index_x_max(current_max_level); ++x) {

            parentIterator.set_new_lzx(current_max_level - 1, z / 2, x / 2);

            for (apr_iterator.set_new_lzx(current_max_level, z, x);
                 apr_iterator.global_index() < apr_iterator.end_index;
                 apr_iterator.set_iterator_to_particle_next_particle()) {

                while (parentIterator.y() != (apr_iterator.y() / 2)) {
                    parentIterator.set_iterator_to_particle_next_particle();
                }

                uint64_t out_idx = out_offset + parentIterator.global_index() + tree_offset_out;
                uint64_t in_idx = in_offset + apr_iterator.global_index();

                float curr = output_ptr[out_idx];
                float tmp = input_ptr[in_idx];

                if(tmp > curr) {
                    idx_ptr[out_idx] = in_idx;
                    output_ptr[out_idx] = tmp;
                }
            }
        }

        /// Downsample tree particles
        if( current_max_level < apr.level_max()) {
            //int z_d = 0;
            int z = 0;

            for(x = 0; x < treeIterator.spatial_index_x_max(current_max_level); ++x) {

                parentIterator.set_new_lzx(current_max_level - 1, z / 2, x / 2);

                for (treeIterator.set_new_lzx(current_max_level, z, x);
                     treeIterator.global_index() < treeIterator.end_index;
                     treeIterator.set_iterator_to_particle_next_particle()) {

                    while (parentIterator.y() != (treeIterator.y() / 2)) {
                        parentIterator.set_iterator_to_particle_next_particle();
                    }

                    uint64_t out_idx = out_offset + parentIterator.global_index() + tree_offset_out;
                    uint64_t in_idx = in_offset + treeIterator.global_index() + tree_offset_in;

                    float curr = output_ptr[out_idx];
                    float tmp = input_ptr[in_idx];

                    if(tmp > curr) {
                        idx_ptr[out_idx] = in_idx;
                        output_ptr[out_idx] = tmp;
                    }
                }
            }
        }
    }


    //*****************************************************************************************************************
    //                              Compute APRTree index offset
    //*****************************************************************************************************************

    template<typename ImageType>
    int64_t compute_tree_offset(APR<ImageType> &apr, unsigned int level, bool init_tree = false) {

        if(init_tree) { apr.apr_tree.init(apr); }

        int number_parts;
        int tree_start;

        if(level >= apr.level_min()) {
            number_parts = apr.apr_access.global_index_by_level_and_zx_end[level].back();
        } else {
            number_parts = 0;
        }

        if(level > apr.apr_tree.tree_access.level_min()) {
            tree_start = apr.apr_tree.tree_access.global_index_by_level_and_zx_end[level - 1].back();
        } else {
            tree_start = 0;
        }

        int64_t tree_offset = number_parts - tree_start;

        return tree_offset;
    }



    /**
    * Computes the required number of intensity values required to represent the image with a given maximum level.
    */
    template<typename ImageType>
    uint64_t number_parts_at_level(APR<ImageType> &apr, unsigned int max_level) {

        unsigned int number_parts;
        unsigned int tree_start;
        unsigned int tree_end;

        if(max_level >= apr.level_min()) {
            number_parts = apr.apr_access.global_index_by_level_and_zx_end[max_level].back();
        } else {
            number_parts = 0;
        }

        if(max_level > apr.apr_tree.tree_access.level_min()) {
            tree_start = apr.apr_tree.tree_access.global_index_by_level_and_zx_end[max_level - 1].back();
        } else {
            tree_start = 0;
        }

        if(max_level >= apr.apr_tree.tree_access.level_min()) {
            tree_end = apr.apr_tree.tree_access.global_index_by_level_and_zx_end[max_level].back();
        } else {
            tree_end = 0;
        }

        unsigned int number_graduated_parts = tree_end - tree_start;

        uint64_t number_parts_out = number_parts + number_graduated_parts;

        return number_parts_out;
    }




    /**
    * Compute a piecewise constant reconstruction using the provided py::array of particle intensities, that may
    * include downsampled values that have to be read using the apr tree
    */
    template<typename U,typename S>
    void interp_img_new_intensities(APR<S>& apr, PixelData<U>& img, py::array &intensities, int level_delta){

        //  Takes in an APR and a python array of intensities to create piece-wise constant image

        py::buffer_info buf = intensities.request();
        auto intptr = (float *) buf.ptr;


        //unsigned int current_max_level = find_max_level(apr, intensities, true);
        unsigned int current_max_level = std::max(apr.level_max()-level_delta, apr.level_min());

        img.init(apr.apr_access.y_num[current_max_level], apr.apr_access.x_num[current_max_level], apr.apr_access.z_num[current_max_level], 0);

        APRIterator apr_iterator = apr.iterator();

        //int max_dim = std::max(std::max(apr.apr_access.org_dims[1], apr.apr_access.org_dims[0]), apr.apr_access.org_dims[2]);
        //int max_level = ceil(std::log2(max_dim));

        /// fill in values from the APR
        for (unsigned int level = apr_iterator.level_min(); level <= current_max_level; ++level) {
            int z = 0;
            int x = 0;

            const float step_size = pow(2, current_max_level - level);

#ifdef PYAPR_HAVE_OPENMP
            const bool parallel_z = apr_iterator.spatial_index_z_max(level) > 1;
            const bool parallel_x = !parallel_z && apr_iterator.spatial_index_x_max(level) > 1;
#endif

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator) if(parallel_z)
#endif
            for (z = 0; z < apr_iterator.spatial_index_z_max(level); z++) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(apr_iterator) if(parallel_x)
#endif
                for (x = 0; x < apr_iterator.spatial_index_x_max(level); ++x) {
                    for (apr_iterator.set_new_lzx(level, z, x);
                         apr_iterator.global_index() < apr_iterator.end_index;
                         apr_iterator.set_iterator_to_particle_next_particle()) {
                        //
                        //  Parallel loop over level
                        //

                        int dim1 = apr_iterator.y() * step_size;
                        int dim2 = apr_iterator.x() * step_size;
                        int dim3 = apr_iterator.z() * step_size;

                        float temp_int;
                        //add to all the required rays

                        temp_int = intptr[apr_iterator.global_index()];

                        const int offset_max_dim1 = std::min((int) img.y_num, (int) (dim1 + step_size));
                        const int offset_max_dim2 = std::min((int) img.x_num, (int) (dim2 + step_size));
                        const int offset_max_dim3 = std::min((int) img.z_num, (int) (dim3 + step_size));

                        for (int64_t q = dim3; q < offset_max_dim3; ++q) {
                            for (int64_t k = dim2; k < offset_max_dim2; ++k) {
                                for (int64_t i = dim1; i < offset_max_dim1; ++i) {
                                    img.mesh[i + (k) * img.y_num + q * img.y_num * img.x_num] = temp_int;
                                }
                            }
                        }
                    }
                }
            }
        }

        /// fill in eventual downsampled values using the tree iterator
        if(current_max_level < apr.level_max()) {

            APRTreeIterator tree_iterator = apr.apr_tree.tree_iterator();

            int64_t tree_offset = compute_tree_offset(apr, current_max_level, false);

            int z = 0;
            int x = 0;

#ifdef PYAPR_HAVE_OPENMP
            const bool parallel_z = tree_iterator.spatial_index_z_max(current_max_level) > 1000;
            const bool parallel_x = !parallel_z && tree_iterator.spatial_index_x_max(current_max_level) > 1000;
#endif

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(tree_iterator) if(parallel_z)
#endif
            for (z = 0; z < tree_iterator.spatial_index_z_max(current_max_level); z++) {
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(dynamic) private(z, x) firstprivate(tree_iterator) if(parallel_x)
#endif
                for (x = 0; x < tree_iterator.spatial_index_x_max(current_max_level); ++x) {
                    for (tree_iterator.set_new_lzx(current_max_level, z, x);
                         tree_iterator.global_index() < tree_iterator.end_index;
                         tree_iterator.set_iterator_to_particle_next_particle()) {

                        int dim1 = tree_iterator.y();
                        int dim2 = tree_iterator.x();
                        int dim3 = tree_iterator.z();

                        img.mesh[dim3*img.x_num*img.y_num + dim2*img.y_num + dim1] = intptr[tree_offset + tree_iterator.global_index()];

                    }
                }
            }
        }
    }

};

#endif //PYLIBAPR_PYAPRFILTERING_HPP
