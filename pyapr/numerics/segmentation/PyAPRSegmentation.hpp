#include "data_structures/APR/APR.hpp"
#include "data_containers/PyAPR.hpp"
#include "data_containers/PyParticleData.hpp"
#include "maxflow-v3.04.src/graph.h"
#include <math.h>
#include <stdlib.h>
#include <algorithm>

namespace py = pybind11;


template<typename T, typename V>
void find_min_max_val(ParticleData<T>& parts, V& vmin, V& vmax) {

    // find minimum and maximum values
    T min_val = std::numeric_limits<T>::max();
    T max_val = std::numeric_limits<T>::min();

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for reduction(min: min_val) reduction(max: max_val)
#endif
    for(size_t idx = 0; idx < parts.data.size(); ++idx) {
        min_val = std::min(min_val, parts[idx]);
        max_val = std::max(max_val, parts[idx]);
    }

    vmin = min_val;
    vmax = max_val;
}

template<typename T>
void segment_apr_py(PyAPR& apr, PyParticleData<T>& input_parts, PyParticleData<uint16_t>& mask_parts,
                    float alpha, float beta, float avg_num_neighbours) {
    segment_apr_cpp(apr.apr, input_parts.parts, mask_parts.parts, alpha, beta, avg_num_neighbours);
}


void err_fn(const char * msg) {
    std::cerr << msg << std::endl;
}


template<typename T>
void segment_apr_cpp(APR& apr, ParticleData<T>& input_parts, ParticleData<uint16_t>& mask_parts,
                     float alpha, float beta, float avg_num_neighbours) {

    APRTimer timer(true);

    // Initialize Graph object
    typedef Graph<float,float,float> GraphType;
    auto *g = new GraphType(apr.total_number_particles(), avg_num_neighbours*apr.total_number_particles(), &err_fn);

    // Add nodes
    timer.start_timer("add nodes");
    g -> add_node(apr.total_number_particles());
    timer.stop_timer();

    // Find minimum and maximum particle values, to be used in the terminal edge costs
    timer.start_timer("find min max");
    float vmin, vmax;
    find_min_max_val(input_parts, vmin, vmax);
    vmax = 0.75*vmax;
    std::cout << "min: " << vmin << " max: " << vmax << std::endl;
    timer.stop_timer();

    // Loop over particles and edd edges
    timer.start_timer("add edges");
    auto it = apr.random_iterator();
    auto neighbour_iterator = apr.random_iterator();

    uint64_t edge_counter = 0;
    int max_neigh_count = 0;

    for(int level = apr.level_max(); level > apr.level_min(); --level) {

    float base_dist = std::pow(2, apr.level_max()-level);

//#ifdef PYAPR_HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) firstprivate(it, neighbour_iterator) reduction(+:edge_counter) reduction(max:max_neigh_count)
//#endif
        for(int z = 0; z < apr.z_num(level); ++z) {
            for(int x = 0; x < apr.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {

                    int ct_id = it;

                    float val = input_parts[ct_id];
                    float cap = alpha * (val - vmin) / (vmax - vmin);

                    // Add terminal edges based on intensity
                    g -> add_tweights(ct_id, cap, alpha-cap);

                    // Neighbour Particle Cell Face definitions [+y,-y,+x,-x,+z,-z] =  [0,1,2,3,4,5]
                    // Edges are bidirectional, so we only need the positive directions
                    int neigh_counter = 0;
                    for (int direction = 0; direction < 6; direction+=2) {
                        it.find_neighbours_in_direction(direction);

                        // For each face, there can be 0-4 neighbours
                        for (int index = 0; index < it.number_neighbours_in_direction(direction); ++index) {
                            if (neighbour_iterator.set_neighbour_iterator(it, direction, index)) {
                                //will return true if there is a neighbour defined
                                float neigh_val = input_parts[neighbour_iterator];
                                int neigh_level = neighbour_iterator.level();

                                float dist = base_dist;
                                if(neigh_level > level) {
                                    dist = base_dist * 1.5f;
                                } else if(neigh_level < level) {
                                    dist = base_dist * 0.75f;
                                }

                                float diff = (neigh_val - val) / dist;
                                float cost_apr = beta * exp(-diff*diff);

                                g->add_edge(ct_id, neighbour_iterator, cost_apr, cost_apr);

                                edge_counter++;
                                neigh_counter++;
                            }
                        }
                    }
                    if(neigh_counter > max_neigh_count) {
                        max_neigh_count = neigh_counter;
                    }
                }
            }
        }
    }
    timer.stop_timer();

    std::cout << "number of edges = " << edge_counter << " (" << (float)edge_counter/(float)apr.total_number_particles() << " x nparts)" << std::endl;
    std::cout << "max number edges " << max_neigh_count << std::endl;
    // Compute the minimum cut

    timer.start_timer("compute minimum cut");
    g -> maxflow();
    timer.stop_timer();

    // Extract the resulting mask
    timer.start_timer("write output");
    mask_parts.init(apr.total_number_particles());

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int idx = 0; idx < mask_parts.size(); ++idx) {
        mask_parts[idx] = 1 * (g->what_segment(idx) == GraphType::SOURCE);
    }
    timer.stop_timer();

    delete g;
}



void compute_graphcut_pixel(py::array_t<float> &input, py::array_t<uint8_t> &output, float alpha, float beta) {

    APRTimer timer(true);

    auto input_buf = input.request();
    auto output_buf = output.request();

    assert(input_buf.ndim == output_buf.ndim);
    for(int i = 0; i < input_buf.ndim; ++i){
        assert(input_buf.shape[i] == output_buf.shape[i]);
    }

    // Pointers to Python array data
    auto *in_ptr = (float *) input_buf.ptr;
    auto *out_ptr = (uint8_t *) output_buf.ptr;

    int z_num = input_buf.shape[0];
    int y_num = input_buf.shape[1];
    int x_num = input_buf.shape[2];

    int npixels = z_num * x_num * y_num;

    typedef Graph<float,float,float> GraphType;
    auto *g = new GraphType(npixels, 3*npixels, &err_fn); // one node for each pixel, up to 6 neighbours

    // Add nodes
    timer.start_timer("add nodes");
    g -> add_node(npixels);
    timer.stop_timer();

    //  Find the minimum and maximum values in the image
    timer.start_timer("find min max");
    float vmin = in_ptr[0];
    float vmax = in_ptr[0];

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static) reduction(min: vmin) reduction(max: vmax)
#endif
    for(int idx = 0; idx < npixels; ++idx) {
        vmin = std::min(vmin, in_ptr[idx]);
        vmax = std::max(vmax, in_ptr[idx]);
    }
    timer.stop_timer();

    // Loop over pixels and add edges
    timer.start_timer("add edges");
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int z = 0; z < z_num; ++z) {
        for(int y = 0; y< y_num; ++y) {
            for(int x = 0; x < x_num; ++x) {
                int ct_id = z * x_num * y_num + y * x_num + x;
                float ct_val =  in_ptr[ct_id];

                float cap = alpha * (ct_val - vmin) / (vmax - vmin);
                g->add_tweights(ct_id, cap, alpha-cap);

                //neighbour x
                if (x < x_num - 1) {
                    int neigh_id = z * x_num * y_num + y * x_num + x + 1;
                    float neigh_val = in_ptr[neigh_id];
                    float cost = beta * exp(-(ct_val-neigh_val)*(ct_val-neigh_val));
                    g->add_edge(ct_id, neigh_id, cost, cost);
                }
                //neighbour y
                if (y < y_num - 1) {
                    int neigh_id = z * x_num * y_num + (y+1) * x_num + x ;
                    float neigh_val = in_ptr[neigh_id];
                    float cost = beta * exp(-(ct_val-neigh_val)*(ct_val-neigh_val));
                    g->add_edge(ct_id, neigh_id, cost, cost);
                }
                //neighbour z
                if (z < z_num - 1) {
                    int neigh_id = (z+1) * x_num * y_num + y * x_num + x;
                    float neigh_val = in_ptr[neigh_id];
                    float cost = beta * exp(-(ct_val-neigh_val)*(ct_val-neigh_val));
                    g->add_edge(ct_id, neigh_id, cost, cost);
                }
            }
        }
    }
    timer.stop_timer();

    timer.start_timer("compute minimum cut");
    g->maxflow();
    timer.stop_timer();

    timer.start_timer("write output");
#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int idx = 0; idx < npixels; ++idx) {
        out_ptr[idx] = 255 * (g->what_segment(idx) == GraphType::SOURCE);
    }
    timer.stop_timer();

    delete g;
}


void AddPyAPRSegmentation(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());
    m2.def("graphcut", &segment_apr_py<float>, "compute graphcut segmentation of an APR",
           py::arg("apr"), py::arg("input_parts"), py::arg("mask_parts"), py::arg("alpha")=1, py::arg("beta")=1,
           py::arg("avg_num_neighbours")=3.3);
    m2.def("graphcut", &segment_apr_py<uint16_t>, "compute graphcut segmentation of an APR",
           py::arg("apr"), py::arg("input_parts"), py::arg("mask_parts"), py::arg("alpha")=1, py::arg("beta")=1,
           py::arg("avg_num_neighbours")=3.3);
    m2.def("graphcut_pixel", &compute_graphcut_pixel, "comput graphcut segmentation of a pixel image",
           py::arg("input"), py::arg("output"), py::arg("alpha")=1, py::arg("beta")=1);
}
