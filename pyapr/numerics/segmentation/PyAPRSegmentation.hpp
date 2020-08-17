#include "data_structures/APR/APR.hpp"
#include "data_containers/PyAPR.hpp"
#include "data_containers/PyParticleData.hpp"
#include "maxflow-v3.04.src/graph.h"

namespace py = pybind11;

void example_graphcut() {

    typedef Graph<int,int,int> GraphType;
	auto *g = new GraphType(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1);

	g -> add_node();
	g -> add_node();

	g -> add_tweights( 0,   /* capacities */  1, 5 );
	g -> add_tweights( 1,   /* capacities */  2, 6 );
	g -> add_edge( 0, 1,    /* capacities */  3, 4 );

	int flow = g -> maxflow();

	printf("Flow = %d\n", flow);
	printf("Minimum cut:\n");
	if (g->what_segment(0) == GraphType::SOURCE)
		printf("node0 is in the SOURCE set\n");
	else
		printf("node0 is in the SINK set\n");
	if (g->what_segment(1) == GraphType::SOURCE)
		printf("node1 is in the SOURCE set\n");
	else
		printf("node1 is in the SINK set\n");

	delete g;
}

template<typename T, typename V>
void find_min_max_val(ParticleData<T>& parts, V& vmin, V& vmax) {

    // find minimum value
    T min_val = std::numeric_limits<T>::max();

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for reduction(min:min_val)
#endif
    for(size_t idx = 0; idx < parts.data.size(); ++idx) {
        if(parts.data[idx] < min_val) {
            min_val = parts.data[idx];
        }
    }

    vmin = min_val;

    // find maximum value
    T max_val = std::numeric_limits<T>::min();

#ifdef PYAPR_HAVE_OPENMP
#pragma omp parallel for reduction(max:min_val)
#endif
    for(size_t idx = 0; idx < parts.data.size(); ++idx) {
        if(parts.data[idx] > max_val) {
            max_val = parts.data[idx];
        }
    }

    vmax = max_val;
}

template<typename T>
void segment_apr_py(PyAPR& apr, PyParticleData<T>& input_parts, PyParticleData<uint16_t>& mask_parts) {
    segment_apr_cpp(apr.apr, input_parts.parts, mask_parts.parts);
}


template<typename T>
void segment_apr_cpp(APR& apr, ParticleData<T>& input_parts, ParticleData<uint16_t>& mask_parts) {

    // create graph object
    auto *g = new Graph<float, float, float>(apr.total_number_particles(), 6*apr.total_number_particles());

    APRTimer timer(true);

    timer.start_timer("add nodes");
    // add nodes
    g -> add_node(apr.total_number_particles());
    timer.stop_timer();

    timer.start_timer("find min max");
    // find min and max value of input particles
    float vmin, vmax;
    find_min_max_val(input_parts, vmin, vmax);
    std::cout << "min value: " << vmin << " max value: " << vmax << std::endl;
    timer.stop_timer();

    timer.start_timer("add terminal edges");
    // initialize linear apr iterator
    auto it = apr.iterator();

    // add terminal edges
    for(int level = apr.level_min(); level <= apr.level_max(); ++level) {
//#ifdef PYAPR_HAVE_OPENMP
//#pragma omp parallel for schedule(dynamic) firstprivate(it)
//#endif
        for(int z = 0; z < apr.z_num(level); ++z) {
            for(int x = 0; x < apr.x_num(level); ++x) {
                for(it.begin(level, z, x); it < it.end(); ++it) {

                    float val = input_parts[it];
                    float cap = (val - vmin) / (vmax - vmin);

                    g -> add_tweights(it, cap, 1-cap);
                }
            }
        }
    }
    timer.stop_timer();

    auto randit = apr.random_iterator();

    timer.start_timer("compute maxflow");
    g -> maxflow();
    timer.stop_timer();


    delete g;
}


void compute_graphcut_pixel(py::array_t<float> &img) {
    auto buf = img.request();

    int z_num = buf.shape[0];
    int y_num = buf.shape[1];
    int x_num = buf.shape[2];

    float *ptr = (float *) buf.ptr;

    std::cout << ptr[13] << std::endl;

    auto *g = new Graph<float, float, float>(apr.total_number_particles(), 6*apr.total_number_particles());
    //g -> add_node();

    // index of (z, y, x)  is z * x_num * y_num + y * x_num + x
    for(int z = 0; z < z_num; ++z) {
        // do stuff
    }


    delete g;
}


void AddPyAPRSegmentation(py::module &m, const std::string &modulename) {

    auto m2 = m.def_submodule(modulename.c_str());
    m2.def("example_graphcut", &example_graphcut, "description");
    m2.def("graphcut", &segment_apr_py<float>, "compute graphcut segmentation of an APR",
           py::arg("apr"), py::arg("input_parts"), py::arg("mask_parts"));
    m2.def("graphcut_pixel", &compute_graphcut_pixel, "help");
}