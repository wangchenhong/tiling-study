#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

py::array_t<double> add(py::array_t<double> a, py::array_t<double> b) {
    auto c = py::array_t<double>(a.size());
    auto a_ptr = a.mutable_data();
    auto b_ptr = b.mutable_data();
    auto c_ptr = c.mutable_data();

    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < a.size(); i++ ) {
        c_ptr[i] = a_ptr[i] + b_ptr[i];
    }
    return c;
}

PYBIND11_MODULE(vec_add, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}