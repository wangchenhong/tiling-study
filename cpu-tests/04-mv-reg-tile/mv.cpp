#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

#define BM 32
#define BN 32

py::array_t<double> kernel0(py::array_t<double> a, py::array_t<double> b) {
    auto M = a.shape(0);
    auto N = a.shape(1);
    auto c = py::array_t<double>(M);
    auto a_ptr = a.mutable_data();
    auto b_ptr = b.mutable_data();
    auto c_ptr = c.mutable_data();

    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < M; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            sum += a_ptr[i * N + j] * b_ptr[j];
        }
        c_ptr[i] = sum;
    }

    return c;
}

py::array_t<double> kernel(py::array_t<double> a, py::array_t<double> b) {
    auto M = a.shape(0);
    auto N = a.shape(1);
    auto c = py::array_t<double>(M);
    auto a_ptr = a.data();
    auto b_ptr = b.data();
    auto c_ptr = c.mutable_data();

    #pragma omp parallel for num_threads(64)
    for (int i = 0; i < M; i+=4) {
        double c0 = 0;
        double c1 = 0;
        double c2 = 0;
        double c3 = 0;
        
        for (int j = 0; j < N; j++) {
            double b0 = b_ptr[j];

            c0 += a_ptr[i * N + j] * b0;
            c1 += a_ptr[(i+1) * N + j] * b0;
            c2 += a_ptr[(i+2) * N + j] * b0;
            c3 += a_ptr[(i+3) * N + j] * b0;
        }
        c_ptr[i] = c0;
        c_ptr[i+1] = c1;
        c_ptr[i+2] = c2;
        c_ptr[i+3] = c3;
    }

    return c;
}

py::array_t<double> kernel_l1(py::array_t<double> a, py::array_t<double> b) {
    auto M = a.shape(0);
    auto N = a.shape(1);
    auto c = py::array_t<double>(M);
    auto a_ptr = a.data();
    auto b_ptr = b.data();
    auto c_ptr = c.mutable_data();

    #pragma omp parallel for num_threads(64)
    for (int ii = 0; ii < M; ii += BM) {
        for (int jj = 0; jj < N; jj += BN) {
            for (int i = ii; i < ii+BM; i+=4) {
                double c0 = 0;
                double c1 = 0;
                double c2 = 0;
                double c3 = 0;
                
                for (int j = jj; j < jj+BN; j++) {
                    double b0 = b_ptr[j];

                    c0 += a_ptr[i * N + j] * b0;
                    c1 += a_ptr[(i+1) * N + j] * b0;
                    c2 += a_ptr[(i+2) * N + j] * b0;
                    c3 += a_ptr[(i+3) * N + j] * b0;
                }
                c_ptr[i] = c0;
                c_ptr[i+1] = c1;
                c_ptr[i+2] = c2;
                c_ptr[i+3] = c3;
            }
        }
    }

    return c;
}

PYBIND11_MODULE(mv, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("kernel", &kernel, "");
    m.def("kernel0", &kernel0, "");
}