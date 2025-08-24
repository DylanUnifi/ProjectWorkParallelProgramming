//
// gram_omp.cpp
// Created by berka on 24/08/2025.
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

/*
 * Compute full symmetric Gram matrix K = X X^T (float32)
 * X: [N, D] row-major, contigous
 * Returns K: [N, N] (float32)
 */

py::array_t<float> gram_sym_omp(py::array_t<float, py::array::c_style | py::array::forcecast> X) {
    auto bx = X.request();
    const int N = static_cast<int>(bx.shape[0]);
    const int D = static_cast<int>(bx.shape[1]);
    const float* __restrict__ x = static_cast<float*>(bx.ptr);

    py::array_t<float> K({N, N});
    auto bk = K.request();
    float* __restrict__ k = static_cast<float *>(bk.ptr);

    // zero init (in case)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N*N; ++i) k[i] = 0.f;

    // upper triangle (including diagonal)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        // diagonal: if X rows are L2-normalized, this will be 1; otherwise compute
        float dot_ii = 0.f;
        for (int d = 0; d < D; ++d) dot_ii += x[i*D + d] * x[i*D + d];
        k[i*N + i] = dot_ii;

        for (int j = i + 1; j < N; ++j) {
            float sum = 0.f;
            const float* xi = x + i*D;
            const float* xj = x + j*D;
            // simple dot product; let compiler vectorize
            for (int d = 0; d < D; ++d) sum += xi[d] * xj[d];
            k[j*N + i] = sum;
        }
    }

    // mirror to lower triangle
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            k[j*N + i] = k[i*N + j];
        }
    }

    return K;
}