#include <Python.h>
#include <torch/extension.h>
#include <tbb/tbb.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <deque>
#include <vector>
#include <random>
#include <omp.h>
#include <cblas.h>

namespace extension_cpp {
    struct MatrixCost {
        long long cost;
        unsigned int break_pt;
    };

    float* get_dot(const float *a, const float *b, const long long n, const long long m, const long long p) {
        float *c = new float[n*p];

        for (long long i = 0; i < n; i++) {
            for (long long j = 0; j < p; j++) {
                c[i*p + j] = 0.0;
            }
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, p, m, 1.0f, a, m, b, p, 0.0f, c, p);
        return c;
    }

    float *mydot(
        const std::vector<torch::Tensor> &a, 
        const MatrixCost *dp, 
        const unsigned int i, 
        const unsigned int j, 
        const unsigned int num_matrices) {

        if (i == j) return a[i].data_ptr<float>();
        else if (i == j-1) {
            return get_dot(a[i].data_ptr<float>(), a[j].data_ptr<float>(), a[i].size(0), a[i].size(1), a[j].size(1));
        }
        else {
            unsigned int k = dp[i*num_matrices + j].break_pt;
            float *x = mydot(a, dp, i, k, num_matrices);
            float *y = mydot(a, dp, k+1, j, num_matrices);
            return get_dot(x, y, a[i].size(0), a[k].size(1), a[j].size(1));
        }
    }

    void get_dot_sequence(
        const std::vector<torch::Tensor> &a, 
        float *out, 
        const unsigned int num_matrices
    ) {
        MatrixCost *dp = new MatrixCost[num_matrices*num_matrices];

        for (unsigned int length = 1; length <= num_matrices; length++) {
            omp_set_num_threads(8);
            #pragma omp parallel for shared(dp)
            for (unsigned int i = 0; i < num_matrices-length+1; i++) {
                unsigned int j = i+length-1;

                if (length == 1) dp[i*num_matrices + j] = {0, i};           
                else if (length == 2) dp[i*num_matrices + j] = {a[i].size(0)*a[i].size(1)*a[j].size(1), j};
                else {
                    long long min_cost = LONG_MAX;
                    unsigned int min_break_pt = i;

                    for (unsigned int k = i; k < j; k++) {
                        MatrixCost x = dp[i*num_matrices + k];
                        MatrixCost y = dp[(k+1)*num_matrices + j];

                        long long new_cost = x.cost + y.cost + a[i].size(0)*a[k].size(1)*a[j].size(1);
                        if (new_cost < min_cost) {
                            min_cost = new_cost;
                            min_break_pt = k;
                        }
                    }

                    dp[i*num_matrices + j] = {min_cost, min_break_pt};
                }
            }
        }

        std::cout << "Optimal Cost = " << dp[num_matrices-1].cost << std::endl;

        float *d = mydot(a, dp, 0, num_matrices-1, num_matrices);
        for (long long k = 0; k < a[0].size(0)*a[num_matrices-1].size(1); k++) out[k] = d[k];

        delete[] dp;
        delete[] d;
    }

    torch::Tensor dot_chain_cpu(const std::vector<torch::Tensor> &a) {
        unsigned int num_matrices = a.size();
        torch::Tensor h = torch::empty({a[0].size(0), a[num_matrices-1].size(1)});
        get_dot_sequence(a, h.data_ptr<float>(), num_matrices);
        return h;
    }

    PYBIND11_MODULE(extension_cpp, m) {
        m.def("dot_chain_cpu", &dot_chain_cpu, "Matrix Chain Multiplication CPU");
    }
}
