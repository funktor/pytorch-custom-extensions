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

void cuda_free(float *d);
float* get_dot_gpu(const float *a, const float *b, const long long n, const long long m, const long long p);

float* get_dot_cpu(const float *a, const float *b, const long long n, const long long m, const long long p) {
    float *c = new float[n*p];

    for (long long i = 0; i < n; i++) {
        for (long long j = 0; j < p; j++) {
            c[i*p + j] = 0.0;
        }
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, p, m, 1.0f, a, m, b, p, 0.0f, c, p);
    return c;
}

float *mydot_cpu(
    const std::vector<torch::Tensor> &a, 
    const unsigned int *dp_part, 
    const unsigned int i, 
    const unsigned int j, 
    const unsigned int num_matrices) {

    if (i == j) return a[i].data_ptr<float>();
    else if (i == j-1) {
        return get_dot_cpu(a[i].data_ptr<float>(), a[j].data_ptr<float>(), a[i].size(0), a[i].size(1), a[j].size(1));
    }
    else {
        unsigned int k = dp_part[i*num_matrices + j];
        float *x = mydot_cpu(a, dp_part, i, k, num_matrices);
        float *y = mydot_cpu(a, dp_part, k+1, j, num_matrices);
        return get_dot_cpu(x, y, a[i].size(0), a[k].size(1), a[j].size(1));
    }
}

float *mydot_gpu(
    const std::vector<torch::Tensor> &a, 
    const unsigned int *dp_part, 
    const unsigned int i, 
    const unsigned int j, 
    const unsigned int num_matrices) {

    if (i == j) return a[i].data_ptr<float>();
    else if (i == j-1) {
        return get_dot_gpu(a[i].data_ptr<float>(), a[j].data_ptr<float>(), a[i].size(0), a[i].size(1), a[j].size(1));
    }
    else {
        unsigned int k = dp_part[i*num_matrices + j];
        float *x = mydot_gpu(a, dp_part, i, k, num_matrices);
        float *y = mydot_gpu(a, dp_part, k+1, j, num_matrices);
        return get_dot_gpu(x, y, a[i].size(0), a[k].size(1), a[j].size(1));
    }
}

void optimal_dot_chain_common(
    const std::vector<torch::Tensor> &a, 
    long long *dp_cost, 
    unsigned int *dp_part, 
    const unsigned int num_matrices) {

    for (unsigned int length = 1; length <= num_matrices; length++) {

        omp_set_num_threads(8);
        #pragma omp parallel for shared(dp_cost, dp_part)
        for (unsigned int i = 0; i < num_matrices-length+1; i++) {
            unsigned int j = i+length-1;

            if (length == 1) {
                dp_cost[i*num_matrices + j] = 0; 
                dp_part[i*num_matrices + j] = i;
            }     
            else if (length == 2) {
                dp_cost[i*num_matrices + j] = a[i].size(0)*a[i].size(1)*a[j].size(1);
                dp_part[i*num_matrices + j] = j;
            }
            else {
                long long min_cost = LONG_MAX;
                unsigned int min_break_pt = i;

                for (unsigned int k = i; k < j; k++) {
                    long long x = dp_cost[i*num_matrices + k];
                    long long y = dp_cost[(k+1)*num_matrices + j];

                    long long new_cost = x + y + a[i].size(0)*a[k].size(1)*a[j].size(1);
                    if (new_cost < min_cost) {
                        min_cost = new_cost;
                        min_break_pt = k;
                    }
                }

                dp_cost[i*num_matrices + j] = min_cost;
                dp_part[i*num_matrices + j] = min_break_pt;
            }
        }
    }
}

void optimal_dot_chain_cpu(const std::vector<torch::Tensor> &a, float *out, const unsigned int num_matrices) {
    long long *dp_cost = new long long[num_matrices*num_matrices];
    unsigned int *dp_part = new unsigned int[num_matrices*num_matrices];

    optimal_dot_chain_common(a, dp_cost, dp_part, num_matrices);

    std::cout << "Optimal Cost = " << dp_cost[num_matrices-1] << std::endl;
    delete[] dp_cost;

    float *d = mydot_cpu(a, dp_part, 0, num_matrices-1, num_matrices);
    for (long long k = 0; k < a[0].size(0)*a[num_matrices-1].size(1); k++) out[k] = d[k];

    delete[] dp_part;
    delete[] d;
}

void optimal_dot_chain_gpu(const std::vector<torch::Tensor> &a, float *out, const unsigned int num_matrices) {
    long long *dp_cost = new long long[num_matrices*num_matrices];
    unsigned int *dp_part = new unsigned int[num_matrices*num_matrices];

    optimal_dot_chain_common(a, dp_cost, dp_part, num_matrices);

    std::cout << "Optimal Cost = " << dp_cost[num_matrices-1] << std::endl;
    delete[] dp_cost;

    float *d = mydot_gpu(a, dp_part, 0, num_matrices-1, num_matrices);
    for (long long k = 0; k < a[0].size(0)*a[num_matrices-1].size(1); k++) out[k] = d[k];

    delete[] dp_part;
    cuda_free(d);
}

namespace extension_cpp {
    torch::Tensor dot_chain_cpu(const std::vector<torch::Tensor> &a) {
        unsigned int num_matrices = a.size();

        omp_set_num_threads(8);
        #pragma omp parallel for shared(a)
        for (unsigned int i = 0; i < num_matrices; i++) {
            TORCH_CHECK(a[i].is_cpu(), "Tensors must be in CPU");
            TORCH_CHECK(a[i].is_contiguous(), "Tensors must be contiguous in memory");
            TORCH_CHECK(a[i].dtype() == torch::kFloat32, "Tensors must be 32-bit float");

            if (i < num_matrices-1) TORCH_CHECK(a[i].size(1) == a[i+1].size(0), "Tensors must be compatible for multiplication")
        }

        torch::Tensor h = torch::empty({a[0].size(0), a[num_matrices-1].size(1)});
        optimal_dot_chain_cpu(a, h.data_ptr<float>(), num_matrices);
        return h;
    }

    torch::Tensor dot_chain_gpu(const std::vector<torch::Tensor> &a) {
        unsigned int num_matrices = a.size();

        omp_set_num_threads(8);
        #pragma omp parallel for shared(a)
        for (unsigned int i = 0; i < num_matrices; i++) {
            TORCH_CHECK(a[i].is_cuda(), "Tensors must be in GPU");
            TORCH_CHECK(a[i].is_contiguous(), "Tensors must be contiguous in memory");
            TORCH_CHECK(a[i].dtype() == torch::kFloat32, "Tensors must be 32-bit float");

            if (i < num_matrices-1) TORCH_CHECK(a[i].size(1) == a[i+1].size(0), "Tensors must be compatible for multiplication")
        }

        torch::Tensor h = torch::empty({a[0].size(0), a[num_matrices-1].size(1)});
        optimal_dot_chain_gpu(a, h.data_ptr<float>(), num_matrices);
        
        return h;
    }

    PYBIND11_MODULE(extension_cpp, m) {
        m.def("dot_chain_cpu", &dot_chain_cpu, "Matrix Chain Multiplication CPU");
        m.def("dot_chain_gpu", &dot_chain_gpu, "Matrix Chain Multiplication GPU");
    }
}
