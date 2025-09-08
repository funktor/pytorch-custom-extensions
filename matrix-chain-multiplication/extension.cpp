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

float* get_dot_cpu(
    const float *a, 
    const float *b, 
    bool a_trans, 
    bool b_trans, 
    const long long a_n, 
    const long long a_m, 
    const long long b_n,
    const long long b_m) {

    long long c_n = (a_trans)?a_m:a_n;
    long long c_m = (b_trans)?b_n:b_m;

    float *c = new float[c_n*c_m];
    cblas_sgemm(
        CblasRowMajor, 
        (a_trans)?CblasTrans:CblasNoTrans, 
        (b_trans)?CblasTrans:CblasNoTrans, 
        c_n, 
        c_m, 
        (a_trans)?a_n:a_m, 
        1.0f, 
        a, 
        a_m, 
        b, 
        b_m, 
        0.0f, 
        c, 
        c_m
    );

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
        return get_dot_cpu(a[i].data_ptr<float>(), a[j].data_ptr<float>(), false, false, a[i].size(0), a[i].size(1), a[j].size(0), a[j].size(1));
    }
    else {
        unsigned int k = dp_part[i*num_matrices + j];
        float *x = mydot_cpu(a, dp_part, i, k, num_matrices);
        float *y = mydot_cpu(a, dp_part, k+1, j, num_matrices);
        return get_dot_cpu(x, y, false, false, a[i].size(0), a[k].size(1), a[j].size(0), a[j].size(1));
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

void optimal_dot_chain_cpu(
    const std::vector<torch::Tensor> &a, 
    float *out, 
    const unsigned int num_matrices) {

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

void compute_backward_pass_cpu(
    const std::vector<torch::Tensor> &ainp, 
    std::vector<torch::Tensor> &aout, 
    float *grad, 
    const unsigned int num_matrices) {

    long long *dp_cost = new long long[num_matrices*num_matrices];
    unsigned int *dp_part = new unsigned int[num_matrices*num_matrices];

    optimal_dot_chain_common(ainp, dp_cost, dp_part, num_matrices);
    delete[] dp_cost;

    omp_set_num_threads(8);
    #pragma omp parallel for shared(aout)
    for (unsigned int i = 0; i < num_matrices; i++) {
        float *out = aout[i].data_ptr<float>();

        if (i == 0) {
            float *out_e = mydot_cpu(ainp, dp_part, i+1, num_matrices-1, num_matrices);

            long long n_e = ainp[1].size(0);
            long long m_e = ainp[num_matrices-1].size(1);

            long long n_grad = ainp[0].size(0);
            long long m_grad = ainp[num_matrices-1].size(1);

            float *d = get_dot_cpu(grad, out_e, false, true, n_grad, m_grad, n_e, m_e);
            for (long long k = 0; k < n_grad*n_e; k++) out[k] = d[k];

            delete[] out_e;
            delete[] d;
        }

        else if (i == num_matrices-1) {
            float *out_s = mydot_cpu(ainp, dp_part, 0, i-1, num_matrices);

            long long n_s = ainp[0].size(0);
            long long m_s = ainp[i-1].size(1);

            long long n_grad = ainp[0].size(0);
            long long m_grad = ainp[num_matrices-1].size(1);

            float *d = get_dot_cpu(out_s, grad, true, false, n_s, m_s, n_grad, m_grad);
            for (long long k = 0; k < m_s*m_grad; k++) out[k] = d[k];

            delete[] out_s;
            delete[] d;
        }

        else {
            float *out_s = mydot_cpu(ainp, dp_part, 0, i-1, num_matrices);
            float *out_e = mydot_cpu(ainp, dp_part, i+1, num_matrices-1, num_matrices);

            long long n_s = ainp[0].size(0);
            long long m_s = ainp[i-1].size(1);

            long long n_e = ainp[i+1].size(0);
            long long m_e = ainp[num_matrices-1].size(1);

            long long n_grad = ainp[0].size(0);
            long long m_grad = ainp[num_matrices-1].size(1);

            long long cost_a =  m_s*n_s*m_grad + m_s*m_grad*n_e;
            long long cost_b =  n_grad*m_grad*n_e + m_s*n_grad*n_e;

            if (cost_a < cost_b) {
                float *out1 = get_dot_cpu(out_s, grad, true, false, n_s, m_s, n_grad, m_grad);
                float *out2 = get_dot_cpu(out1, out_e, false, true, m_s, m_grad, n_e, m_e);
                for (long long k = 0; k < m_s*n_e; k++) out[k] = out2[k];

                delete[] out_s;
                delete[] out_e;
                delete[] out1;
                delete[] out2;
            }

            else {
                float *out1 = get_dot_cpu(grad, out_e, false, true, n_grad, m_grad, n_e, m_e);
                float *out2 = get_dot_cpu(out_s, out1, true, false, n_s, m_s, n_grad, n_e);
                for (long long k = 0; k < m_s*n_e; k++) out[k] = out2[k];

                delete[] out_s;
                delete[] out_e;
                delete[] out1;
                delete[] out2;
            }
        }
    }
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

    std::vector<torch::Tensor> dot_chain_cpu_backward(const torch::Tensor &grad, const std::vector<torch::Tensor> &a) {
        unsigned int num_matrices = a.size();

        omp_set_num_threads(8);
        #pragma omp parallel for shared(a)
        for (unsigned int i = 0; i < num_matrices; i++) {
            TORCH_CHECK(a[i].is_cpu(), "Tensors must be in CPU");
            TORCH_CHECK(a[i].is_contiguous(), "Tensors must be contiguous in memory");
            TORCH_CHECK(a[i].dtype() == torch::kFloat32, "Tensors must be 32-bit float");

            if (i < num_matrices-1) TORCH_CHECK(a[i].size(1) == a[i+1].size(0), "Tensors must be compatible for multiplication")
        }

        TORCH_CHECK(grad.is_cpu(), "Grad must be in CPU");
        TORCH_CHECK(grad.is_contiguous(), "Grad must be contiguous in memory");
        TORCH_CHECK(grad.dtype() == torch::kFloat32, "Grad must be 32-bit float");
        TORCH_CHECK(grad.size(0) == a[0].size(0) && grad.size(1) == a[num_matrices-1].size(1), "Grad must be compatible for multiplication")

        std::vector<torch::Tensor> aout;
        for (unsigned int i = 0; i < num_matrices; i++) {
            torch::Tensor h = torch::empty_like(a[i]);
            aout.push_back(h);
        }

        compute_backward_pass_cpu(a, aout, grad.data_ptr<float>(), num_matrices);
        return aout;
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
        m.def("dot_chain_cpu_backward", &dot_chain_cpu_backward, "Matrix Chain Multiplication CPU Backward");
        m.def("dot_chain_gpu", &dot_chain_gpu, "Matrix Chain Multiplication GPU");
    }
}

