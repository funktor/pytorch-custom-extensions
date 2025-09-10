#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <deque>
#include <vector>
#include <random>
#include <algorithm>

#define BLOCK_WIDTH 1024
#define COARSE_FACTOR 16
#define MODIFIED_BLOCK_WIDTH COARSE_FACTOR*BLOCK_WIDTH

bool are_equal(float *x, float *y, long start, long end) {
    for (long i = start; i < end; i++) {
        if (fabs(x[i]-y[i]) > 1e-3) {
            std::cout << i << " " << x[i] << " " << y[i] << std::endl;
            return false;
        }
    }
    return true;
}

void generate_data(float *x, long n) {
    static std::random_device dev;
    static std::mt19937 rng(dev());

    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (long i = 0; i < n; i++) x[i] = dist(rng);
}

void print_vector(float *x, long start, long end) {
    std::cout << "[";
    for (long i = start; i <= end; i++) std::cout << x[i] << ", ";
    std::cout << "]" << std::endl;
}

__device__
void co_rank(float *a, float *b, long *indices, const long n, const long m, const long k) {
    long left = 0;
    long right = (k < n)?k-1:n-1;
    long j = -1;

    while (left <= right) {
        long mid = (left + right) >> 1;
        long q = k-(mid+1);
        if (q >= m || b[q] >= a[mid]) {
            j = mid;
            left = mid+1;
        }
        else {
            right = mid-1;
        }
    }

    indices[0] = j;
    indices[1] = k-j-2;
}

__host__ __device__
void merge(float *a, float *b, float *c, long s_a, long e_a, long s_b, long e_b, long s_c, long e_c) {
    long i = s_a;
    long j = s_b;
    long k = s_c;

    while (i <= e_a && j <= e_b){
        if (a[i] < b[j]) c[k++] = a[i++];
        else c[k++] = b[j++];
    }

    while (i <= e_a) {
        c[k++] = a[i++];
    }

    while (j <= e_b) {
        c[k++] = b[j++];
    }
}

__global__
void merge_sorted_arrays(float *a, float *b, float *c, const long n, const long m) {
    long idx = blockDim.x*blockIdx.x + threadIdx.x;

    long s_idx = idx*COARSE_FACTOR;
    long e_idx = (idx + 1)*COARSE_FACTOR;
    e_idx = (e_idx < n+m)?e_idx-1:n+m-1;

    if (s_idx < n+m) {
        long *indices_s = new long[2];
        long *indices_e = new long[2];

        co_rank(a, b, indices_s, n, m, s_idx);
        co_rank(a, b, indices_e, n, m, e_idx+1);

        merge(a, b, c, indices_s[0]+1, indices_e[0], indices_s[1]+1, indices_e[1], s_idx, e_idx);
    }
}

int main(){
    long n = 2000;
    long m = 1000;

    float *a, *b, *c_host, *c_device;

    size_t size_a = sizeof(float)*n;
    size_t size_b = sizeof(float)*m;
    size_t size_c = sizeof(float)*(n+m);

    c_host = new float[n+m];

    cudaMallocManaged(&a, size_a);
    cudaMallocManaged(&b, size_b);
    cudaMallocManaged(&c_device, size_c);

    generate_data(a, n);
    generate_data(b, m);

    std::sort(a, a+n);
    std::sort(b, b+m);

    merge_sorted_arrays<<<(n+m+MODIFIED_BLOCK_WIDTH-1)/MODIFIED_BLOCK_WIDTH, BLOCK_WIDTH>>>(a, b, c_device, n, m);
    cudaDeviceSynchronize();

    merge(a, b, c_host, 0, n-1, 0, m-1, 0, n+m-1);

    std::cout << are_equal(c_host, c_device, 0, n+m-1) << std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c_device);
    delete[] c_host;
}