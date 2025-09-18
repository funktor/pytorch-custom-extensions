#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <deque>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#define TILE_WIDTH 8192
#define BLOCK_WIDTH 1024
#define COARSE_FACTOR 128
#define MODIFIED_BLOCK_WIDTH 16384

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
    std::cout << std::endl;
}

__device__
long co_rank(float *a, float *b, const long n, const long m, const long k) {
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

    return j;
}

__host__ __device__
long merge(float *a, float *b, float *c, const long s_a, const long s_b, const long s_c, const long e_a, const long e_b, const long e_c) {
    long i = s_a;
    long j = s_b;
    long k = s_c;

    while (i >= 0 && j >= 0 && k >= 0 && i < e_a && j < e_b && k < e_c){
        if (a[i] < b[j]) c[k++] = a[i++];
        else c[k++] = b[j++];
    }

    while (i >= 0 && k >= 0 && i < e_a && k < e_c) {
        c[k++] = a[i++];
    }

    while (j >= 0 && k >= 0 && j < e_b && k < e_c) {
        c[k++] = b[j++];
    }

    return k;
}

__global__
void merge_sorted_arrays(float *a, float *b, float *c, const long n, const long m) {
    __shared__ float shared[TILE_WIDTH];
    __shared__ long a_metadata[2];
    __shared__ long b_metadata[2];
    __shared__ long c_metadata[1];

    long c_block_start = BLOCK_WIDTH*blockIdx.x*COARSE_FACTOR;
    long c_block_end = (BLOCK_WIDTH*(blockIdx.x+1)*COARSE_FACTOR < n+m)?BLOCK_WIDTH*(blockIdx.x+1)*COARSE_FACTOR:n+m;

    if (threadIdx.x == 0) {
        a_metadata[1] = co_rank(a, b, n, m, c_block_start);
        b_metadata[1] = c_block_start-a_metadata[1]-2;
        c_metadata[0] = c_block_start;
    }
    
    __syncthreads();

    while (c_metadata[0] < c_block_end) {
        long c_tile_end = ((c_metadata[0] + TILE_WIDTH) < c_block_end)?(c_metadata[0] + TILE_WIDTH):c_block_end;

        if (threadIdx.x == 0) {
            a_metadata[0] = a_metadata[1] + 1;
            b_metadata[0] = b_metadata[1] + 1;

            a_metadata[1] = co_rank(a, b, n, m, c_tile_end);
            b_metadata[1] = c_tile_end-a_metadata[1]-2;
        }

        __syncthreads();

        long p = a_metadata[1]-a_metadata[0]+1;
        long q = b_metadata[1]-b_metadata[0]+1;

        for (long i = threadIdx.x; i < TILE_WIDTH; i += BLOCK_WIDTH) {
            if (i + a_metadata[0] <= a_metadata[1]) shared[i] = a[i + a_metadata[0]];
            else if (i + b_metadata[0] - p <= b_metadata[1]) shared[i] = b[i + b_metadata[0] - p];
            else shared[i] = 1.0e300;
        }
    
        __syncthreads();

        long per_thread = (TILE_WIDTH + BLOCK_WIDTH - 1)/BLOCK_WIDTH;

        long a_start = co_rank(&shared[0], &shared[p], p, q, threadIdx.x*per_thread);
        long b_start = threadIdx.x*per_thread-a_start-2;

        long c_start = threadIdx.x*per_thread;
        long c_end = (threadIdx.x+1)*per_thread;
        c_end = (c_end < TILE_WIDTH)?c_end:TILE_WIDTH;

        merge(&shared[0], &shared[p], &c[c_metadata[0]], a_start+1, b_start+1, c_start, p, q, c_end);
        __syncthreads();

        if (threadIdx.x == 0) c_metadata[0] += TILE_WIDTH;
        __syncthreads();
    }
}

__global__
void merge_sorted_arrays2(float *a, float *b, float *c, const long n, const long m) {
    long idx = blockDim.x*blockIdx.x + threadIdx.x;
    long s_idx = idx*COARSE_FACTOR;

    if (s_idx < n + m) {
        long a_start = co_rank(a, b, n, m, s_idx);
        long b_start = s_idx-a_start-2;

        merge(a, b, c, a_start+1, b_start+1, s_idx, n, m, s_idx + COARSE_FACTOR);
    }
}

int main(){
    long n = 1e6;
    long m = 1e6;

    float *a, *b, *c_host, *c_device;

    size_t size_a = sizeof(float)*n;
    size_t size_b = sizeof(float)*m;
    size_t size_c = sizeof(float)*(n+m);

    c_host = new float[n+m];

    cudaError_t err = cudaMallocManaged(&a, size_a);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error A : %s\n", cudaGetErrorString(err));
    }

    err = cudaMallocManaged(&b, size_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error B: %s\n", cudaGetErrorString(err));
    }

    err = cudaMallocManaged(&c_device, size_c);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error C: %s\n", cudaGetErrorString(err));
    }

    generate_data(a, n);
    generate_data(b, m);

    std::sort(a, a+n);
    std::sort(b, b+m);

    auto start = std::chrono::high_resolution_clock::now();
    merge_sorted_arrays<<<(n+m+(BLOCK_WIDTH*COARSE_FACTOR)-1)/(BLOCK_WIDTH*COARSE_FACTOR), BLOCK_WIDTH>>>(a, b, c_device, n, m);
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "CUDA Duration = " << duration.count() << " ms" << std::endl;

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
    }

    start = std::chrono::high_resolution_clock::now();
    merge(a, b, c_host, 0, 0, 0, n, m, n+m);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Standard Duration = " << duration.count() << " ms" << std::endl;

    std::cout << are_equal(c_host, c_device, 0, n+m-1) << std::endl;

    // print_vector(a, 0, n-1);
    // print_vector(b, 0, m-1);
    // print_vector(c_host, 0, n+m-1);
    // print_vector(c_device, 0, n+m-1);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c_device);
    delete[] c_host;
}