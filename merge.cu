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

#define TILE_WIDTH 1024
#define BLOCK_WIDTH 1024
#define COARSE_FACTOR 16
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
    __shared__ float a_shared[TILE_WIDTH];
    __shared__ float b_shared[TILE_WIDTH];
    __shared__ long block_metadata[4];

    if (threadIdx.x == 0) {
        block_metadata[0] = co_rank(a, b, n, m, blockDim.x*blockIdx.x*COARSE_FACTOR);
        block_metadata[1] = blockDim.x*blockIdx.x*COARSE_FACTOR - block_metadata[0] - 2;
        block_metadata[2] = co_rank(a, b, n, m, blockDim.x*(blockIdx.x + 1)*COARSE_FACTOR);
        block_metadata[3] = blockDim.x*(blockIdx.x + 1)*COARSE_FACTOR - block_metadata[2] - 2;

        block_metadata[2] = (block_metadata[2]+1 < n)?block_metadata[2]+1:n;
        block_metadata[3] = (block_metadata[3]+1 < m)?block_metadata[3]+1:m;
    }
    
    __syncthreads();

    for (long i = threadIdx.x; i < TILE_WIDTH; i += BLOCK_WIDTH) {
        if (i + block_metadata[0] + 1 < block_metadata[2]) a_shared[i] = a[i + block_metadata[0] + 1];
        else a_shared[i] = 1.0e300;
        
        if (i + block_metadata[1] + 1 < block_metadata[3]) b_shared[i] = b[i + block_metadata[1] + 1];
        else b_shared[i] = 1.0e300;
    }

    __syncthreads();

    long idx = blockDim.x*blockIdx.x + threadIdx.x;
    long a_curr_start = co_rank(a, b, n, m, idx*COARSE_FACTOR);
    long c_curr = idx*COARSE_FACTOR;

    while (block_metadata[0] < block_metadata[2] || block_metadata[1] < block_metadata[3]) {
        long c_end = ((idx + 1)*COARSE_FACTOR < (n+m))?(idx + 1)*COARSE_FACTOR:(n+m);
        
        if (c_curr < c_end) {
            c_curr = merge(
                a_shared, 
                b_shared, 
                c, 
                a_curr_start-block_metadata[0], 
                idx*COARSE_FACTOR-a_curr_start-1-(block_metadata[1]+1), 
                c_curr, 
                TILE_WIDTH, 
                TILE_WIDTH, 
                c_end
            );
        }
        
        __syncthreads();

        if (threadIdx.x == 0) {
            block_metadata[0] += TILE_WIDTH;
            block_metadata[1] += TILE_WIDTH;
        }
        
        __syncthreads();
        
        for (long i = threadIdx.x; i < TILE_WIDTH; i += BLOCK_WIDTH) {
            if (i + block_metadata[0] + 1 < block_metadata[2]) a_shared[i] = a[i + block_metadata[0] + 1];
            else a_shared[i] = 1.0e300;
            
            if (i + block_metadata[1] + 1 < block_metadata[3]) b_shared[i] = b[i + block_metadata[1] + 1];
            else b_shared[i] = 1.0e300;
        }

        __syncthreads();
    }
}

int main(){
    long n = 1e4;
    long m = 1e4;

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
    merge_sorted_arrays<<<(n+m+MODIFIED_BLOCK_WIDTH-1)/MODIFIED_BLOCK_WIDTH, BLOCK_WIDTH>>>(a, b, c_device, n, m);
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