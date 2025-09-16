#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <Python.h>
#include <torch/extension.h>
#include <tbb/tbb.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <deque>
#include <vector>
#include <random>
#include <omp.h>
#include <cblas.h>

#define TILE_WIDTH 32
#define COARSE_FACTOR 8
#define MODIFIED_TILE_WIDTH 256

__global__ 
void cuda_mul(const float *a, const float *b, float *c, const long long n, const long long m, const long long p) {
    __shared__ float Mds[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;

    float res = 0.0;
    for (int ph = 0; ph < ceil(m/float(TILE_WIDTH)); ph++) {
        if (row < n && (ph*TILE_WIDTH + tx) < m) Mds[ty*TILE_WIDTH+tx] = a[row*m + ph*TILE_WIDTH + tx];
        else Mds[ty*TILE_WIDTH+tx] = 0.0f;

        if ((ph*TILE_WIDTH + ty) < m && col < p) Nds[ty*TILE_WIDTH+tx] = b[(ph*TILE_WIDTH+ty)*p + col];
        else Nds[ty*TILE_WIDTH+tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) res += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx];
        __syncthreads();
    }

    if (row < n && col < p) c[row*p+col] = res;
}

void cuda_free(float *d) {
    cudaFree(d);
}

float* cublas_mul(const float *a, const float *b, const long long n, const long long m, const long long p) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *c = new float[n*p];
    cudaMallocManaged(&c, n*p*sizeof(float));

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, n, m, &alpha, b, p, a, m, &beta, c, p);

    cudaDeviceSynchronize();
    cublasDestroy(handle);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

    return c;
}

float* get_dot_gpu(const float *a, const float *b, const long long n, const long long m, const long long p) {
    float *c = new float[n*p];
    cudaMallocManaged(&c, n*p*sizeof(float));

    dim3 bd(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gd((p+TILE_WIDTH-1)/TILE_WIDTH, (n+TILE_WIDTH-1)/TILE_WIDTH, 1);

    cuda_mul<<<gd, bd>>>(a, b, c, n, m, p);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

    return c;
}