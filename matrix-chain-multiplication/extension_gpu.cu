#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>

#define TILE_WIDTH 32
#define COARSE_FACTOR 8

__global__ 
void cuda_mul(const float *a, const float *b, float *c, const long long n, const long long m, const long long p) {
    __shared__ float Mds[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH*TILE_WIDTH];

    long long bx = blockIdx.x;
    long long by = blockIdx.y;
    long long tx = threadIdx.x;
    long long ty = threadIdx.y;

    long long row = by*TILE_WIDTH + ty;
    long long col_start = bx*TILE_WIDTH*COARSE_FACTOR + tx;

    float Pval[COARSE_FACTOR];
    for (long long r = 0; r < COARSE_FACTOR; r++) Pval[r] = 0.0f;

    for (long long ph = 0; ph < ceil(m/float(TILE_WIDTH)); ph++) {
        if (row < n && (ph*TILE_WIDTH + tx) < m) Mds[ty*TILE_WIDTH+tx] = a[row*m + ph*TILE_WIDTH + tx];
        else Mds[ty*TILE_WIDTH+tx] = 0.0f;

        for (long long r = 0; r < COARSE_FACTOR; r++) {
            long long col = col_start + r*TILE_WIDTH;

            if ((ph*TILE_WIDTH + ty) < m && col < p) Nds[ty*TILE_WIDTH+tx] = b[(ph*TILE_WIDTH+ty)*p + col];
            else Nds[ty*TILE_WIDTH+tx] = 0.0f;
            __syncthreads();

            for (long long i = 0; i < TILE_WIDTH; i++) Pval[r] += Mds[ty*TILE_WIDTH+i]*Nds[i*TILE_WIDTH+tx];
            __syncthreads();
        }
    }

    for (long long r = 0; r < COARSE_FACTOR; r++) {
        long long col = col_start + r*TILE_WIDTH;
        if (row < n && col < p) c[row*p+col] = Pval[r];
    }
}

void cuda_mul_launcher(const float *a, const float *b, float *c, const long long n, const long long m, const long long p) {
    dim3 bd(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gd((p+TILE_WIDTH-1)/TILE_WIDTH, (n+TILE_WIDTH-1)/TILE_WIDTH, 1);

    cuda_mul<<<gd, bd>>>(a, b, c, n, m, p);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}