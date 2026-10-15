#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Collatz Kernel: parallel stopping times and peak calculation
__global__ void collatz_kernel(
    unsigned long long start,
    int count,
    int max_steps,
    int* __restrict__ out_steps,
    unsigned long long* __restrict__ out_peaks
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < count) {
        unsigned long long n = start + idx;
        int steps = 0;
        unsigned long long peak = n;
        while (n > 1ULL && steps < max_steps) {
            if ((n & 1ULL) == 0ULL) {
                n >>= 1ULL;
            } else {
                n = 3ULL * n + 1ULL;
            }
            if (n > peak) peak = n;
            steps++;
        }
        out_steps[idx] = steps;
        out_peaks[idx] = peak;
    }
}

// Elliptic curve a_p point count Kernel (Euler criterion over F_p)
__global__ void elliptic_ap_kernel(
    long long a,
    long long b,
    const long long* __restrict__ primes,
    int* __restrict__ out_ap,
    int num_primes
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_primes) {
        long long p = primes[idx];
        if (p <= 2) {
            out_ap[idx] = 0;
            return;
        }
        long long sum_leg = 0;
        long long exp = (p - 1) / 2;
        for (long long x = 0; x < p; ++x) {
            long long val = ((x * x % p) * x % p + a * x % p + b) % p;
            if (val < 0) val += p;
            if (val != 0) {
                // Modular exponentiation val^exp % p
                long long base = val;
                long long e = exp;
                long long res = 1;
                while (e > 0) {
                    if (e & 1) res = (res * base) % p;
                    base = (base * base) % p;
                    e >>= 1;
                }
                sum_leg += (res == p - 1) ? -1 : 1;
            }
        }
        out_ap[idx] = (int)(-sum_leg);
    }
}

// Matmul Tiled Kernel
__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" {

int hip_collatz_batch(
    unsigned long long start,
    int count,
    int max_steps,
    int* h_steps,
    unsigned long long* h_peaks
) {
    int *d_steps = NULL;
    unsigned long long *d_peaks = NULL;
    if (hipMalloc((void**)&d_steps, count * sizeof(int)) != hipSuccess) return -1;
    if (hipMalloc((void**)&d_peaks, count * sizeof(unsigned long long)) != hipSuccess) {
        hipFree(d_steps);
        return -2;
    }

    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    hipLaunchKernelGGL(collatz_kernel, dim3(blocks), dim3(threads), 0, 0, start, count, max_steps, d_steps, d_peaks);
    hipDeviceSynchronize();

    hipMemcpy(h_steps, d_steps, count * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_peaks, d_peaks, count * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    hipFree(d_steps);
    hipFree(d_peaks);
    return 0;
}

int hip_elliptic_ap_batch(
    long long a,
    long long b,
    const long long* h_primes,
    int* h_ap,
    int num_primes
) {
    long long *d_primes = NULL;
    int *d_ap = NULL;
    if (hipMalloc((void**)&d_primes, num_primes * sizeof(long long)) != hipSuccess) return -1;
    if (hipMalloc((void**)&d_ap, num_primes * sizeof(int)) != hipSuccess) {
        hipFree(d_primes);
        return -2;
    }
    hipMemcpy(d_primes, h_primes, num_primes * sizeof(long long), hipMemcpyHostToDevice);

    int threads = 256;
    int blocks = (num_primes + threads - 1) / threads;
    hipLaunchKernelGGL(elliptic_ap_kernel, dim3(blocks), dim3(threads), 0, 0, a, b, d_primes, d_ap, num_primes);
    hipDeviceSynchronize();

    hipMemcpy(h_ap, d_ap, num_primes * sizeof(int), hipMemcpyDeviceToHost);
    hipFree(d_primes);
    hipFree(d_ap);
    return 0;
}

int hip_matmul(
    const float* h_A,
    const float* h_B,
    float* h_C,
    int M,
    int N,
    int K
) {
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    if (hipMalloc((void**)&d_A, M * K * sizeof(float)) != hipSuccess) return -1;
    if (hipMalloc((void**)&d_B, K * N * sizeof(float)) != hipSuccess) {
        hipFree(d_A);
        return -2;
    }
    if (hipMalloc((void**)&d_C, M * N * sizeof(float)) != hipSuccess) {
        hipFree(d_A);
        hipFree(d_B);
        return -3;
    }

    hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    hipLaunchKernelGGL(matmul_kernel, blocks, threads, 0, 0, d_A, d_B, d_C, M, N, K);
    hipDeviceSynchronize();

    hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    return 0;
}

}
