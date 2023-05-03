#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_functions.h>
#include <thrust/sort.h>
#include <math.h>
#include <algorithm>
#include <numeric>
#include <random>
#include "timing.h"

#define ITEM_PER_THREAD 8

__global__ void gossKernel(int NumberCount, int topN, int randN, float* predictions, float* train, float* gradients, int* usedSet) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < NumberCount) {
        for (int i = index * ITEM_PER_THREAD; i < (index + 1) * ITEM_PER_THREAD; i++) {
            if (i >= NumberCount) {
                break;
            }
            gradients[i] = std::fabs(predictions[i] - train[i]);
        }
    }
}

/*__global__ void getUsedSet(int* usedSet, int* indices, int topN, int randN) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = index * ITEM_PER_THREAD; i < (index + 1) * ITEM_PER_THREAD; i++) {
        if (i > topN + randN) {
            break;
        }
        usedSet[i] = indices[i];
    }
}*/

void getUsedSet(int* usedSet, std::vector<int> &indices, std::vector<int> &randSet, int topN, int randN) {
    for (int i  = 0; i < topN; i++) {
        usedSet[i] = indices[i];
    }
    for (int i = topN; i < randN + topN; i++) {
        usedSet[i] = randSet[i - topN];
    }
}

void gossCuda(int NumberCount, int topN, int randN, float* predictions, float* train, int* usedSet) {
    const int threadsPerBlock = 512;
    const int blocks = (NumberCount / ITEM_PER_THREAD + threadsPerBlock - 1) / threadsPerBlock;

    float* gradients = (float*)calloc(NumberCount, sizeof(float));

    float* device_pred;
    float* device_train;
    float* device_grad;
    int* device_indices;
    int* device_usedset;

    Timer timer5;
    cudaMalloc(&device_pred, sizeof(float) * NumberCount);
    cudaMalloc(&device_train, sizeof(float) * NumberCount);
    cudaMalloc(&device_grad, sizeof(float) * NumberCount);
    cudaMalloc(&device_usedset, sizeof(int) * (topN + randN));

    cudaMemcpy(device_pred, predictions, sizeof(float) * NumberCount, cudaMemcpyHostToDevice);
    cudaMemcpy(device_train, train, sizeof(float) * NumberCount, cudaMemcpyHostToDevice);
    double t5 = timer5.elapsed();

    Timer timer1;
    gossKernel<<<blocks, threadsPerBlock>>>(NumberCount, topN, randN, device_pred, device_train, device_grad, device_usedset);
    cudaThreadSynchronize();
    double t1 = timer1.elapsed();

    Timer timer2;
    std::vector<int> indices(NumberCount);
    iota(indices.begin(), indices.end(), 0);
    cudaMemcpy(gradients, device_grad, sizeof(float) * NumberCount, cudaMemcpyDeviceToHost);
    thrust::sort_by_key(indices.data(), indices.data() + NumberCount, gradients);
    double t2 = timer2.elapsed();

    Timer timer3;
    std::vector<int> randSet;
    std::sample(indices.begin() + topN, indices.end(), std::back_inserter(randSet), randN, std::mt19937{std::random_device{}()});
    double t3 = timer3.elapsed();

    Timer timer4;
    getUsedSet(usedSet, indices, randSet, topN, randN);
    double t4 = timer4.elapsed();
    
    /* Timer timer7;
    curandGenerator_t gen;
    float* dev_rand;
    // float* rand = (float*)calloc(NumberCount - topN, sizeof(float));
    cudaMalloc(&dev_rand, (NumberCount - topN) * sizeof(float));
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, dev_rand, (NumberCount - topN));
    // cudaMemcpy(rand, dev_rand, (NumberCount - topN) * sizeof(float), cudaMemcpyDeviceToHost);
    double t7 = timer7.elapsed();

    Timer timer3;
    thrust::sort_by_key(indices.data() + topN, indices.data() + NumberCount, rand);
    double t3 = timer3.elapsed();

    Timer timer4;
    getUsedSet<<<blocks, threadsPerBlock>>>(device_usedset, device_indices, topN, randN);
    double t4 = timer4.elapsed();

    Timer timer6;
    cudaMemcpy(usedSet, device_usedset, sizeof(int) * (topN + randN), cudaMemcpyDeviceToHost);
    double t6 = timer6.elapsed(); */

    printf("TOTAL TIME  : %.6fs\n", t1 + t2 + t3 + t4 + t5);
    printf("Compute grad: %.6fs\n", t1);
    printf("Sort by grad: %.6fs\n", t2);
    printf("Sampling    : %.6fs\n", t3);
    printf("New dataset : %.6fs\n", t4);
    printf("Datamovement : %.6fs\n", t5);
}