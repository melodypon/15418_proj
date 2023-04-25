#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_functions.h>
#include <thrust/sort.h>
#include <math.h>
#include <algorithm>
#include <numeric>
#include "timing.h"

#define THREADS 512

__global__ void count_non_zero(float *features, int num_data, int *count, float *maxs) {
    int start = blockIdx.x * num_data;
    int index = threadIdx.x;
    int dataPerThread = num_data / THREADS;
    int conflicts = 0;
    float max_value = 0;

    for (int i = start + index * dataPerThread; i < start + (index + 1) * dataPerThread; i++) {
        if (i - start < num_data && features[i] != 0.0f) {
			conflicts++;
            if (features[i] > max_value) {
                max_value = features[i];
            }
		}
    }

    count[blockIdx.x] += conflicts;
    if (max_value > maxs[blockIdx.x]){
        maxs[blockIdx.x] = max_value;
    }
}

__global__ void merge_features(float *features, float *new_features, int num_features, int num_new_feature, int num_data, float *bin_range, int* bundle_map) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int row = index; row < index + 1; row++) {
        if (row > num_data) {
            break;
        }

        for (int col = 0; col < num_features; col++) {
            if (features[col * num_data + row] != 0) {
				new_features[bundle_map[col] * num_data + row] += features[col * num_data + row] + bin_range[bundle_map[col]];
			}
        }
    }
}

void bundle_features(std::vector<int> &order, int max_conflict, int *counts, std::vector<std::vector<int> > &bundles, std::vector<int> &bundle_map) {
	int num_features = order.size();
	std::vector<int> bundle_conflict_counts;
	for (int i = 0; i < num_features; i++) {
		bool need_new = true;
		for (int j = 0; j < bundles.size(); j++) {
            int count = counts[i];
			if (count + bundle_conflict_counts[j] <= max_conflict) {
				bundles[j].push_back(i);
				bundle_conflict_counts[j] = count + bundle_conflict_counts[j];
				need_new = false;
                bundle_map[i] = j;
				break;
			}
		}
		if (need_new) {
			std::vector<int> new_bundle;
			new_bundle.push_back(i);
			bundles.push_back(new_bundle);
			bundle_conflict_counts.push_back(counts[i]);
            bundle_map[i] = bundles.size() - 1;
		}
	}
}

void efbCuda(std::vector<std::vector<float> > features, int max_conflict) {
    int num_features = features.size();
    int num_data = features[0].size();

    // Count non zeros
    int* count = (int*)calloc(num_features, sizeof(int));
    float* maxs = (float*)calloc(num_features, sizeof(float));
    float* device_features;
    int* device_count;
    float* device_maxs;
    cudaMalloc(&device_features, sizeof(float) * num_features * num_data);
    cudaMalloc(&device_count, sizeof(int) * num_features);
    cudaMalloc(&device_maxs, sizeof(float) * num_features);
    cudaMemcpy(device_features, features.data(), sizeof(float) * num_features * num_data, cudaMemcpyHostToDevice);

    Timer timer1;
    count_non_zero<<<num_features, THREADS>>>(device_features, num_data, device_count, device_maxs);
    cudaThreadSynchronize();
    double t1 = timer1.elapsed();

    cudaMemcpy(count, device_count, sizeof(int) * num_features, cudaMemcpyDeviceToHost);
    cudaMemcpy(maxs, device_maxs, sizeof(float) * num_features, cudaMemcpyDeviceToHost);

    // Sort non zeros
    Timer timer2;
    std::vector<int> order(num_features);
    iota(order.begin(), order.end(), 0);
    thrust::sort_by_key(order.data(), order.data() + num_features, count);
    double t2 = timer2.elapsed();

    // Bundle features
	Timer timer3;
	std::vector<std::vector<int> > bundles;
    std::vector<int> bundle_map(num_features);
	bundle_features(order, max_conflict, count, bundles, bundle_map);
	double t3 = timer3.elapsed();

    float* bin_range = (float*)calloc(num_features, sizeof(float));
    for (auto F: bundles) {
        float total_bin = 0;
        for (auto f: F) {
            total_bin += maxs[f];
            bin_range[f] = total_bin;
        }
    }

    // Merge features
    int num_new_feature = bundles.size();
    float* new_features = (float*)calloc(num_new_feature * num_data, sizeof(float));
    float* device_new_features;
    float* device_bin_range;
    int *device_bundle_map;
    cudaMalloc(&device_features, sizeof(float) * num_features * num_data);
    cudaMalloc(&device_new_features, sizeof(float) * num_new_feature * num_data);
    cudaMalloc(&device_bin_range, sizeof(float) * num_features);
    cudaMalloc(&device_bundle_map, sizeof(int) * num_features);
    cudaMemcpy(device_features, features.data(), sizeof(float) * num_features * num_data, cudaMemcpyHostToDevice);
    cudaMemcpy(device_bin_range, bin_range, sizeof(float) * num_features, cudaMemcpyHostToDevice);
    cudaMemcpy(device_bundle_map, bundle_map.data(), sizeof(int) * num_features, cudaMemcpyHostToDevice);

    Timer timer4;
    const int blocks = (num_features * num_data + THREADS - 1) / THREADS;
	merge_features<<<blocks, THREADS>>>(device_features, device_new_features, num_features, num_new_feature, num_data, device_bin_range, device_bundle_map);
	cudaThreadSynchronize();
    double t4 = timer4.elapsed();

    cudaMemcpy(new_features, device_new_features, sizeof(float) * num_features * num_data, cudaMemcpyDeviceToHost);

    printf("TOTAL TIME : %.6fs\n", t1 + t2 + t3 + t4);
	printf("Build graph: %.6fs\n", t1);
	printf("Sort order : %.6fs\n", t2);
	printf("Bundle feat: %.6fs\n", t3);
	printf("Merge feat : %.6fs\n", t4);
}