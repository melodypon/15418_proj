#include <random>
#include <math.h>
#include <algorithm>
#include <iostream>
#include "timing.h"
#include <queue>

std::vector<float> read_inputs(int NumberCount,int minimum, int maximum) {
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::vector<float> values(NumberCount); 
    std::uniform_real_distribution<> dis(minimum, maximum);
    std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
    return values;
}

void compute_L2_gradients(std::vector<float> &predictions, std::vector<float> &train, std::vector<float> &gradient) {
    int NumberCount = predictions.size();
    
    #pragma omp parallel for schedule(dynamic, 1024) 
    for (int i = 0; i < NumberCount; i++) {
        // omit the coefficient 2 (doesn't matter)
        // discard gradients with low abs value
        gradient[i] = std::fabs(predictions[i] - train[i]);
    } 
}

void getUsedSet(std::vector<int> &usedSet, std::vector<int> &indices, std::vector<int> &randSet, int topN, int randN) {
    #pragma omp parallel for schedule(dynamic, 1024) 
    for (int i  = 0; i < topN; i++) {
        usedSet[i] = indices[i];
    }
    #pragma omp parallel for schedule(dynamic, 1024) 
    for (int i = topN; i < randN + topN; i++) {
        usedSet[i] = randSet[i - topN];
    }
}

void kWayMerge(int k, int dataPerPartition, std::vector<int> &indicies, std::vector<float> &gradients, std::vector<int> &splittingPoints) {
    std::vector<int> result(indicies.size());
    int curr_idx = 0;
    auto cmp = [](std::pair<float, int> left, std::pair<float, int> right) { return left.first < right.first; };
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >, decltype(cmp)> min_heap(cmp);
    std::vector<int> curr_indices(k);
    for (int i = 0; i < k; i++) {
        curr_indices[i] = splittingPoints[i] + 1;
        int index = indicies[splittingPoints[i]];
        min_heap.push(std::make_pair(gradients[index], index));
    }
    while (!min_heap.empty()) {
        std::pair<float, int> top = min_heap.top();
        min_heap.pop();
        result[curr_idx] = top.second;
        curr_idx++;
        int partitionIndex = top.second / dataPerPartition;
        partitionIndex = partitionIndex == k + 1 ? k : partitionIndex;
        if (curr_indices[partitionIndex] < splittingPoints[partitionIndex + 1]) {
            int index = indicies[curr_indices[partitionIndex]];
            min_heap.push(std::make_pair(gradients[index], index));
            curr_indices[partitionIndex]++;
        }
    }
    indicies = result;
}
void check_correctness(std::vector<int> &indicies, std::vector<float> &gradients, int NumberCount) {
    std::vector<int> answer_indices(NumberCount);
    iota(answer_indices.begin(), answer_indices.end(), 0);
    std::stable_sort(answer_indices.begin(), answer_indices.end(), [&gradients](size_t i1, size_t i2) {return gradients[i1] > gradients[i2];});
    for (int i = 0; i < NumberCount; i++) {
        if (answer_indices[i] != indicies[i]) {
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    int NumberCount = 400000;
    int minimum = 0, maximum = 1000;
    float a = 0.2, b = 0.2;
    int topN = a * NumberCount, randN = b * NumberCount;
    // Assume regression
    std::vector<float> predictions = read_inputs(NumberCount, minimum, maximum);
    std::vector<float> train = read_inputs(NumberCount, minimum, maximum);

    Timer timer1;
    std::vector<float> gradients(NumberCount);
    compute_L2_gradients(predictions, train, gradients);
    double t1 = timer1.elapsed();
    
    Timer timer2;
    int numPartition = 5;
    int dataPerPartition = NumberCount / numPartition;
    std::vector<int> splittingPoints(numPartition + 1);
    for (int i = 0; i < numPartition; i++) {
        splittingPoints[i] = i * dataPerPartition;
    }
    splittingPoints[numPartition] = NumberCount;
    std::vector<int> indices(NumberCount);
    #pragma omp parallel for schedule(static) 
    for (int i = 0; i < numPartition; i++) {
        iota(indices.begin() + splittingPoints[i], indices.begin() + splittingPoints[i+1], splittingPoints[i]);
        std::stable_sort(indices.begin() + splittingPoints[i], indices.begin() + splittingPoints[i+1], [&](size_t i1, size_t i2) {return gradients[i1] > gradients[i2];});
    }
    kWayMerge(numPartition, dataPerPartition, indices, gradients, splittingPoints);
    // iota(indices.begin(), indices.end(), 0);
    // std::stable_sort(indices.begin(), indices.end(), [&gradients](size_t i1, size_t i2) {return gradients[i1] > gradients[i2];});
    double t2 = timer2.elapsed();
    check_correctness(indices, gradients, NumberCount);

    Timer timer3;
    std::vector<int> randSet;
    std::vector<int> usedSet(topN + randN);
    std::sample(indices.begin() + topN, indices.end(), std::back_inserter(randSet), randN, std::mt19937{std::random_device{}()});
    double t3 = timer3.elapsed();

    Timer timer4;
    getUsedSet(usedSet, indices, randSet, topN, randN);
    double t4 = timer4.elapsed();
    
    printf("TOTAL TIME  : %.6fs\n", t1 + t2 + t3 + t4);
    printf("Compute grad: %.6fs\n", t1);
    printf("Sort by grad: %.6fs\n", t2);
    printf("Sampling    : %.6fs\n", t3);
    printf("New dataset : %.6fs\n", t4);
}
