#include "mpi.h"
#include "timing.h"
#include <random>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <queue>

std::vector<float> read_inputs(int NumberCount,int minimum, int maximum) {
    // use random inputs for now
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::vector<float> values(NumberCount); 
    std::uniform_real_distribution<> dis(minimum, maximum);
    std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
    return values;
}

void compute_L2_gradients(std::vector<float> &predictions, std::vector<float> &train, std::vector<float> &gradient) {
    int NumberCount = predictions.size();
    for (int i = 0; i < NumberCount; i++) {
        // omit the coefficient 2 (doesn't matter)
        // discard gradients with low abs value
        gradient[i] = std::fabs(predictions[i] - train[i]);
    } 
}

void getUsedSet(std::vector<int> &usedSet, std::vector<int> &indices, std::vector<int> &randSet, int topN, int randN) {
    for (int i  = 0; i < topN; i++) {
        usedSet[i] = indices[i];
    }
    for (int i = topN; i < randN + topN; i++) {
        usedSet[i] = randSet[i - topN];
    }
}

void kWayMerge(int k, int dataPerPartition, std::vector<int> &indicies, std::vector<float> &gradients, int displs[], int recvcounts[]) {
    std::vector<int> result(indicies.size());
    int curr_idx = 0;
    auto cmp = [](std::pair<float, int> left, std::pair<float, int> right) { return left.first < right.first; };
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >, decltype(cmp)> min_heap(cmp);
    std::vector<int> curr_indices(k);
    for (int i = 0; i < k; i++) {
        curr_indices[i] = displs[i] + 1;
        int index = indicies[displs[i]];
        min_heap.push(std::make_pair(gradients[index], index));
    }
    while (!min_heap.empty()) {
        std::pair<float, int> top = min_heap.top();
        min_heap.pop();
        result[curr_idx] = top.second;
        curr_idx++;
        int partitionIndex = top.second / dataPerPartition;
        partitionIndex = partitionIndex == k + 1 ? k : partitionIndex;
        if (curr_indices[partitionIndex] < displs[partitionIndex] + recvcounts[partitionIndex]) {
            int index = indicies[curr_indices[partitionIndex]];
            min_heap.push(std::make_pair(gradients[index], index));
            curr_indices[partitionIndex]++;
        }
    }
    indicies = result;
}

void kWayMerge2(int k, int dataPerPartition, std::vector<int> &indicies, std::vector<float> &gradients, std::vector<int> &splittingPoints) {
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
    int pid;
    int nproc;
    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes specificed at start of run
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int NumberCount = 100000;
    int minimum = 0, maximum = 1000000;
    float a = 0.2, b = 0.2;
    int topN = a * NumberCount, randN = b * NumberCount;

    // Assume regression
    std::vector<float> predictions(NumberCount);
    std::vector<float> train(NumberCount);

    if (pid == 0) {
        predictions = read_inputs(NumberCount, minimum, maximum);
        train = read_inputs(NumberCount, minimum, maximum);
    }
    Timer timer6;
    std::vector<float> gradients(NumberCount);

    int recvcounts[nproc];
    int displs[nproc];
    int itemPerNode = NumberCount / nproc;
    for (int i = 0; i < nproc; i++) {
      displs[i] = i * itemPerNode;
      recvcounts[i] = itemPerNode;
    }
    recvcounts[nproc-1] = NumberCount - (nproc-1) * itemPerNode;

    std::vector<float> private_pred(recvcounts[pid]);
    std::vector<float> private_train(recvcounts[pid]);
    std::vector<float> private_grad(recvcounts[pid]);
    double t6 = timer6.elapsed();

    Timer timer1;
    MPI_Scatterv(static_cast<void*>(predictions.data()), recvcounts, displs,
                 MPI_FLOAT, static_cast<void*>(private_pred.data()), recvcounts[pid],
                 MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(static_cast<void*>(train.data()), recvcounts, displs,
                 MPI_FLOAT, static_cast<void*>(private_train.data()), recvcounts[pid],
                 MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    compute_L2_gradients(private_pred, private_train, private_grad);

    MPI_Gatherv(static_cast<void*>(private_grad.data()), recvcounts[pid], MPI_FLOAT,
                static_cast<void*>(gradients.data()), recvcounts, displs,
                MPI_FLOAT, 0, MPI_COMM_WORLD);
    double t1 = timer1.elapsed();
    
    Timer timer2;
    std::vector<int> indices(NumberCount);
    /* std::vector<int> priv_indices(recvcounts[pid]);
    iota(priv_indices.begin(), priv_indices.end(), displs[pid]);
    int begin = displs[pid];
    std::stable_sort(priv_indices.begin(), priv_indices.end(), [&](size_t i1, size_t i2) {return private_grad[i1 - begin] > private_grad[i2 - begin];}); */

    int numPartition = 5;
    int dataPerPartition = recvcounts[pid] / numPartition;
    int begin = displs[pid];
    std::vector<int> splittingPoints(numPartition + 1);
    for (int i = 0; i < numPartition; i++) {
        splittingPoints[i] = i * dataPerPartition;
    }
    splittingPoints[numPartition] = recvcounts[pid];
    std::vector<int> priv_indices(recvcounts[pid]);
    #pragma omp parallel for schedule(static) 
    for (int i = 0; i < numPartition; i++) {
        iota(priv_indices.begin() + splittingPoints[i], priv_indices.begin() + splittingPoints[i+1], splittingPoints[i]);
        std::stable_sort(priv_indices.begin() + splittingPoints[i], priv_indices.begin() + splittingPoints[i+1], [&](size_t i1, size_t i2) {return private_grad[i1 - begin] > private_grad[i2 - begin];});
    }
    kWayMerge2(numPartition, dataPerPartition, priv_indices, private_grad, splittingPoints);
    double t2 = timer2.elapsed();
    Timer timer7;
    MPI_Gatherv(static_cast<void*>(priv_indices.data()), recvcounts[pid], MPI_INT,
                static_cast<void*>(indices.data()), recvcounts, displs,
                MPI_INT, 0, MPI_COMM_WORLD);
    double t7 = timer7.elapsed();

    if (pid == 0) {

        Timer timer3;
        kWayMerge(nproc, itemPerNode, indices, gradients, displs, recvcounts);
        // iota(indices.begin(), indices.end(), 0);
        // std::stable_sort(indices.begin(), indices.end(), [&gradients](size_t i1, size_t i2) {return gradients[i1] > gradients[i2];});
        double t3 = timer3.elapsed();
        check_correctness(indices, gradients, NumberCount);
        
        Timer timer4;
        std::vector<int> randSet;
        std::vector<int> usedSet(topN + randN);
        std::sample(indices.begin() + topN, indices.end(), std::back_inserter(randSet), randN, std::mt19937{std::random_device{}()});
        double t4 = timer4.elapsed();
        
        Timer timer5;
        getUsedSet(usedSet, indices, randSet, topN, randN);
        double t5 = timer5.elapsed();

        printf("TOTAL TIME  : %.6fs\n", t1 + t2 + t3 + t4 + t5 + t6 + t7);
        printf("Compute grad: %.6fs\n", t1);
        printf("Sort by grad: %.6fs\n", t2);
        printf("gather sorted inputs : %.6fs\n", t7);
        printf("k way merge: %.6fs\n", t3);
        printf("Sampling    : %.6fs\n", t4);
        printf("New dataset : %.6fs\n", t5);
        printf("Other computation : %.6fs\n", t6);
    }

    MPI_Finalize();
}
