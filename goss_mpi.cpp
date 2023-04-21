#include "mpi.h"
#include <random>
#include <math.h>
#include <algorithm>
#include <iostream>

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

int main(int argc, char* argv[]) {
    int pid;
    int nproc;
    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes specificed at start of run
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    int NumberCount = 40000;
    int minimum = 0, maximum = 10;
    float a = 0.2, b = 0.2;
    int topN = a * NumberCount, randN = b * NumberCount;

    // Assume regression
    std::vector<float> predictions(NumberCount);
    std::vector<float> train(NumberCount);

    if (pid == 0) {
        predictions = read_inputs(NumberCount, minimum, maximum);
        train = read_inputs(NumberCount, minimum, maximum);
    }
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

    if (pid == 0) {
        std::vector<int> indices(NumberCount);
        iota(indices.begin(), indices.end(), 0);
        std::stable_sort(indices.begin(), indices.end(), [&gradients](size_t i1, size_t i2) {return gradients[i1] > gradients[i2];});
        std::vector<int> randSet;
        std::vector<int> usedSet(topN + randN);
        std::sample(indices.begin() + topN, indices.end(), std::back_inserter(randSet), randN, std::mt19937{std::random_device{}()});
        getUsedSet(usedSet, indices, randSet, topN, randN);
    }

    MPI_Finalize();
}
