#include <random>
#include <math.h>
#include <algorithm>
#include <iostream>

std::vector<float> read_inputs(int NumberCount,int minimum, int maximum) {
    // used random inputs for now
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::vector<float> values(NumberCount); 
    std::uniform_real_distribution<> dis(minimum, maximum);
    std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
    return values;
}

void compute_L2_gradients(std::vector<float> &predictions, std::vector<float> &train, std::vector<float> &gradient) {
    int NumberCount = predictions.size();
    #pragma omp parallel for schedule(static) 
    for (int i = 0; i < NumberCount; i++) {
        // omit the coefficient 2 (doesn't matter)
        // discard gradients with low abs value
        gradient[i] = std::fabs(predictions[i] - train[i]);
    } 
}

void getUsedSet(std::vector<int> &usedSet, std::vector<int> &indices, std::vector<int> &randSet, int topN, int randN) {
    #pragma omp parallel for schedule(static) 
    for (int i  = 0; i < topN; i++) {
        usedSet[i] = indices[i];
    }
    #pragma omp parallel for schedule(static) 
    for (int i = topN; i < randN + topN; i++) {
        usedSet[i] = randSet[i - topN];
    }
}

int main(int argc, char* argv[]) {
    int NumberCount = 10;
    int minimum = 0, maximum = 10;
    float a = 0.1, b = 0.1;
    int topN = a * NumberCount, randN = b * NumberCount;
    // Assume regression
    std::vector<float> predictions = read_inputs(NumberCount, minimum, maximum);
    std::vector<float> train = read_inputs(NumberCount, minimum, maximum);
    std::vector<float> gradients(NumberCount);
    compute_L2_gradients(predictions, train, gradients);
    std::vector<int> indices(NumberCount);
    iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&gradients](size_t i1, size_t i2) {return gradients[i1] > gradients[i2];});
    std::vector<int> randSet;
    std::vector<int> usedSet(topN + randN);
    std::sample(indices.begin() + topN, indices.end(), std::back_inserter(randSet), randN, std::mt19937{std::random_device{}()});
    getUsedSet(usedSet, indices, randSet, topN, randN);
    /* for (int i = 0; i < NumberCount; i++) {
        std::cout << i << " " << gradients[i] << " " << indices[i] << '\n';
    }
    for (auto x : usedSet) {
        std::cout << x << ' ';
    }
    std::cout << "\n"; */
}