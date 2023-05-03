#include <random>
#include <math.h>
#include <algorithm>
#include <iostream>
#include "timing.h"

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
    int NumberCount = 100000;
    int minimum = 0, maximum = 1000000;
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
    std::vector<int> indices(NumberCount);
    iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&gradients](size_t i1, size_t i2) {return gradients[i1] > gradients[i2];});
    double t2 = timer2.elapsed();

    Timer timer3;
    std::vector<int> randSet;
    std::vector<int> usedSet(topN + randN);
    std::sample(indices.begin() + topN, indices.end(), std::back_inserter(randSet), randN, std::mt19937{std::random_device{}()});
    double t3 = timer3.elapsed();

    Timer timer4;
    getUsedSet(usedSet, indices, randSet, topN, randN);
    double t4 = timer4.elapsed();
    /* for (int i = 0; i < NumberCount; i++) {
        std::cout << i << " " << gradients[i] << " " << indices[i] << '\n';
    }
    for (auto x : usedSet) {
        std::cout << x << ' ';
    }
    std::cout << "\n"; */

    printf("TOTAL TIME  : %.6fs\n", t1 + t2 + t3 + t4);
    printf("Compute grad: %.6fs\n", t1);
    printf("Sort by grad: %.6fs\n", t2);
    printf("Sampling    : %.6fs\n", t3);
    printf("New dataset : %.6fs\n", t4);
}
