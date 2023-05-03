#include <random>
#include <math.h>
#include <algorithm>
#include <iostream>
#include "timing.h"

void gossCuda(int NumberCount, int topN, int randN, float* predictions, float* train, int* usedSet);

std::vector<float> read_inputs(int NumberCount,int minimum, int maximum) {
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::vector<float> values(NumberCount); 
    std::uniform_real_distribution<> dis(minimum, maximum);
    std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
    return values;
}

int main(int argc, char* argv[]) {
    int NumberCount = 10000000;
    int minimum = 0, maximum = 1000000;
    float a = 0.2, b = 0.2;
    int topN = a * NumberCount, randN = b * NumberCount;

    // Assume regression
    std::vector<float> predictions = read_inputs(NumberCount, minimum, maximum);
    std::vector<float> train = read_inputs(NumberCount, minimum, maximum);
    std::vector<int> usedSet(randN + topN, 0);

    gossCuda(NumberCount, topN, randN, predictions.data(), train.data(), usedSet.data());
}
