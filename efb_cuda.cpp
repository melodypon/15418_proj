#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <numeric>
#include "timing.h"

void efbCuda(std::vector<std::vector<float> > features, int max_conflict);

void read_features(std::vector<std::vector<float> > &features) {
	// Flight delay
	for (int i = 0; i < 617; i++) {
		features.push_back(std::vector<float>());
	}
	std::string line;
	std::ifstream infile("part-00000-e36056e8-fe85-4a72-b3ec-9e9d5deb5cf8-c000.csv");

	int count = 0;
	while(count < 30000) {
		getline(infile, line);
		std::size_t found = 0;
		int i = 0;
		features[i].push_back(std::stof(line.substr(found, found + 3)));
		i++;
		while (found != std::string::npos) {
			found = line.find(",", found + 1);
			features[i].push_back(std::stof(line.substr(found + 1, found + 4)));
	        i++;
	    }
		count++;
    }
    std::cout << features.size() << std::endl;
    std::cout << features[0].size() << std::endl;
}

int main(int argc, char* argv[]) {
	int max_conflict = 10000;
	std::vector<std::vector<float> > features;
	read_features(features);

    efbCuda(features, max_conflict);
}