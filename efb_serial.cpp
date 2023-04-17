#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "timing.h"

void read_features(std::vector<std::vector<float> > &features) {
	// Simple test case
	/*
	std::vector<int> f1 = {1,2,3,0,0,0};
	std::vector<int> f2 = {0,0,0,1,2,2};
	std::vector<int> f3 = {1,2,3,0,0,0};
	features = {f1, f2, f3};
	*/

	// LETOR 4.0 Datasets
	for (int i = 0; i < 46; i++) {
		features.push_back(std::vector<float>());
	}
	std::string line;
	std::ifstream infile("Querylevelnorm.txt");

	while(getline(infile, line)) {
		std::size_t found = 0;
		int i = 0;
		while (found != std::string::npos) {
			found = line.find(":", found + 1);
			features[i].push_back(std::stof(line.substr(found + 1, found + 9)));
	        i++;
	    }
    }
    std::cout << features.size() << std::endl;
    std::cout << features[0].size() << std::endl;
}

int count_conflict(std::vector<float> &f1, std::vector<float> &f2) {
	int size = f1.size();
	int conflict = 0;
	for (int i = 0; i < size; i++) {
		if (f1[i] != 0.0f || f2[i] != 0.0f) {
			conflict++;
		}
	}
	return conflict;
}

int count_non_zero(std::vector<float> &f1) {
	int size = f1.size();
	int conflict = 0;
	for (int i = 0; i < size; i++) {
		if (f1[i] != 0.0f) {
			conflict++;
		}
	}
	return conflict;
}

void build_graph(std::vector<std::vector<int> > &graph, std::vector<std::vector<float> > &features) {
	int num_features = features.size();

	for (int i = 0; i < num_features; i++) {
		for (int j = i + 1; j < num_features; j++) {
			int conflict = count_conflict(features[i], features[j]);
			graph[i][j] = conflict;
			graph[j][i] = conflict;
		}
	}	
}

void sort_order(std::vector<std::vector<int> > &graph, std::vector<int> &order) {
	int num_features = graph.size();

	std::vector<std::pair<int, int> > list;
	for (int i = 0; i < num_features; i++) {
		int sum = 0;
		for (int j = 0; j < num_features; j++) {
			sum += graph[i][j];
		}
		list.push_back(std::pair<int, int>(sum, i));
	}

	sort(list.begin(), list.end(), std::greater<std::pair<int, int>>());

	for (int i = 0; i < num_features; i++) {
		order.push_back(list[i].first);
	}
}

int union_conflicts(std::vector<float> &conflicts, std::vector<float> &features) {
	int size = conflicts.size();
	int count = 0;

	for (int i = 0; i < size; i++) {
		conflicts[i] = (conflicts[i] != 0.0f || features[i] != 0.0f) ? 1.0f : 0.0f;
		count += conflicts[i];
	}

	return count;
}

void bundle_features(std::vector<int> &order, int max_conflict, std::vector<std::vector<float> > &features, std::vector<std::vector<int> > &bundles) {
	int num_features = order.size();
	std::vector<std::vector<float> > bundle_conflicts;
	std::vector<int> bundle_conflict_counts;
	for (int i = 0; i < num_features; i++) {
		bool need_new = true;
		for (int j = 0; j < bundles.size(); j++) {
			int count = count_conflict(features[i], bundle_conflicts[j]);
			if (count <= max_conflict) {
				bundles[j].push_back(i);
				bundle_conflict_counts[j] = union_conflicts(bundle_conflicts[j], features[i]);
				need_new = false;
				break;
			}
		}
		if (need_new) {
			std::vector<int> new_bundle;
			new_bundle.push_back(i);
			bundles.push_back(new_bundle);
			std::vector<float> conflicts(features[i]);
			bundle_conflicts.push_back(conflicts);
			bundle_conflict_counts.push_back(count_non_zero(features[i]));
		}
	}
	// for (auto i: bundle_conflict_counts) {
	// 	std::cout << i << " ";
	// }
	// std::cout << std::endl;
}

float num_of_bin(std::vector<float> &feature) {
	float max = 0.0f;
	for (auto i: feature) {
		max = (i > max) ? i : max;
	}
	return max;
}

void merge_features(std::vector<std::vector<float> > &features, std::vector<std::vector<int> > &bundles, std::vector<std::vector<float> > &new_features) {
	int num_data = features[0].size();
	
	for (auto F: bundles) {
		std::vector<float> bin_range = {0};
		float total_bin = 0;
		for (auto f: F) {
			total_bin += num_of_bin(features[f]);
			bin_range.push_back(total_bin);
		}
		std::vector<float> new_feature;
		for (int i = 0; i < num_data; i++) {
			float new_bin = 0;
			for (int j = 0; j < F.size(); j++) {
				int f = F[j];
				if (features[f][i] != 0) {
					new_bin += features[f][i] + bin_range[j];
				}
			}
			new_feature.push_back(new_bin);
		}
		new_features.push_back(new_feature);
	}
}

int main(int argc, char* argv[]) {
	int max_conflict = 10000;
	std::vector<std::vector<float> > features;
	read_features(features);
	int num_features = features.size();

	// Greedy Bundle
	Timer timer1;
	std::vector<std::vector<int> > graph(num_features, std::vector<int>(num_features));
	build_graph(graph, features);
	double t1 = timer1.elapsed();

	Timer timer2;
	std::vector<int> order;
	sort_order(graph, order);
	double t2 = timer2.elapsed();

	Timer timer3;
	std::vector<std::vector<int> > bundles;
	bundle_features(order, max_conflict, features, bundles);
	double t3 = timer3.elapsed();

	// for (auto i: bundles) {
	// 	std::cout << "bundle: ";
	// 	for (auto j: i) {
	// 		std::cout << j << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// Merge Exclusive Features
	Timer timer4;
	std::vector<std::vector<float> > new_features;
	merge_features(features, bundles, new_features);
	double t4 = timer4.elapsed();

	std::cout << new_features.size() << std::endl;
	std::cout << new_features[0].size() << std::endl;

	printf("TOTAL TIME : %.6fs\n", t1 + t2 + t3 + t4);
	printf("Build graph: %.6fs\n", t1);
	printf("Sort order : %.6fs\n", t2);
	printf("Bundle feat: %.6fs\n", t3);
	printf("Merge feat : %.6fs\n", t4);

	// for (auto i: new_features) {
	// 	std::cout << "feature: ";
	// 	for (auto j: i) {
	// 		std::cout << j << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
}