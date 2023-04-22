#include "mpi.h"
#include "timing.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <numeric>

void read_features(std::vector<std::vector<float> > &features) {
	// Simple test case
	/*
	std::vector<int> f1 = {1,2,3,0,0,0};
	std::vector<int> f2 = {0,0,0,1,2,2};
	std::vector<int> f3 = {1,2,3,0,0,0};
	features = {f1, f2, f3};
	*/

	// LETOR 4.0 Datasets
	// for (int i = 0; i < 46; i++) {
	// 	features.push_back(std::vector<float>());
	// }
	// std::string line;
	// std::ifstream infile("Querylevelnorm.txt");

	// while(getline(infile, line)) {
	// 	std::size_t found = 0;
	// 	int i = 0;
	// 	while (found != std::string::npos) {
	// 		found = line.find(":", found + 1);
	// 		features[i].push_back(std::stof(line.substr(found + 1, found + 9)));
	//         i++;
	//     }
    // }
    // std::cout << features.size() << std::endl;
    // std::cout << features[0].size() << std::endl;

	// Flight delay
	for (int i = 0; i < 617; i++) {
		features.push_back(std::vector<float>());
	}
	std::string line;
	std::ifstream infile("part-00000-e36056e8-fe85-4a72-b3ec-9e9d5deb5cf8-c000.csv");

	int count = 0;
	while(count < 20000) {
		getline(infile, line);
		std::size_t found = 0;
		int i = 0;
		features[i].push_back(std::stof(line.substr(found, found + 3)));
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
    int pid;
    int nproc;
    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes specificed at start of run
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	int max_conflict = 10000;

    // Read features
	std::vector<std::vector<float> > features;
    int num_features;
    int num_data;
    if (pid == 0) {
        read_features(features);
        num_features = features.size();
        num_data = features[0].size();
    }
    MPI_Bcast(&num_features, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (pid != 0) {
        for (int i = 0; i < num_features; i++) {
            features.push_back(std::vector<float>(num_data));
        }
    }

    // Calculation for MPI
    int recvcounts[nproc];
    int displs[nproc];
    int itemPerNode = num_data / nproc;
    for (int i = 0; i < nproc; i++) {
      displs[i] = i * itemPerNode;
      recvcounts[i] = itemPerNode;
    }
    recvcounts[nproc-1] = num_data - (nproc-1) * itemPerNode;

    std::vector<std::vector<float> > priv_features;
    for (int i = 0; i < num_features; i++) {
        std::vector<float> feature(recvcounts[pid]);
        MPI_Scatterv(static_cast<void*>(features[i].data()), recvcounts, displs,
                     MPI_FLOAT, static_cast<void*>(feature.data()), recvcounts[pid],
                     MPI_FLOAT, 0, MPI_COMM_WORLD);
        priv_features.push_back(feature);
    }

	// Greedy Bundle
	Timer timer1;
    std::vector<int> non_zeros(num_features);
    for (int i = 0; i < num_features; i++) {
        non_zeros[i] = count_non_zero(priv_features[i]);
    }
    std::vector<int> all_count(num_features * nproc);
    MPI_Gather(static_cast<void*>(non_zeros.data()), num_features, MPI_INT,
               static_cast<void*>(all_count.data()), num_features, MPI_INT,
               0, MPI_COMM_WORLD);
    if (pid == 0) {
        for (int i = 0; i < num_features; i++) {
            for (int j = 1; j < nproc; j++) {
                non_zeros[i] += all_count[j * num_features + i];
            }
        }
    }
	double t1 = timer1.elapsed();

    Timer timer2;
    std::vector<int> order(num_features);
    if (pid == 0) {
        iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&non_zeros](size_t i1, size_t i2) {return non_zeros[i1] > non_zeros[i2];});
    }
    double t2 = timer2.elapsed();

    Timer timer3;
    int num_bundles;
    std::vector<std::vector<int> > bundles;
    if (pid == 0) {
        bundle_features(order, max_conflict, features, bundles);
        num_bundles = bundles.size();
    }
    double t3 = timer3.elapsed();

    MPI_Bcast(&num_bundles, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> bundle_size(num_bundles);
    std::vector<int> all_bundle(num_features);
    if (pid == 0) {
        int count = 0;
        for (int i = 0; i < num_bundles; i++) {
            bundle_size[i] = bundles[i].size();
            for (int j = 0; j < bundle_size[i]; j++) {
                all_bundle[count] = bundles[i][j];
                count++;
            }
        }
    }
    MPI_Bcast(bundle_size.data(), num_bundles, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_bundle.data(), num_features, MPI_INT, 0, MPI_COMM_WORLD);
    if (pid != 0) {
        int count = 0;
        for (int i = 0; i < num_bundles; i++) {
            std::vector<int> bundle;
            for (int j = 0; j < bundle_size[i]; j++) {
                bundle.push_back(all_bundle[count]);
                count++;
            }
            bundles.push_back(bundle);
        }
    }

    // Merge Exclusive Features
    Timer timer4;
	std::vector<std::vector<float> > priv_new_features;
	merge_features(priv_features, bundles, priv_new_features);

    std::vector<std::vector<float> > new_features;
    for (int i = 0; i < num_bundles; i++) {
        new_features.push_back(std::vector<float>(num_data));
    }
    for (int i = 0; i < num_bundles; i++) {
        MPI_Gatherv(static_cast<void*>(new_features[i].data()), recvcounts[pid], MPI_FLOAT,
                    static_cast<void*>(new_features[i].data()), recvcounts, displs,
                    MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
	double t4 = timer4.elapsed();

    if (pid == 0) {
        std::cout << new_features.size() << std::endl;
        std::cout << new_features[0].size() << std::endl;

        printf("TOTAL TIME : %.6fs\n", t1 + t2 + t3 + t4);
        printf("Build graph: %.6fs\n", t1);
        printf("Sort order : %.6fs\n", t2);
        printf("Bundle feat: %.6fs\n", t3);
        printf("Merge feat : %.6fs\n", t4);
    }

    MPI_Finalize();
}