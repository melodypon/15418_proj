all:	efb_serial efb_openmp efb_mpi goss_serial goss_openmp goss_mpi goss_cuda

efb_serial:	efb_serial.cpp
	g++ -o efb_serial -std=c++14 -fvisibility=hidden -lpthread -O2 efb_serial.cpp

efb_openmp:	efb_openmp.cpp
	g++ -o efb_openmp -std=c++14 -fvisibility=hidden -lpthread -O2 -fopenmp efb_openmp.cpp

efb_mpi:	efb_mpi.cpp
	mpic++ -o efb_mpi -std=c++14 -fvisibility=hidden -lpthread -Wall -Wextra efb_mpi.cpp

goss_cuda:	goss_cuda_main.o goss_cuda_cuda.o
	g++ -m64 -O3 -Wall -o goss_cuda goss_cuda_main.o goss_cuda_cuda.o -L/usr/local/cuda-11.7/lib64/ -lcudart -lcurand 

goss_cuda_main.o:	goss_cuda.cpp
	g++ -m64 goss_cuda.cpp -O3 -Wall -c -o goss_cuda_main.o

goss_cuda_cuda.o:	goss_cuda.cu
	nvcc -O3 -m64 goss_cuda.cu --gpu-architecture compute_61 -ccbin /usr/bin/gcc -c -o goss_cuda_cuda.o

goss_serial:	goss_serial.cpp
	g++ -o goss_serial -std=c++17 -fvisibility=hidden -lpthread -O2 goss_serial.cpp

goss_openmp:	goss_openMP.cpp
	g++ -o goss_openmp -std=c++17 -fvisibility=hidden -lpthread -O2 -fopenmp goss_openMP.cpp	

goss_mpi:	goss_mpi.cpp
	mpic++ -o goss_mpi -std=c++17 -fvisibility=hidden -lpthread -Wall -Wextra goss_mpi.cpp

clean:
	$(RM) efb_serial efb_openmp efb_mpi goss_serial goss_openmp goss_mpi goss_cuda