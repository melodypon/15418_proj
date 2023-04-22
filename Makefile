all:	efb_serial efb_openmp goss_serial goss_openmp goss_mpi

efb_serial:	efb_serial.cpp
	g++ -o efb_serial -std=c++14 -fvisibility=hidden -lpthread -O2 efb_serial.cpp

efb_openmp:	efb_openmp.cpp
	g++ -o efb_openmp -std=c++14 -fvisibility=hidden -lpthread -O2 -fopenmp efb_openmp.cpp

goss_serial:	goss_serial.cpp
	g++ -o goss_serial -std=c++17 -fvisibility=hidden -lpthread -O2 goss_serial.cpp

goss_openmp:	goss_openMP.cpp
	g++ -o goss_openmp -std=c++17 -fvisibility=hidden -lpthread -O2 -fopenmp goss_openMP.cpp	

goss_mpi:	goss_mpi.cpp
	mpic++ -o goss_mpi -std=c++17 -fvisibility=hidden -lpthread -Wall -Wextra goss_mpi.cpp

clean:
	$(RM) efb_serial efb_openmp goss_serial goss_openmp goss_mpi