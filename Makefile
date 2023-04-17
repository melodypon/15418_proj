all:efb_serial efb_openmp
	g++ -o efb_serial -std=c++14 -fvisibility=hidden -lpthread -O2 -fopenmp efb_serial.cpp
	g++ -o efb_openmp -std=c++14 -fvisibility=hidden -lpthread -O2 -fopenmp efb_openmp.cpp
clean:
	$(RM) efb_serial