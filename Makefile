all:efb_serial
	g++ -std=c++11 -o efb_serial efb_serial.cpp
clean:
	$(RM) efb_serial