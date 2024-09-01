all: main.cpp graveler.cu
	g++ main.cpp -o main
	nvcc graveler.cu -o graveler