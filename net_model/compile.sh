rm *.so *.txt
g++ -shared -fPIC net_model.cpp -o library.so -fopenmp -lpthread

