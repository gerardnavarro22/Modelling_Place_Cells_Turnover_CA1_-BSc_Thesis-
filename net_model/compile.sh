rm *.so *.txt
g++ -shared -fPIC -o library.so net_model.cpp

