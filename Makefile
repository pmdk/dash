CXX := g++
CFLAGS := -std=c++17 -march=native -I./ -lrt -pthread -O3 -g 

Test_Cuckoo: Test_Cuckoo_O
	$(CXX) $(CFLAGS) -o src/test_cuckoo src/test_cuckoo.o -lpmemobj -lpmem

Test_Cuckoo_O: src/test_cuckoo.cpp
	$(CXX) $(CFLAGS) -c src/test_cuckoo.cpp -o src/test_cuckoo.o -lpmemobj -lpmem

Test_CCEH: Test_CCEH_O
	$(CXX) $(CFLAGS) -g -o src/CCEH/test_cceh src/CCEH/test_cceh.o -lpmemobj -lpmem

Test_CCEH_O: src/CCEH/test_cceh.cpp util/hash.h
	$(CXX) $(CFLAGS) -g -c src/CCEH/test_cceh.cpp -o src/CCEH/test_cceh.o -DINPLACE -lpmemobj -lpmem

Test_Level: Test_Level_O
	$(CXX) $(CFLAGS) -g -o src/Level/test_level src/Level/test_level.o -lpmemobj -lpmem

Test_Level_O: src/Level/test_level.cpp util/hash.h
	$(CXX) $(CFLAGS) -g -c src/Level/test_level.cpp -o src/Level/test_level.o -DINPLACE -lpmemobj -lpmem

load_factor: test/load_factor.cc
	$(CXX) $(CFLAGS) -g -o test/load_factor test/load_factor.cc
clean:
	rm -rf src/*.o
	rm -rf src/CCEH/*.o
