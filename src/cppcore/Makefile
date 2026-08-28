# eventcore build (experimental C++ core branch).  gcc-toolset-11 per the
# host build convention; plain -shared -fPIC, no external dependencies.
CXX ?= /opt/rh/gcc-toolset-11/root/usr/bin/g++
CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra
libeventcore.so: eventcore.cpp
	$(CXX) $(CXXFLAGS) -shared -o $@ $<
clean:
	rm -f libeventcore.so
