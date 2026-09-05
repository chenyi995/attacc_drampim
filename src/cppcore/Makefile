# eventcore build (experimental C++ core branch).  gcc-toolset-11 per the
# host build convention; plain -shared -fPIC, no external dependencies.
# NOTE: this must be '=' and not '?='.  make PREDEFINES CXX=g++, so '?='
# never fires and the build silently fell back to the system compiler
# (gcc 8.5 on the lab hosts) -- the stale/mismatched .so then aborted at
# runtime with std::length_error.  A command-line 'make CXX=...' still
# overrides this.  (chenyi9 2026-09-03)
CXX = /opt/rh/gcc-toolset-11/root/usr/bin/g++
CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra
libeventcore.so: eventcore.cpp
	$(CXX) $(CXXFLAGS) -shared -o $@ $<
clean:
	rm -f libeventcore.so
