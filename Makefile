#When -std=c++11 isn't recognized, a command similar to the one below will be needed.
#$ make CXXFLAGS='-W -Wall -pedantic -std=c++0x -g -O2 -fPIC'
#Alternatively, change the makefile by uncommenting the line below and commenting out the original line.
#CXXFLAGS+=-W -Wall -pedantic -std=c++0x -g -O3 -fPIC
CXXFLAGS+=-W -Wall -pedantic -std=c++11 -g -O2 -fPIC

.PHONY: all clean

all: _gdnn.so

_gdnn.so: gdnn.o

%.so:
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -shared -o $@ $^

clean:
	rm *.o
	rm *.so

