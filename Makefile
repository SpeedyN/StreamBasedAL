# This is a make file for Mondrian forest package
# -----------------------------------------------

# Compiler options
CC = gcc

INCLUDEPATH = -I/usr/include -I$(HOME)/local/include

LINKPATH = -L/usr/lib -L$(HOME)/local/lib -L/usr/local/lib

#CFLAGS = -c -Wall -DNDEBUG -Wno-deprecated -g 
CFLAGS = -c -std=c++11 -O3 -march=native -mtune=native
LDFLAGS = -lconfig++ -lboost_serialization -larmadillo -llapack -lblas -lstdc++ -lm
# Source directory and files
SOURCEDIR = src
HEADERS := $(wildcard $(SOURCEDIR)/*.h)
SOURCES := $(wildcard $(SOURCEDIR)/*.cpp)
OBJECTS := $(SOURCES:.cpp=.o)

# Target output
BUILDTARGET = StreamBasedAL_MF

# Build
all: $(BUILDTARGET)
$(BUILDTARGET): $(OBJECTS) $(SOURCES) $(HEADERS)
	$(CC) $(LINKPATH) -pg $(OBJECTS) -o $@  $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDEPATH) $< -o $@

clean:
	@echo 'Cleaning...'
	rm -f $(SOURCEDIR)/*~ $(SOURCEDIR)/*.o
	rm -f $(BUILDTARGET)
