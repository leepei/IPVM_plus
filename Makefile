CC = gcc
CXX = g++
CFLAGS = -Wall -Wconversion -O3 -fPIC -std=c++11
LIBS = -lblas -llapack

all: train predict

train: linear.o train.c
	$(CXX) $(CFLAGS) -o train train.c linear.o $(LIBS)

predict: linear.o predict.c
	$(CXX) $(CFLAGS) -o predict predict.c linear.o $(LIBS)

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

clean:
	rm -f *~ linear.o train predict
