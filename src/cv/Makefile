# Makefile
CXX = g++
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
CXXFLAGS = -std=c++17 -Wall -Wextra $(OPENCV_FLAGS)

SOURCES = gridtables.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = table_detector

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(OPENCV_FLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean all
