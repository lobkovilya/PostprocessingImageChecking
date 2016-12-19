FLAGS=-O2 -std=c++11 -lopencv_highgui -lopencv_imgproc -lopencv_core
PROJ_NAME=detector

all: NoiseIrregularityDetector.o main.o BlockGrid.o
	g++ $(FLAGS) $^ -o $(PROJ_NAME)

BlockGrid.o: BlockGrid.cc BlockGrid.h
	g++ -c $(FLAGS) BlockGrid.cc

NoiseIrregularityDetector.o: NoiseIrregularityDetector.cc NoiseIrregularityDetector.h BlockGrid.o
	g++ -c $(FLAGS) NoiseIrregularityDetector.cc 

main.o: main.cc NoiseIrregularityDetector.h
	g++ -c $(FLAGS) main.cc

test: test.cc
	g++ $(FLAGS) $^ -o $@

clean:
	rm -f $(PROJ_NAME) *.o
