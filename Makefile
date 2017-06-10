all: jpeg_encoder.o main.o
	g++ -O3 -Wall -Werror -pedantic jpeg_encoder.o main.o -o jpeg_enc -lOpenCL

main.o:
	g++ -O3 -Wall -Werror -pedantic -c src/main.cpp

jpeg_encoder.o:
	g++ -O3 -Wall -Werror -pedantic -c src/jpeg_encoder.cpp
