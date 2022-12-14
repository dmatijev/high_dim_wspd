#CC   	= gcc -g
#CLINKER	= gcc -g
CC 	= g++ -std=c++11
CLINKER = g++ -std=c++11
#CC   	= /usr/sww/opt/SUNWspro-4.0/bin/cc -g
#CLINKER	= /usr/sww/opt/SUNWspro-4.0/bin/cc -g
# OPTIONS	= -dalign -xchip=ultra -fast -xtarget=native -xO5 -xarch=v8plus

LIBS 	= -lm

all: wsp

wsp: Makefile wsp.o wsp.h util.o gen.o findwsp.o
	$(CLINKER) $(OPTIONS) -o wsp wsp.o util.o gen.o findwsp.o $(LIBS)

wsp.o: Makefile wsp.c wsp.h
	$(CC) $(OPTIONS) -c wsp.c

util.o: Makefile util.c wsp.h
	$(CC) $(OPTIONS) -c util.c

gen.o: Makefile gen.c wsp.h
	$(CC) $(OPTIONS) -c gen.c

findwsp.o: Makefile findwsp.c wsp.h
	$(CC) $(OPTIONS) -c findwsp.c

clean:
	-rm -f *.o

