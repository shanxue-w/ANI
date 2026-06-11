all:run

run:test
	./test 

test:main.o Euler_1D.o
	gcc -g $^ -o $@ -lm

main.o:main.c
	gcc -c -g $< -o $@
	
Euler_1D.o:Euler_1D.c
	gcc -c -g $< -o $@

.PHONY:clean
clean:
	-rm test *.o 
