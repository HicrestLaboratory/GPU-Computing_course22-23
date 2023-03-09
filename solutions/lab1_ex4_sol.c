#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define dtype float
#define STR(s) #s
#define XSTR(s) STR(s)

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage: lab1_ex2 n\n");
        return(1);
    }

    struct timeval temp_1, temp_2;
    printf("argv[1] = %s\n", argv[1]);

    int i, j, n = atoi(argv[1]), len = 1;
    dtype *a, *b;

    for(i=0; i<n; i++) len *= 2;
    printf("Vectors len in %d\n", len);
    a = (dtype*)malloc(sizeof(dtype)*len);
    b = (dtype*)malloc(sizeof(dtype)*len);
    time_t t;
    srand((unsigned) time(&t));
    int typ = (strcmp( XSTR(dtype) ,"int")==0);
    if (typ) {
        int rand_range = (1<<11);
        printf("rand_range= %d\n", rand_range);
        for (i=0; i<len; i++)
            a[i] = rand()/(rand_range);
    } else {
        for (i=0; i<len; i++)
            a[i] = (dtype)rand()/((dtype)RAND_MAX);
    }

    gettimeofday(&temp_1, (struct timezone*)0);
    for (i=0; i<len; i++)
        b[i] = (i!=0) ? (a[i] + b[i-1]) : a[i];
    gettimeofday(&temp_2, (struct timezone*)0);

    int test = 1;
    dtype tmp;
    for (i=0; i<len && test==1; i++) {
        tmp = (dtype) 0;
        for (j=0; j<=i; j++) {
            tmp += a[j];
        }
        test = (tmp == b[i]);
    }

    char* test_str = (test == 1)? "\x1B[32mDONE!\x1B[37m" : "\x1B[31mERROR!\x1B[37m";
    printf("\nTEST (b[i] == Sum_0^i(a[i]) for each i?): \t %s\n", test_str);
    printf("Computation runs in %9.8f CPU time\n", ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0));
}
