#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "../include/helper_cuda.h"

memory_clock_reate() * memori_interface
 memori_interface 10 * 512  su A100
 1 memory controller va a 512 bit (/8 per i byte) (*2 perchè double data rate (calcola sia in clock 0->1 che 1->0))

 + usare cuda_divice_prop stampare frequenza memoria,
 max clock 1215 su A30
 1.215 * 10^9 * (512/8 *10) * 2 / 10^9 ---> 1.5 petabyte ()
 ^
 |
 clock * conversione * mis_controller * 2 (datarate) / 10^9 (--> riportare a byte*second)

 bandwith: (b_r + b_w) / 10^9 / tmp in sec (b_r per kernel)

 devon da cuda device property copiare parte device query
 scrivere cuda kernel che stampi proprietà archittettura (prima get device count) (get davice )
 basta che guardo esempio cuda samples ...
 per calcolare in stesso file della vector add, stampa
