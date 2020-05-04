/**
 * Random number
 */
#include "mt19937-64_custom.c"

#ifndef MT19937_64_H_CUSTOM

    #define MT19937_64_H_CUSTOM

void init_mt19937_parallel(int size);
void reinit_mt19937_parallel(int size);
void uninit_mt19937_parallel(void);

void init_genrand64(unsigned long long seed, int pos);
void init_by_array64(unsigned long long init_key[],
		     unsigned long long key_length,
             int pos);

unsigned long long genrand64_int64(int pos);
long long genrand64_int63(int pos);
double genrand64_real1(int pos);
double genrand64_real2(int pos);
double genrand64_real3(int pos);


#endif
