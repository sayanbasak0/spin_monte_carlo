/* 
   Modified for parallelization - 
   by: Sayan Basak
   specify no. of threads: init_mt19937_parallel(size)
   seed for each thread: init_genrand64(seed, thread_num)
      or init_by_array64(init_key, key_length, thread_num)
   ---------------------------------------------------------
   
   A C-program for MT19937-64 (2004/9/29 version).
   Coded by Takuji Nishimura and Makoto Matsumoto.

   This is a 64-bit version of Mersenne Twister pseudorandom number
   generator.

   Before using, initialize the state by using init_genrand64(seed)  
   or init_by_array64(init_key, key_length).

   Copyright (C) 2004, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   References:
   T. Nishimura, ``Tables of 64-bit Mersenne Twisters''
     ACM Transactions on Modeling and 
     Computer Simulation 10. (2000) 348--357.
   M. Matsumoto and T. Nishimura,
     ``Mersenne Twister: a 623-dimensionally equidistributed
       uniform pseudorandom number generator''
     ACM Transactions on Modeling and 
     Computer Simulation 8. (Jan. 1998) 3--30.

   Any feedback is very welcome.
   http://www.math.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove spaces)
*/


#include <stdio.h>

#define NN 312
#define MM 156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define LM 0x7FFFFFFFULL /* Least significant 31 bits */

#define CACHE_STRIDE 512

/* The array for the state vector */
static unsigned long long **mt; 
/* mti==NN+1 means mt[NN] is not initialized */
static int *mti; 
static unsigned long long **mag01;
static int prev_size=0;

/* initialization for number of seeds required */
void init_mt19937_parallel(int size)
{
    size = CACHE_STRIDE*size;
    int i;
    mt = (unsigned long long **)malloc(size*sizeof(unsigned long long*));
    mag01 = (unsigned long long **)malloc(size*sizeof(unsigned long long*));
    mti = (int *)malloc(size*sizeof(int));
    for (i=0; i<size; i++)
    {
        mti[i] = NN+1;
        mt[i] = (unsigned long long *)malloc(NN*sizeof(unsigned long long));
        mag01[i] = (unsigned long long *)malloc(2*sizeof(unsigned long long));
        mag01[i][0] = 0ULL;
        mag01[i][1] = MATRIX_A;
    }
    prev_size = size;
}

void reinit_mt19937_parallel(int size)
{
    size = CACHE_STRIDE*size;
    int i;
    for (i=0; i<prev_size; i++)
    {
        free(mt[i]);
        free(mag01[i]);
    }
    if (prev_size!=0)
    {
        free(mt);
        free(mag01);
        free(mti);
    }

    mt = (unsigned long long **)malloc(size*sizeof(unsigned long long*));
    mag01 = (unsigned long long **)malloc(size*sizeof(unsigned long long*));
    mti = (int *)malloc(size*sizeof(int));
    
    for (i=0; i<size; i++)
    {
        mti[i] = NN+1;
        mt[i] = (unsigned long long *)malloc(NN*sizeof(unsigned long long));
        mag01[i] = (unsigned long long *)malloc(2*sizeof(unsigned long long));
        mag01[i][0] = 0ULL;
        mag01[i][1] = MATRIX_A;
    }
    prev_size = size;
}

void uninit_mt19937_parallel(void)
{
    int i;
    for (i=0; i<prev_size; i++)
    {
        free(mt[i]);
        free(mag01[i]);
    }
    if (prev_size!=0)
    {
        free(mt);
        free(mag01);
        free(mti);
    }
}

/* initializes mt[NN] with a seed */
void init_genrand64(unsigned long long seed, int thread)
{
    thread = CACHE_STRIDE*thread;
    mt[thread][0] = seed;
    for (mti[thread]=1; mti[thread]<NN; mti[thread]++) 
        mt[thread][mti[thread]] =  (6364136223846793005ULL * (mt[thread][mti[thread]-1] ^ (mt[thread][mti[thread]-1] >> 62)) + mti[thread]);
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
void init_by_array64(unsigned long long init_key[],
             unsigned long long key_length,
             int thread)
{
    int tid = CACHE_STRIDE*thread;
    unsigned long long i, j, k;
    init_genrand64(19650218ULL, thread);
    i=1; j=0;
    k = (NN>key_length ? NN : key_length);
    for (; k; k--) {
        mt[tid][i] = (mt[tid][i] ^ ((mt[tid][i-1] ^ (mt[tid][i-1] >> 62)) * 3935559000370003845ULL))
          + init_key[j] + j; /* non linear */
        i++; j++;
        if (i>=NN) { mt[tid][0] = mt[tid][NN-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=NN-1; k; k--) {
        mt[tid][i] = (mt[tid][i] ^ ((mt[tid][i-1] ^ (mt[tid][i-1] >> 62)) * 2862933555777941757ULL))
          - i; /* non linear */
        i++;
        if (i>=NN) { mt[tid][0] = mt[tid][NN-1]; i=1; }
    }

    mt[tid][0] = 1ULL << 63; /* MSB is 1; assuring non-zero initial array */ 
}

/* generates a random number on [0, 2^64-1]-interval */
unsigned long long genrand64_int64(int thread)
{
    int i;
    unsigned long long x;
    int tid = CACHE_STRIDE*thread;

    if (mti[tid] >= NN) { /* generate NN words at one time */

        /* if init_genrand64() has not been called, */
        /* a default initial seed is used     */
        if (mti[tid] == NN+1) 
            init_genrand64(5489ULL, thread); 

        for (i=0;i<NN-MM;i++) {
            x = (mt[tid][i]&UM)|(mt[tid][i+1]&LM);
            mt[tid][i] = mt[tid][i+MM] ^ (x>>1) ^ mag01[tid][(int)(x&1ULL)];
        }
        for (;i<NN-1;i++) {
            x = (mt[tid][i]&UM)|(mt[tid][i+1]&LM);
            mt[tid][i] = mt[tid][i+(MM-NN)] ^ (x>>1) ^ mag01[tid][(int)(x&1ULL)];
        }
        x = (mt[tid][NN-1]&UM)|(mt[tid][0]&LM);
        mt[tid][NN-1] = mt[tid][MM-1] ^ (x>>1) ^ mag01[tid][(int)(x&1ULL)];

        mti[tid] = 0;
    }
  
    x = mt[tid][mti[tid]++];

    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);

    return x;
}

/* generates a random number on [0, 2^63-1]-interval */
long long genrand64_int63(int thread)
{
    return (long long)(genrand64_int64(thread) >> 1);
}

/* generates a random number on [0,1]-real-interval */
double genrand64_real1(int thread)
{
    return (genrand64_int64(thread) >> 11) * (1.0/9007199254740991.0);
}

/* generates a random number on [0,1)-real-interval */
double genrand64_real2(int thread)
{
    return (genrand64_int64(thread) >> 11) * (1.0/9007199254740992.0);
}

/* generates a random number on (0,1)-real-interval */
double genrand64_real3(int thread)
{
    return ((genrand64_int64(thread) >> 12) + 0.5) * (1.0/4503599627370496.0);
}




