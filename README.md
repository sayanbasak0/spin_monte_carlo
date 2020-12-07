# Monte Carlo and Zero Temperature simulations w/ Parallelization (OpenMP)

## System consists of Hard Spins, with O(n>0) symmetry, on a (d>0)-dimensional Lattice, with square symmetry, interacting with nearest neighbors only.

### Required code:
* che_arg.c
* mt19937-64_custom.c
* mt19937-64_custom.h

### Old code:
* XY_hysteresis_Z2_RF_6.c
* mt19937-64.c
* mt19937-64.h

### Code under construction:
* spin_model.c

## Compile: 
### with new compilers:
gcc version 6+:
```
$ gcc -fopenmp -O3 che_arg.c -lm -o a.out
```
Intel version 2018+: 
```
$ icc -qopenmp -O3 che_arg.c -o a.out
```
### with old compilers:
Uncomment line in *che_arg.c*:
```
#define OLD_COMPILER 1 
```
and compile with gcc:
```
$ gcc -fopenmp -O3 che_arg.c -lm -o a.out
```
or compile with Intel: 
```
$ icc -qopenmp/-fopenmp -O3 che_arg.c -o a.out
```


### List options: 
```
$ ./a.out
```

### Example run:
```
$ ./a.out -L 64 64 64 -BC 1 1 0 -J 1 1 1 -h 0 -th_step 100 -th_algos 1 1 1 1 -av_step 100 -av_algos 1 2 1 1 -av_smpl 16 -fn EQ_init_Randm -Tmin 0.9 -Tmax 1 -dT 0.2 -out T -out m -out m_avg 
```

### Input through parameters file:
```
$ ./a.out `cat param.txt`
```

Example parameter file for equilibrium simulation of 3D Ising Model:
```
$ cat param1.txt
-L 8 8 8 
-BC 1 1 0
-J 1 1 1 
-h 0
-th_step 100 
-th_algos 3 2 0 1 1 0 1 0 0 1 
-av_step 100 
-av_intr 0
-av_smpl 16 
-av_algos 3 0 1 1 1 1 1 2 1 1
-Tmin 0.9 
-Tmax 1 
-dT 0.2
-fn EQ_init_Randm 
-out T
-out m
-out m_avg
```
``EQ_init_Randm`` (Random Spin Initialization) can be replaced by :
- ``EQ_init_hOrdr`` (Spin ordered along h Initialization) 
- ``EQ_init_Ordrd`` (Ordered Spin Initialization) 
  - provide starting order with ```-order <S[1]> ... <S[dim_S]>```
- ``EQ_init_Load`` (Load spin config from file) 
  - provide filename with ```-Sconfig "filename.ext"```

``-th_algos`` interleave as many types of Monte-Carlo updates you want during thermalization.
- first argument indicates the total.
- the next arguments has to be in sets of 3 ( x total) :
  1. MC algorithm (Metropolis|Glauber|Wolff/Swendsen-Wang)
  2. MC update type (Checkerboard/Swendsen-Wang|Random/Wolff|Linear/Wolff)
  3. Steps for each algorithm and update type

``-av_algos`` interleave as many types of Monte-Carlo updates you want during averaging. Same instructions as above.

Example parameter file for evolution starting from a random spin configuration, saves output while averaging after thermalization of 3D Ising Model:
```
$ cat param2.txt
-L 64 64 64 
-BC 1 1 0
-J 1 1 1 
-h 0
-th_step 100 
-th_algos 1 1 0 1
-av_step 100 
-av_intr 1 
-av_smpl 16 
-av_algos 1 2 1 1
-T 4.50 
-fn Evo_T_Randm
-out T
-out m
-out m_avg
```
``Evo_T_Randm`` (Random Spin Initialization) can be replaced by :
- ``Evo_T_hOrdr`` (Spin ordered along h Initialization) 
- ``Evo_T_Ordrd`` (Ordered Spin Initialization) 
  - provide starting order with ```-order "filename.ext"```
- ``Evo_T_Load`` (Load spin config from file) 
  - provide filename with ```-Sconfig "filename.ext"```

example parameter file for zero temperature non-equilibrium simulation of Random Field Ising Model:
> (recompile with `#define GAUSSIAN_FIELD 1`) \
> -or- \
> (recompile with `#define BIMODAL_FIELD 1`) \
> -and/or- \
> (recompile with `#define GAUSSIAN_BOND 1`) \
> -or- \
> (recompile with `#define BIMODAL_BOND 1`) 
```
$ cat param3.txt
-L 64 64 64 
-BC 1 1 0
-J 1 1 1 
-RF 2.27
-RB 0.00 0.00 0.00
-T 0
-fn ZTNE_dec
-out h
-out m
```
``ZTNE_dec`` (decreasing h) can be replaced by ``ZTNE_inc`` (increasing h)

## Requires recompilations after editing **che_arg.c** to run different models.
* *dim_S* = Spin dimension(s) - O(n) model - Hard spins
* *dim_L* = Lattice dimension(s) or the Spatial dimension(s)
* *GAUSSIAN_FIELD* = Gaussian Distribution of Random Fields
* *BIMODAL_FIELD* = Bimodal Distribution of Random Fields
* *GAUSSIAN_BOND* = Gaussian Distribution of Random Bonds
* *BIMODAL_BOND* = Bimodal Distribution of Random Bonds
* *SAVE_SPIN_AFTER* = Save Spin frequency


---
\* ***This code is under construction***
