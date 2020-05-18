# Monte Carlo and Zero Temperature simulations w/ Parallelization

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
$ ./a.out -L 64 64 64 -th_step 100 -th_algo 1 -av_step 100 -av_algo 2 -av_updt 0 -fn EQ_init_Randm -smpl 16 -Tmin 0.9 -Tmax 1 -dT 0.2 -out T -out m -out m_avg 
```

### Input through parameters file:
```
$ ./a.out `cat param.txt`
```

where, parameter file can look like this:
```
$ cat param.txt
-L 64 64 64 
-BC 1 1 0
-J 1 1 1 
-h 0
-th_step 100 
-th_algo 1 
-th_updt 0 
-av_step 100 
-av_smpl 16 
-av_algo 2 
-av_updt 1 
-Tmin 0.9 
-Tmax 1 
-dT 0.2
-fn EQ_init_Randm 
-out T
-out m
-out m_avg

```

## Requires recompilations after edits in **che_arg.c** to run different models.
* *dim_S* = Spin dimension(s) - O(n) model - Hard spins
* *dim_L* = Lattice dimension(s) or the Spatial dimension(s)
* *RANDOM_FIELD* = Gaussian/Bimodal Distribution
* *RANDOM_BOND* = Gaussian/Bimodal Distribution


---
\* ***This file is under construction***
