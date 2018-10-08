__device__  double some_rand_func_0_1;
__device__  int some_rand_func_int;
__device__  int no_of_sites;
__device__  int no_of_black_white_sites[2];
__device__  int* black_white_checkerboard;
__device__  int* N_N_I;
__device__  double* spin;
__device__  double order[2];
__device__  double J[2];
__device__  double* J_random;
__device__  double h[2];
__device__  double* h_random;
__device__  double T;
extern "C" __device__ int
update_spin_single(
    int x4/* xyzi */,
    signed char* x2/* spin_local */);
;
extern "C" __device__ double
Energy_minimum(
    int x6/* xyzi */,
    signed char* x13/* spin_local */,
    signed char* x3/* field_local */);
;
extern "C" __device__ double
Energy_old(
    int x6/* xyzi */,
    signed char* x16/* spin_local */,
    signed char* x3/* field_local */);
;
extern "C" __device__ double
Energy_new(
    int x10/* xyzi */,
    signed char* x6/* spin_local */,
    signed char* x7/* field_local */);
;
extern "C" __device__ double
update_probability_Metropolis(
    int x12/* xyzi */);
;
extern "C" __device__ double
update_probability_Glauber(
    int x1/* xyzi */);
;
extern "C" __device__ double
Energy_minimum_old_XY(
    int x6/* xyzi */,
    signed char* x13/* spin_local */);
;
extern "C" __device__ double
Energy_minimum_new_XY(
    int x6/* xyzi */,
    signed char* x14/* spin_local */);
;
extern "C" __device__ double
update_to_minimum_checkerboard(
    int x1/* xyzi */,
    signed char* x2/* spin_local */);
;
