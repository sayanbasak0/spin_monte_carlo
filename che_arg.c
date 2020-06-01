// using bitbucket     
// author: Sayan Basak 
//---------------------

// #define enable_CUDA_CODE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "mt19937-64_custom.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <unistd.h> // chdir 
#include <errno.h> // strerror
#include <sys/types.h>
#include <sys/stat.h>
#ifdef enable_CUDA_CODE
#include <cuda_runtime_api.h>
#include <cuda.h>
#define CUDA_with_managed 1
#endif
// #define VARIABLE_SEPARATE
#ifdef VARIABLE_SEPARATE
#include "variable.h"
#endif

#define dim_L 3 // Lattice dimensions
#define dim_S 1 // Spin dimensions

#define MARSAGLIA 1 // uncomment only one
// #define REJECTION 1 // uncomment only one
// #define BOX_MULLER 1 // uncomment only one
// Comment next line for newer compilers // gcc-6.3.0 or newer // intel-18.0.1.163 or newer
// #define OLD_COMPILER 1 
// Uncomment ^it if OpenMP reduction has compile errors 

// #define BUNDLE 4

// #define CONST_RATE 1 // uncomment only one
// #define DIVIDE_BY_SLOPE 1 // uncomment only one
// #define BINARY_DIVISION 1 // uncomment only one
// #define DYNAMIC_BINARY_DIVISION 1 // uncomment only one
// #define DYNAMIC_BINARY_DIVISION_BY_SLOPE 1 // uncomment only one

// #define RANDOM_FIELD 1 // for random field disorder
#define GAUSSIAN_FIELD 1 // for random gaussian fields
// #define BIMODAL_FIELD 1 // for random bimodal fields

// #define RANDOM_BOND 1 // for random bond disorder
// #define GAUSSIAN_BOND 1 // for random gaussian bonds
#define BIMODAL_BOND 1 // for random bimodal bonds

#if !defined(RANDOM_BOND) && !defined(RANDOM_FIELD) && !(dim_S-1)
#define C_IM 1
#endif
#if !defined(RANDOM_BOND) && defined(RANDOM_FIELD) && !(dim_S-1)
#define RFIM 1
#endif
#if defined(RANDOM_BOND) && !defined(RANDOM_FIELD) && !(dim_S-1)
#define RBIM 1
#endif
#if defined(RANDOM_BOND) && defined(RANDOM_FIELD) && !(dim_S-1)
#define RFBIM 1
#endif

#if defined(C_IM) || defined(RFIM) || defined(RBIM) || defined(RFBIM)
#define ZTNE_IM_MULTIPLE 1
#endif

#define SAVE_SPIN_AFTER -1
// #define TRAINING_DATA 1

#if defined(SAVE_SPIN_AFTER) || defined(TRAINING_DATA)
#define SLICE_STRIDE 10
#endif

#define TYPE_VOID 0
#define TYPE_INT 1
#define TYPE_LONGINT 2
#define TYPE_FLOAT 3
#define TYPE_DOUBLE 4

#define CHECKPOINT_TIME 3000.00//300.00 // intervals between checkpoints in seconds
#define RESTORE_CHKPT_VALUE 1 // 0 for initialization, 1 for restoring

// #define UPDATE_CHKR_NON_EQ 1 // uncomment only one
#ifndef UPDATE_CHKR_NON_EQ
// #define UPDATE_ALL_NON_EQ 1 // uncomment only one
#endif
#define UPDATE_CHKR_EQ_MC 1 // for checkerboard updates - parallelizable

#ifdef _OPENMP
#define PARALLEL_RANDOM_MC_SWEEP 1 // parallelized random updates
#endif
#define UPDATE_WOLFF_BFS 1
#ifdef UPDATE_WOLFF_BFS
#define PARALLEL_WOLFF 1 // not optimized
#else 
#define UPDATE_WOLFF_DFS 1
#endif
// #define UPDATE_WOLFF_GHOST 1 // https://journals-aps-org.ezproxy.lib.purdue.edu/pre/pdf/10.1103/PhysRevE.98.063306

// #define SWENDSEN_WANG 1

// #define CHECK_AVALANCHE 1

#define PRINT_OUTPUT 1

// #define OLD_FUNCTION 1

// #define EXPLORE_ENERGY_LANDSCAPE 1

#define WALLTIME_LIMIT 3600//170000.00 // in seconds

#define BLACK_WHITE 1

//========================================================================//
//====================  Variables                     ====================//
    
    #ifndef VARIABLE_SEPARATE
    FILE *pFile_1, *pFile_2, *pFile_phase, *pFile_output = NULL, *pFile_chkpt, *pFile_temp, *pFile_ising_spin, *pFile_ising_h;
    char output_file_0[256];
    #endif
    
    #ifdef TRAINING_DATA
    FILE *pFile_train_data[16];
    // char output_train_data[16][256];
    #endif

    const double pie = 3.14159265358979323846;
    double k_B = 1;

    unsigned int *random_seed;
    int num_of_threads;
    int num_of_procs;
    int cache_size=512;
    double start_time;
    int gpu_threads=16;
    #ifdef enable_CUDA_CODE
    __device__ int dev_gpu_threads=16;
    #endif
    // long int CHUNK_SIZE = 256; 
    int output_prepend = 0; 
    int output_append = 1;

//========================================================================//
//====================  Lattice size                  ====================//
    int lattice_size[dim_L] = { 128, 128, 128 }; // lattice_size[dim_L] = { 128 }{ 128, 128, 128 };
    long int no_of_sites;
    long int no_of_black_sites;
    long int no_of_white_sites;
    long int no_of_black_white_sites[2];

//====================  Checkerboard variables        ====================//
    // long int *black_white_checkerboard[2]; 
    long int *black_white_checkerboard; 
    #if defined (UPDATE_CHKR_NON_EQ) || defined (UPDATE_CHKR_EQ_MC)
        int black_white_checkerboard_reqd = 1;
    #else
        int black_white_checkerboard_reqd = 0;
    #endif
    long int *black_checkerboard; int black_checkerboard_reqd = 0;
    long int *white_checkerboard; int white_checkerboard_reqd = 0;

//====================  Ising hysteresis              ====================//
    long int *nucleation_sites; 
    #if defined(ZTNE_IM_MULTIPLE) || defined(UPDATE_WOLFF_BFS) || defined (UPDATE_CHKR_NON_EQ) || defined (UPDATE_CHKR_EQ_MC)
        int nucleation_sites_reqd = 1; 
    #else
        int nucleation_sites_reqd = 0; 
    #endif
    long int remaining_sites;

//====================  Random MC variables           ====================//
    long int *random_sites;
    #ifdef _OPENMP
        int random_sites_reqd = 1;
    #else
        int random_sites_reqd = 0;
    #endif
    // int cluster_reqd = 1;

//====================  Wolff/Cluster variables       ====================//
    double reflection_plane[dim_S];
    double reflection_matrix[dim_S*dim_S];
    long int *cluster; 
    #if defined(UPDATE_WOLFF_BFS) || defined(UPDATE_WOLFF_DFS) || defined (UPDATE_CHKR_NON_EQ) || defined (UPDATE_CHKR_EQ_MC)
    int cluster_reqd = 1;
    #else
    int cluster_reqd = 0;
    #endif
    int *bond_cluster; 
    #ifdef SWENDSEN_WANG
    int bond_cluster_reqd = 1;
    #else
    int bond_cluster_reqd = 0;
    #endif
    double delta_E_cluster = 0;
    double ghost_spin[dim_S];
    double ghost_matrix[dim_S*dim_S];

//====================  Near neighbor /Boundary Cond  ====================//
    long int *N_N_I; int N_N_I_reqd = 1;
    double BC[dim_L] = { 1, 1, 0 }; // double BC[dim_L] = { 1 }; // 1 -> Periodic | 0 -> Open | -1 -> Anti-periodic(not implemented) -- Boundary Condition

//====================  Spin variable                 ====================//
    double *spin; int spin_reqd = 1;
    double *spin_bkp;
    #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE) || defined (CONST_RATE)
        int spin_bkp_reqd = 1;
    #else
        int spin_bkp_reqd = 0;
    #endif
    double *spin_temp;
    #if defined (UPDATE_ALL_NON_EQ) || defined (UPDATE_CHKR_NON_EQ)
        int spin_temp_reqd = 1;
    #else
        int spin_temp_reqd = 0;
    #endif
    double *spin_old; int spin_old_reqd = 0;
    double *spin_new; int spin_new_reqd = 0;
    // int *spin_sum; int spin_sum_reqd = 1; 
    // double spin_0[dim_S];

//====================  Initialization type           ====================//
    double order[dim_S] = { 1.0 };//  Initialize all spins along this direction (Norm(order) = Norm(spin) = 1)
    int h_order = 0; // 1/0 (Yes/No) Initialize spin along RF direction
    int r_order = 0; // 1/0 (Yes/No) Random initialize spin
    int or_ho_ra = 0; // 0/1/2 ordered/h-ordered/random initialization
    char o_h_r[] = "ohr";
    int ax_ro = 1; // 0/1 axial/rotating driving field
    char a_r[] = "ar";

//====================  MC-update type                ====================//
    int MC_algo_type = 1; // 0 -> Glauber , 1 -> Metropolis, 2 -> Wolff
    int MC_algo_type_thrm = 1; // 0 -> Glauber , 1 -> Metropolis, 2 -> Wolff
    int MC_algo_type_avg = 1; // 0 -> Glauber , 1 -> Metropolis, 2 -> Wolff
    char G_M_W[] = "GMW";
    int MC_update_type = 0; // 0 -> Checkerboard updates/ Full sweep (Swendsen-Wang) , 1 -> Random updates/ Wolff nucleate at Random site , 2 -> Linear updates/
    int MC_update_type_thrm = 0; // 0 -> Checkerboard updates/ Full sweep (Swendsen-Wang) , 1 -> Random updates/ Wolff nucleate at Random site , 2 -> Linear updates/
    int MC_update_type_avg = 0; // 0 -> Checkerboard updates/ Full sweep (Swendsen-Wang) , 1 -> Random updates/ Wolff nucleate at Random site , 2 -> Linear updates/
    char C_R_L[] = "CRL";

//====================  MC-update iterations          ====================//
    long int thermal_i = 10*10; // 128*10*10*10; // thermalizing MCS 
    long int average_j = 10*10; // 128*10*10; // no. of measurements 
    long int sampling_inter = 16;  // random no. (between 1 & sampling_inter) of MCS before taking each measurement // *=sampling_inter-genrand64_int64(0)%sampling_inter

//====================  NN-interaction (J)            ====================//
    double J[dim_L] = { 1.0, 1.0, 1.0 }; // double J[dim_L] = { 1.0 }; 
    int output_J = 0;
    double sigma_J[dim_L] = { 0.0, 0.0, 0.0 }; // double sigma_J[dim_L] = { 0.0 };
    int output_sigma_J = 0;
    double *J_random;
    #ifdef RANDOM_BOND
        int J_random_reqd = 1;
    #else
        int J_random_reqd = 0;
    #endif
    double J_max = 0.0; double J_min = 0.0;
    double J_i_max = 0.0, J_i_min = 0.0; // for hysteresis
    double del_J = 0.01, del_J_cutoff = 0.0001;
    double J_dev_avg[dim_L];
    double J_dev_std[dim_L];

//====================  on-site field (h)             ====================//
    double h[dim_S] = { 0.0 }; // uniform field 
    int output_h = 0;
    double sigma_h[dim_S] = { 2.27 }; // random field strength ( R = { R_x, R_y, ... } )
    int output_sigma_h = 0;
    double *h_random;
    #ifdef RANDOM_FIELD
        int h_random_reqd = 1;
    #else
        int h_random_reqd = 0;
    #endif
    double h_max = 2*dim_L+0.01; double h_min = -2*dim_L-0.01; // for hysteresis (axial)
    double h_i_max = 0.0, h_i_min = 0.0; 
    double del_h = 1.0/8.0/* 0.1 */, del_h_cutoff = 0.00001;
    double del_phi = 0.01, del_phi_cutoff = 0.00000001; // for hysteresis (rotating)
    double h_dev_avg[dim_S];
    double h_dev_std[dim_S];
    double *field_site; int field_site_reqd = 0; // field experienced by spin due to nearest neighbors and on-site field

//====================  Temperature                   ====================//
    double T = 4.49;
    double Temp_min = 4.510;
    double Temp_max = 4.550;
    double delta_T = 0.1;
    int output_T = 0;

//====================  Avalanche delta(S)            ====================//
    double delta_S_squared[dim_S] = { 0.0 };
    int output_delta_S_squared = 0;
    double delta_S_abs[dim_S] = { 0.0 };
    int output_delta_S_abs = 0;
    double delta_S_max = 0.0 ;
    int output_delta_S_max = 0;
    double delta_M[dim_S] = { 0.0 };
    int output_delta_M = 0;
    double delta_E = 0.0 ;
    int output_delta_E = 0;

//====================  Magnetization <M>             ====================//
    double m[dim_S];
    int output_m = 0;
    double m_bkp[dim_S];
    
    double abs_m[dim_S];
    int output_abs_m = 0;
    double m_sum[dim_S];
    double m_avg[dim_S];
    int output_m_avg = 0;

    double m_abs_sum = 0;
    double m_abs_avg = 0;
    int output_m_abs_avg = 0;
    double m_abs_vec_sum[dim_S] = { 0 };
    double m_abs_vec_avg[dim_S] = { 0 };
    int output_m_abs_vec_avg = 0;
    double m_2_sum = 0;
    double m_2_avg = 0;
    int output_m_2_avg = 0;

    double m_2_vec_sum[dim_S] = { 0 };
    double m_2_vec_avg[dim_S] = { 0 };
    int output_m_2_vec_avg = 0;

    double m_4_sum = 0;
    double m_4_avg = 0;
    int output_m_4_avg = 0;
    double m_4_vec_sum[dim_S] = { 0 };
    double m_4_vec_avg[dim_S] = { 0 };
    int output_m_4_vec_avg = 0;

    double m_ab_sum[dim_S*dim_S] = { 0 };
    double m_ab_avg[dim_S*dim_S] = { 0 };
    int output_m_ab_avg = 0;

//====================  Energy <E>                    ====================//
    double E = 0;
    int output_E = 0;
    double E_bkp = 0;
    double E_sum = 0;
    double E_avg = 0;
    int output_E_avg = 0;
    double E_2_sum = 0;
    double E_2_avg = 0;
    int output_E_2_avg = 0;

//====================  Helicity <Y>                  ====================//
    // double Y_1[dim_S*dim_S*dim_L] = { 0 };
    // double Y_2[dim_S*dim_S*dim_L] = { 0 };
    double Y_1[dim_L] = { 0 };
    double Y_2[dim_L] = { 0 };
    int output_Y_1 = 0;
    int output_Y_2 = 0;
    // double Y_1_sum[dim_S*dim_S*dim_L] = { 0 };
    // double Y_2_sum[dim_S*dim_S*dim_L] = { 0 };
    // double Y_1_avg[dim_S*dim_S*dim_L] = { 0 };
    // double Y_2_avg[dim_S*dim_S*dim_L] = { 0 };
    double Y_1_sum[dim_L] = { 0 };
    double Y_2_sum[dim_L] = { 0 };
    double Y_1_avg[dim_L] = { 0 };
    double Y_2_avg[dim_L] = { 0 };
    int output_Y_1_avg = 0;
    int output_Y_2_avg = 0;
    // double Y_ab_mu[dim_S*dim_S*dim_L] = { 0 };
    double Y_x_mu[dim_L] = { 0 };
    int output_Y_x_mu = 0;

//====================  Specific heat Cv              ====================//
    double Cv = 0;
    int output_Cv = 0;

//====================  Susceptibility (Tensor) X     ====================//
    double X = 0;
    int output_X = 0;
    double X_ab[dim_S*dim_S] = { 0 };
    int output_X_ab = 0;

//====================  Binder Parameter B            ====================//
    double B = 0;
    int output_B = 0;
    double B_a[dim_S] = { 0 };
    int output_B_a = 0;
    
//====================  Hysteresis                    ====================//
    long int hysteresis_MCS = 10; 
    int hysteresis_repeat = 4;
    int repeat_sweep = 0;
    int repeat_loop = 0;
    long int h_counter = 0;
    double del_m_counter = 1.0/128.0;
    double m_counter = 1.0;
    // #define CUTOFF_BY_SUM 1 // for find_change sum
    #ifdef CUTOFF_BY_SUM
    double CUTOFF_SPIN = 1.0000000000000e-10; 
    #endif
    #define CUTOFF_BY_MAX 1 // for find_change max
    #ifdef CUTOFF_BY_MAX
    double CUTOFF_SPIN = 1.00000000000000e-14; 
    #endif
    double CUTOFF_M = 1.0/256.0;//1.00000000e-08;
    double CUTOFF_S_SQ = 1.00000000e-08;
    int cutoff_check[2] = { 0, 0 };
    double Spin_Saturation_Limit = 0.80;

//====================  CUDA device ptr               ====================//
    #ifdef enable_CUDA_CODE
        #ifdef CUDA_with_managed
        __managed__ __device__ double *dev_spin;
        __managed__ __device__ double *dev_spin_temp;
        __managed__ __device__ double *dev_spin_bkp;
        __managed__ __device__ double *dev_CUTOFF_SPIN;
        __managed__ __device__ double *dev_CUTOFF_S_SQ;
        __managed__ __device__ double *dev_J;
        __managed__ __device__ double *dev_J_random;
        __managed__ __device__ double *dev_h;
        __managed__ __device__ double *dev_h_random;
        __managed__ __device__ long int *dev_N_N_I;
        __managed__ __device__ long int *dev_black_white_checkerboard;
        __managed__ __device__ long int *dev_no_of_black_white_sites;
        __managed__ __device__ double *dev_m;
        __managed__ __device__ double *dev_m_bkp;
        __managed__ __device__ double *dev_spin_reduce;
        __managed__ __device__ double *dev_delta_spin_abs_reduce;
        __managed__ __device__ double *dev_delta_spin_squared_reduce;
        __managed__ __device__ double *dev_delta_spin_max_reduce;
        #else
        __device__ double *dev_spin;
        __device__ double *dev_spin_temp;
        __device__ double *dev_spin_bkp;
        __device__ double *dev_CUTOFF_SPIN;
        __device__ double *dev_CUTOFF_S_SQ;
        __device__ double *dev_J;
        __device__ double *dev_J_random;
        __device__ double *dev_h;
        __device__ double *dev_h_random;
        __device__ long int *dev_N_N_I;
        __device__ long int *dev_black_white_checkerboard;
        __device__ long int *dev_no_of_black_white_sites;
        __device__ double *dev_m;
        __device__ double *dev_m_bkp;
        __device__ double *dev_spin_reduce;
        __device__ double *dev_delta_spin_reduce;
        __device__ double *dev_delta_spin_squared_reduce;
        __device__ double *dev_delta_spin_max_reduce;
        #endif 
    #endif 
    
    long int no_of_sites_max_power_2;
    long int no_of_sites_remaining_power_2;

//========================================================================//

//========================================================================//
//====================  Functions                     ====================//
//========================================================================//
//====================  OPENMP Parallel used          ====================//

    int thread_num_if_parallel()
    {
        #ifdef _OPENMP
        return omp_get_thread_num();
        #else
        return 0;
        #endif
    }
    
    double get_time_if_parallel()
    {
        #ifdef _OPENMP
        return omp_get_wtime();
        #else
        // return (double) time(NULL);
        // return (double) clock()/CLOCKS_PER_SEC;
        static double nanosec_to_sec = 1000000000.0;
        struct timespec t_local ;
        clock_gettime(CLOCK_MONOTONIC, &t_local);
        return (double)(t_local.tv_sec)+(double)(t_local.tv_nsec)/nanosec_to_sec;
        #endif
    }

//====================  Allocate/Free Memory          ====================//
    
    int free_memory()
    {
        printf("\nFree variables ( ");
        if (spin_reqd == 1)
        {
            printf("spin, ");
            fflush(stdout);
            // usleep(1000000);
            free(spin);
        }
        if (N_N_I_reqd == 1)
        {
            printf("N_N_I, ");
            fflush(stdout);
            // usleep(1000000);
            free(N_N_I);
        }
        if (black_white_checkerboard_reqd == 1)
        {
            printf("black_white_checkerboard, ");
            fflush(stdout);
            // usleep(1000000);
            free(black_white_checkerboard);
            // free(black_white_checkerboard[0]);
            // free(black_white_checkerboard[1]);
        }
        if (random_sites_reqd == 1)
        {
            printf("spin, ");
            fflush(stdout);
            // usleep(1000000);
            free(random_sites);
        }
        if (spin_bkp_reqd == 1)
        {
            printf("spin_bkp, ");
            fflush(stdout);
            // usleep(1000000);
            free(spin_bkp);
        }
        if (spin_temp_reqd == 1)
        {
            printf("spin_temp, ");
            fflush(stdout);
            // usleep(1000000);
            free(spin_temp);
        }
        if (h_random_reqd == 1)
        {
            printf("h_random, ");
            fflush(stdout);
            // usleep(1000000);
            free(h_random);
        }
        if (J_random_reqd == 1)
        {
            printf("J_random, ");
            fflush(stdout);
            // usleep(1000000);
            free(J_random);
        }
        if (cluster_reqd == 1)
        {
            printf("cluster, ");
            fflush(stdout);
            // usleep(1000000);
            free(cluster);
        }
        if (bond_cluster_reqd == 1)
        {
            printf("bond_cluster, ");
            fflush(stdout);
            // usleep(1000000);
            free(bond_cluster);
        }
        if (nucleation_sites_reqd == 1)
        {
            printf("nucleation_sites, ");
            fflush(stdout);
            // usleep(1000000);
            free(nucleation_sites);
        }
        if (spin_old_reqd == 1)
        {
            printf("spin_old, ");
            fflush(stdout);
            // usleep(1000000);
            free(spin_old);
        }
        if (spin_new_reqd == 1)
        {
            printf("spin_new, ");
            fflush(stdout);
            // usleep(1000000);
            free(spin_new);
        }
        if (field_site_reqd == 1)
        {
            printf("field_site, ");
            fflush(stdout);
            // usleep(1000000);
            free(field_site);
        }
        #ifdef _OPENMP
            // free(random_seed);
            // printf("random_seed[for_omp_parallelization()] ");
            // fflush(stdout);
            printf("mt19937 variables ");
            fflush(stdout);
            // usleep(1000000);
            uninit_mt19937_parallel();
        #endif
        
        #ifdef enable_CUDA_CODE
        cudaFree(dev_CUTOFF_SPIN);
        cudaFree(dev_CUTOFF_S_SQ);
        cudaFree(dev_spin);
        cudaFree(dev_spin_temp);
        cudaFree(dev_spin_bkp);
        cudaFree(dev_J);
        #ifdef RANDOM_BOND
        cudaFree(dev_J_random);
        #endif
        cudaFree(dev_h);
        #ifdef RANDOM_FIELD
        cudaFree(dev_h_random);
        #endif
        cudaFree(dev_N_N_I);
        cudaFree(dev_black_white_checkerboard);
        cudaFree(dev_no_of_black_white_sites);
        cudaFree(dev_m);
        cudaFree(dev_m_bkp);
        cudaFree(dev_spin_reduce);
        cudaFree(dev_delta_spin_abs_reduce);
        cudaFree(dev_delta_spin_squared_reduce);
        cudaFree(dev_delta_spin_max_reduce);
        #endif
        printf(")\n");
        return 0;
    }

    int allocate_memory()
    {
        printf("\nVariables allocated ( ");
        static int first_call = 1;
        int j_L;
        no_of_sites = 1;
        
        for (j_L=0; j_L<dim_L; j_L++)
        {
            no_of_sites = no_of_sites*lattice_size[j_L];
        }
        static long int no_of_sites_local = 1;
        if (first_call == 1)
        {
            no_of_sites_local = no_of_sites;
        }

        int free_and_allocate = 0;
        if (no_of_sites_local != no_of_sites)
        {
            free_and_allocate = 1;
            no_of_sites_local = no_of_sites;
        }

        static int spin_reqd_local = 0;
        if (spin_reqd_local == 0 && spin_reqd == 1)
        {
            printf("spin");
            spin = (double*)malloc(dim_S*(no_of_sites+1)*sizeof(double));
            spin_reqd_local = 1;
        }
        else 
        {
            if (spin_reqd_local == 1 && spin_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(spin);
                    spin = (double*)malloc(dim_S*(no_of_sites+1)*sizeof(double));
                }
            }
            else 
            {
                if (spin_reqd_local == 1 && spin_reqd == 0)
                {
                    free(spin);
                    spin_reqd_local = 0;
                }
            }
        }
        
        static int N_N_I_reqd_local = 0;
        if (N_N_I_reqd_local == 0 && N_N_I_reqd == 1)
        {
            printf(", N_N_I");
            N_N_I = (long int*)malloc(2*dim_L*(no_of_sites+1)*sizeof(long int));
            N_N_I_reqd_local = 1;
        }
        else 
        {
            if (N_N_I_reqd_local == 1 && N_N_I_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(N_N_I);
                    N_N_I = (long int*)malloc(2*dim_L*(no_of_sites+1)*sizeof(long int));
                }
            }
            else 
            {
                if (N_N_I_reqd_local == 1 && N_N_I_reqd == 0)
                {
                    free(N_N_I);
                    N_N_I_reqd_local = 0;
                }
            }
        }
        
        static int black_white_checkerboard_reqd_local = 0;
        if (black_white_checkerboard_reqd_local == 0 && black_white_checkerboard_reqd == 1)
        {
            printf(", black_white_checkerboard");
            if (no_of_sites % 2 == 1)
            {
                no_of_black_white_sites[0] = (no_of_sites + 1) / 2;
                no_of_black_white_sites[1] = (no_of_sites - 1) / 2;
            }
            else
            {
                no_of_black_white_sites[0] = no_of_sites / 2;
                no_of_black_white_sites[1] = no_of_sites / 2;
            }
            // black_white_checkerboard[0] = (long int*)malloc(no_of_black_white_sites[0]*sizeof(long int));
            // black_white_checkerboard[1] = (long int*)malloc(no_of_black_white_sites[1]*sizeof(long int));
            black_white_checkerboard = (long int*)malloc(2*no_of_black_white_sites[0]*sizeof(long int));
            black_white_checkerboard_reqd_local = 1;
        }
        else 
        {
            if (black_white_checkerboard_reqd_local == 1 && black_white_checkerboard_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    if (no_of_sites % 2 == 1)
                    {
                        no_of_black_white_sites[0] = (no_of_sites + 1) / 2;
                        no_of_black_white_sites[1] = (no_of_sites - 1) / 2;
                    }
                    else
                    {
                        no_of_black_white_sites[0] = no_of_sites / 2;
                        no_of_black_white_sites[1] = no_of_sites / 2;
                    }
                    // free(black_white_checkerboard[0]);
                    // free(black_white_checkerboard[1]);
                    free(black_white_checkerboard);
                    // black_white_checkerboard[0] = (long int*)malloc(no_of_black_white_sites[0]*sizeof(long int));
                    // black_white_checkerboard[1] = (long int*)malloc(no_of_black_white_sites[1]*sizeof(long int));
                    black_white_checkerboard = (long int*)malloc(2*no_of_black_white_sites[0]*sizeof(long int));
                }
            }
            else 
            {
                if (black_white_checkerboard_reqd_local == 1 && black_white_checkerboard_reqd == 0)
                {
                    // free(black_white_checkerboard[0]);
                    // free(black_white_checkerboard[1]);
                    free(black_white_checkerboard);
                    black_white_checkerboard_reqd_local = 0;
                }
            }
        }

        static int random_sites_reqd_local = 0;
        if (random_sites_reqd_local == 0 && random_sites_reqd == 1)
        {
            printf(", random_sites");
            #ifdef PARALLEL_RANDOM_MC_SWEEP
            random_sites = (long int*)malloc(omp_get_max_threads()*sizeof(long int));
            #else
            random_sites = (long int*)malloc(((no_of_sites + 1) / 2)*sizeof(long int));
            #endif
            random_sites_reqd_local = 1;
        }
        else 
        {
            if (random_sites_reqd_local == 1 && random_sites_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(random_sites);
                    #ifdef PARALLEL_RANDOM_MC_SWEEP
                    random_sites = (long int*)malloc(omp_get_max_threads()*sizeof(long int));
                    #else
                    random_sites = (long int*)malloc(((no_of_sites + 1) / 2)*sizeof(long int));
                    #endif
                }
            }
            else 
            {
                if (random_sites_reqd_local == 1 && random_sites_reqd == 0)
                {
                    free(random_sites);
                    random_sites_reqd_local = 0;
                }
            }
        }

        static int spin_bkp_reqd_local = 0;
        if (spin_bkp_reqd_local == 0 && spin_bkp_reqd == 1)
        {
            printf(", spin_bkp");
            spin_bkp = (double*)malloc(dim_S*(no_of_sites+1)*sizeof(double));
            spin_bkp_reqd_local = 1;
        }
        else 
        {
            if (spin_bkp_reqd_local == 1 && spin_bkp_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(spin_bkp);
                    spin_bkp = (double*)malloc(dim_S*(no_of_sites+1)*sizeof(double));
                }
            }
            else 
            {
                if (spin_bkp_reqd_local == 1 && spin_bkp_reqd == 0)
                {
                    free(spin_bkp);
                    spin_bkp_reqd_local = 0;
                }
            }
        }

        static int spin_temp_reqd_local = 0;
        if (spin_temp_reqd_local == 0 && spin_temp_reqd == 1)
        {
            printf(", spin_temp");
            spin_temp = (double*)malloc(dim_S*(no_of_sites+1)*sizeof(double));
            spin_temp_reqd_local = 1;
        }
        else 
        {
            if (spin_temp_reqd_local == 1 && spin_temp_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(spin_temp);
                    spin_temp = (double*)malloc(dim_S*(no_of_sites+1)*sizeof(double));
                }
            }
            else 
            {
                if (spin_temp_reqd_local == 1 && spin_temp_reqd == 0)
                {
                    free(spin_temp);
                    spin_temp_reqd_local = 0;
                }
            }
        }
        
        static int h_random_reqd_local = 0;
        if (h_random_reqd_local == 0 && h_random_reqd == 1)
        {
            printf(", h_random");
            h_random = (double*)malloc(dim_S*no_of_sites*sizeof(double));
            h_random_reqd_local = 1;
        }
        else 
        {
            if (h_random_reqd_local == 1 && h_random_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(h_random);
                    h_random = (double*)malloc(dim_S*no_of_sites*sizeof(double));
                }
            }
            else 
            {
                if (h_random_reqd_local == 1 && h_random_reqd == 0)
                {
                    free(h_random);
                    h_random_reqd_local = 0;
                }
            }
        }
        
        static int J_random_reqd_local = 0;
        if (J_random_reqd_local == 0 && J_random_reqd == 1)
        {
            printf(", J_random");
            J_random = (double*)malloc(2*dim_L*(no_of_sites+1)*sizeof(double));
            J_random_reqd_local = 1;
        }
        else 
        {
            if (J_random_reqd_local == 1 && J_random_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(J_random);
                    J_random = (double*)malloc(2*dim_L*(no_of_sites+1)*sizeof(double));
                }
            }
            else 
            {
                if (J_random_reqd_local == 1 && J_random_reqd == 0)
                {
                    free(J_random);
                    J_random_reqd_local = 0;
                }
            }
        }
        
        static int cluster_reqd_local = 0;
        if (cluster_reqd_local == 0 && cluster_reqd == 1)
        {
            printf(", cluster");
            cluster = (long int*)malloc((no_of_sites+1)*sizeof(long int));
            cluster_reqd_local = 1;
        }
        else 
        {
            if (cluster_reqd_local == 1 && cluster_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(cluster);
                    cluster = (long int*)malloc((no_of_sites+1)*sizeof(long int));
                }
            }
            else 
            {
                if (cluster_reqd_local == 1 && cluster_reqd == 0)
                {
                    free(cluster);
                    cluster_reqd_local = 0;
                }
            }
        }

        static int bond_cluster_reqd_local = 0;
        if (bond_cluster_reqd_local == 0 && bond_cluster_reqd == 1)
        {
            printf(", bond_cluster");
            bond_cluster = (int*)malloc(dim_L*(no_of_sites+1)*sizeof(int));
            bond_cluster_reqd_local = 1;
        }
        else 
        {
            if (bond_cluster_reqd_local == 1 && bond_cluster_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(bond_cluster);
                    bond_cluster = (int*)malloc(dim_L*(no_of_sites+1)*sizeof(int));
                }
            }
            else 
            {
                if (bond_cluster_reqd_local == 1 && bond_cluster_reqd == 0)
                {
                    free(bond_cluster);
                    bond_cluster_reqd_local = 0;
                }
            }
        }
        
        static int nucleation_sites_reqd_local = 0;
        if (nucleation_sites_reqd_local == 0 && nucleation_sites_reqd == 1)
        {
            printf(", nucleation_sites");
            nucleation_sites = (long int*)malloc(no_of_sites*sizeof(long int));
            nucleation_sites_reqd_local = 1;
        }
        else 
        {
            if (nucleation_sites_reqd_local == 1 && nucleation_sites_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(nucleation_sites);
                    nucleation_sites = (long int*)malloc(no_of_sites*sizeof(long int));
                }
            }
            else 
            {
                if (nucleation_sites_reqd_local == 1 && nucleation_sites_reqd == 0)
                {
                    free(nucleation_sites);
                    nucleation_sites_reqd_local = 0;
                }
            }
        }
        
        static int spin_old_reqd_local = 0;
        if (spin_old_reqd_local == 0 && spin_old_reqd == 1)
        {
            printf(", spin_old");
            spin_old = (double*)malloc(dim_S*no_of_sites*sizeof(double));
            spin_old_reqd_local = 1;
        }
        else 
        {
            if (spin_old_reqd_local == 1 && spin_old_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(spin_old);
                    spin_old = (double*)malloc(dim_S*no_of_sites*sizeof(double));
                }
            }
            else 
            {
                if (spin_old_reqd_local == 1 && spin_old_reqd == 0)
                {
                    free(spin_old);
                    spin_old_reqd_local = 0;
                }
            }
        }

        static int spin_new_reqd_local = 0;
        if (spin_new_reqd_local == 0 && spin_new_reqd == 1)
        {
            printf(", spin_new");
            spin_new = (double*)malloc(dim_S*no_of_sites*sizeof(double));
            spin_new_reqd_local = 1;
        }
        else 
        {
            if (spin_new_reqd_local == 1 && spin_new_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(spin_new);
                    spin_new = (double*)malloc(dim_S*no_of_sites*sizeof(double));
                }
            }
            else 
            {
                if (spin_new_reqd_local == 1 && spin_new_reqd == 0)
                {
                    free(spin_new);
                    spin_new_reqd_local = 0;
                }
            }
        }

        static int field_site_reqd_local = 0;
        if (field_site_reqd_local == 0 && field_site_reqd == 1)
        {
            printf(", field_site");
            field_site = (double*)malloc(dim_S*no_of_sites*sizeof(double));
            field_site_reqd_local = 1;
        }
        else 
        {
            if (field_site_reqd_local == 1 && field_site_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(field_site);
                    field_site = (double*)malloc(dim_S*no_of_sites*sizeof(double));
                }
            }
            else 
            {
                if (field_site_reqd_local == 1 && field_site_reqd == 0)
                {
                    free(field_site);
                    field_site_reqd_local = 0;
                }
            }
        }
        printf(" )\n");
        return 0;

    }
    
//====================  Custom                        ====================//

    long int custom_int_pow(long int base, int power)
    {
        if (power > 0)
        {
            return base * custom_int_pow(base, (power - 1));
        }
        return 1;
    }

    double custom_double_pow(double base, int power)
    {
        if (power > 0)
        {
            return base * custom_double_pow(base, (power - 1));
        }
        return 1.0;
    }

    int custom_int_sign(long int x)
    {
        if (x<0)
        {
            return -1;
        }
        else // if (x>0)
        {
            return 1;
        }
        return 0;
    }

    double custom_double_sign(double x)
    {
        if (x<0.0)
        {
            return -1.0;
        }
        else // if (x>0.0)
        {
            return 1.0;
        }
        return 0.0;
    }

    long int nearest_neighbor(long int xyzi, int j_L, int k_L)
    {
        long int xyzinn;
        int jj_L;
        // long int temp_1 = pow(lattice_size, j);
        long int temp_1 = 1;
        for(jj_L=0; jj_L<j_L; jj_L++)
        {
            temp_1 = temp_1 * lattice_size[jj_L];
        }
        // long int temp_1 = custom_int_pow(lattice_size, j);
        int temp_2 = ((long int)(xyzi/temp_1))%lattice_size[j_L];
        xyzinn = xyzi - (temp_2) * temp_1;
        xyzinn = xyzinn + ((temp_2 + ((k_L*2)-1) + lattice_size[j_L])%lattice_size[j_L]) * temp_1;
        return xyzinn;
    }

    int direction_index(long int xyzi, int site_to_dir_index[], long int dir_mult[])
    {
        int j_L;
        // long int temp_1 = pow(lattice_size, j);
        long int temp_1 = 1;
        
        for(j_L=0; j_L<dim_L; j_L++)
        {
            dir_mult[j_L] = temp_1;
            site_to_dir_index[j_L] = ((long int)(xyzi/temp_1))%lattice_size[j_L];
            temp_1 = temp_1 * lattice_size[j_L];
        }
        // long int temp_1 = custom_int_pow(lattice_size, j);
        return 0;
    }

    double generate_bimodal()
    {
        long long r = genrand64_int64(thread_num_if_parallel());

        return (double) (2*(r%2)-1);
    }

    double generate_gaussian() // Marsaglia polar method
    {
        static int which_method = 0;
        double sigma = 1.0;
        #ifdef REJECTION
            if (which_method == 0)
            {
                which_method = !which_method;
                printf("\n Rejection Sampling method used to generate Gaussian Distribution.\n");
            }
            double X, r;

            double P_max = 1; // 1/( sqrt(2*pie) * sigma );
            double P_x;

            do
            {
                X = (-10 + 20 * genrand64_real1(thread_num_if_parallel()));
                r = genrand64_real1(thread_num_if_parallel());
                
                P_x =  exp( - X*X / (2 * sigma*sigma ) );
            }
            while (r >= P_x/P_max);
        
            return ( sigma * (double) X);
        #endif

        double U1, U2, rho, theta, W, mult;
        static double X1, X2;
        static int call = 0;
        

        if (call == 1)
        {
            call = !call;
            return (sigma * (double) X2);
        }

        #ifdef MARSAGLIA
            if (which_method == 0)
            {
                which_method = !which_method;
                printf("\n Marsaglia Polar method used to generate Gaussian Distribution.\n");
            }
            do
            {
                U1 = -1.0 + 2.0 * genrand64_real1(thread_num_if_parallel()) ; // ((double) rand () / RAND_MAX) * 2;
                U2 = -1.0 + 2.0 * genrand64_real1(thread_num_if_parallel()) ; // ((double) rand () / RAND_MAX) * 2;
                
                W = U1*U1 + U2*U2;
            }
            while (W >= 1 || W == 0);

            mult = sqrt ((-2.0 * log (W)) / W);
        #endif

        #ifdef BOX_MULLER
            if (which_method == 0)
            {
                which_method = !which_method;
                printf("\n Box-Muller transform used to generate Gaussian Distribution.\n");
            }
            
            rho = genrand64_real3(thread_num_if_parallel());
            theta = genrand64_real2(thread_num_if_parallel());
            
            U1 = sin(2*pie*theta);
            U2 = cos(2*pie*theta);

            mult = sqrt (-2 * log (1-rho));
        #endif


        X1 = U1 * mult;
        X2 = U2 * mult;
        
        call = !call;

        return (sigma * (double) X1);
    }

    #ifdef enable_CUDA_CODE
        __global__ void copy_in_device_var(double* to_var, double* from_var, long int sites)
        {
            int index = threadIdx.x + blockIdx.x*blockDim.x;
            if (index < sites)
            {
                to_var[index] = from_var[index];
            }
            return; 
        }
    #endif

//====================  Initialization                ====================//

    int initialize_h_zero()
    {
        long int i;
        #pragma omp parallel for
        for (i=0; i<dim_S*no_of_sites; i++)
        {
            h_random[i] = 0;
        }
        // memset(h_random, 0, dim_S*no_of_sites);
        return 0;
    }

    int initialize_h_random_bimodal()
    {
        long int i;
        int j_S;
        
        #ifdef RANDOM_FIELD
        h_i_min = 0;
        h_i_max = 0;
        initialize_h_zero();
        for(j_S=0; j_S<dim_S; j_S++)
        {
            h_dev_avg[j_S] = 0;
            h_dev_std[j_S] = 0;
        }
        #pragma omp parallel for private(j_S) reduction(min:h_i_min) reduction(max:h_i_max) reduction(+:h_dev_avg[:dim_S],h_dev_std[:dim_S])
        for(i=0; i<no_of_sites; i=i+1)
        {
            double r_bim = generate_bimodal();
            for(j_S=0; j_S<dim_S; j_S++)
            {
                h_random[dim_S*i + j_S] = sigma_h[j_S] * r_bim;
                h_dev_avg[j_S] += h_random[dim_S*i + j_S];
                h_dev_std[j_S] += h_random[dim_S*i + j_S]*h_random[dim_S*i + j_S];

                if (h_random[dim_S*i + j_S]>h_i_max)
                {
                    h_i_max = h_random[dim_S*i + j_S];
                }
                else if (h_random[dim_S*i + j_S]<h_i_min)
                {
                    h_i_min = h_random[dim_S*i + j_S];
                }
            }
        }

        for(j_S=0; j_S<dim_S; j_S++)
        {
            h_dev_avg[j_S] = h_dev_avg[j_S] / no_of_sites;
            h_dev_std[j_S] = sqrt(h_dev_std[j_S] / no_of_sites - h_dev_avg[j_S]);
        }
        #endif

        return 0;
    }

    int initialize_h_random_gaussian()
    {
        long int i;
        int j_S;
        
        #ifdef RANDOM_FIELD
        h_i_min = 0;
        h_i_max = 0;
        initialize_h_zero();
        for(j_S=0; j_S<dim_S; j_S++)
        {
            h_dev_avg[j_S] = 0;
            h_dev_std[j_S] = 0;
            #pragma omp parallel for reduction(min:h_i_min) reduction(max:h_i_max) reduction(+:h_dev_avg[:dim_S],h_dev_std[:dim_S])
            for(i=0; i<no_of_sites; i++)
            {
                h_random[dim_S*i + j_S] = sigma_h[j_S] * generate_gaussian();
                h_dev_avg[j_S] += h_random[dim_S*i + j_S];
                h_dev_std[j_S] += h_random[dim_S*i + j_S]*h_random[dim_S*i + j_S];

                if (h_random[dim_S*i + j_S]>h_i_max)
                {
                    h_i_max = h_random[dim_S*i + j_S];
                }
                else if (h_random[dim_S*i + j_S]<h_i_min)
                {
                    h_i_min = h_random[dim_S*i + j_S];
                }
            }
            
            h_dev_avg[j_S] = h_dev_avg[j_S] / no_of_sites;
            h_dev_std[j_S] = sqrt( h_dev_std[j_S] / no_of_sites - h_dev_avg[j_S] );
        }
        #endif

        return 0;
    }

    int initialize_J_zero()
    {
        long int i;
        #pragma omp parallel for
        for (i=0; i<2*dim_L*(no_of_sites+1); i++)
        {
            J_random[i] = 0;
        }
        // memset(J_random, 0, 2*dim_L*(no_of_sites+1));
        return 0;
    }
    
    int initialize_J_random_bimodal()
    {
        long int i;
        int j_L, k_L;
        
        #ifdef RANDOM_BOND
        J_i_min = 0;
        J_i_max = 0;
        initialize_J_zero();
        for(j_L=0; j_L<dim_L; j_L=j_L+1)
        {
            J_dev_avg[j_L] = 0;
            J_dev_std[j_L] = 0;
            #pragma omp parallel for reduction(min:J_i_min) reduction(max:J_i_max) reduction(+:J_dev_avg[:dim_L],J_dev_std[:dim_L])
            for(i=0; i<no_of_sites; i=i+1)
            {
                J_random[2*dim_L*i + 2*j_L] = sigma_J[j_L] * generate_bimodal();
                J_dev_avg[j_L] += J_random[2*dim_L*i + 2*j_L];
                J_dev_std[j_L] += J_random[2*dim_L*i + 2*j_L]*J_random[2*dim_L*i + 2*j_L];

                if (J_random[2*dim_L*i + 2*j_L]>J_i_max)
                {
                    J_i_max = J_random[2*dim_L*i + 2*j_L];
                }
                else if (J_random[2*dim_L*i + 2*j_L]<J_i_min)
                {
                    J_i_min = J_random[2*dim_L*i + 2*j_L];
                }
                J_random[2*dim_L*N_N_I[2*dim_L*i + 2*j_L] + 2*j_L + 1] = J_random[2*dim_L*i + 2*j_L];
            }
            J_dev_avg[j_L] = J_dev_avg[j_L] / no_of_sites;
            J_dev_std[j_L] = sqrt(J_dev_std[j_L] / no_of_sites - J_dev_avg[j_L]);
        }
        #endif
        
        return 0;
    }

    int initialize_J_random_gaussian()
    {
        long int i;
        int j_L, k_L;
        
        #ifdef RANDOM_BOND
        J_i_min = 0;
        J_i_max = 0;
        initialize_J_zero();
        for(j_L=0; j_L<dim_L; j_L=j_L+1)
        {
            J_dev_avg[j_L] = 0;
            J_dev_std[j_L] = 0;
            #pragma omp parallel for reduction(min:J_i_min) reduction(max:J_i_max) reduction(+:J_dev_avg[:dim_L],J_dev_std[:dim_L])
            for(i=0; i<no_of_sites; i=i+1)
            {
                J_random[2*dim_L*i + 2*j_L] = sigma_J[j_L] * generate_gaussian();
                J_dev_avg[j_L] += J_random[2*dim_L*i + 2*j_L];
                J_dev_std[j_L] += J_random[2*dim_L*i + 2*j_L]*J_random[2*dim_L*i + 2*j_L];

                if (J_random[2*dim_L*i + 2*j_L]>J_i_max)
                {
                    J_i_max = J_random[2*dim_L*i + 2*j_L];
                }
                else if (J_random[2*dim_L*i + 2*j_L]<J_i_min)
                {
                    J_i_min = J_random[2*dim_L*i + 2*j_L];
                }
                J_random[2*dim_L*N_N_I[2*dim_L*i + 2*j_L] + 2*j_L + 1] = J_random[2*dim_L*i + 2*j_L];
                J_random[2*dim_L*N_N_I[2*dim_L*i + 2*j_L] + 2*j_L + 1] = J_random[2*dim_L*i + 2*j_L];
            }
            J_dev_avg[j_L] = J_dev_avg[j_L] / no_of_sites;
            J_dev_std[j_L] = sqrt(J_dev_std[j_L] / no_of_sites - J_dev_avg[j_L]);
        }
        #endif
        
        return 0;
    }

    int initialize_nearest_neighbor_index_2()
    {
        long int i; 
        int j_L, k_L;
        // no_of_sites = 1;
        // for (j_L=0; j_L<dim_L; j_L++)
        // {
        //     no_of_sites = no_of_sites*lattice_size[j_L];
        // }
        // N_N_I = (long int*)malloc(2*dim_L*no_of_sites*sizeof(long int));  
        
        #pragma omp parallel for private(j_L,k_L) //memcheck
        for(i=0; i<no_of_sites; i++)
        {
            int site_to_dir_index[dim_L];
            long int dir_mult[dim_L];
            direction_index(i, site_to_dir_index, dir_mult);
            for(j_L=0; j_L<dim_L; j_L=j_L+1)
            {
                for(k_L=0; k_L<2; k_L=k_L+1)
                {
                    // long int nn = N_N_I[i*2*dim_L + 2*j_L + k_L];
                    if (site_to_dir_index[j_L]==0 && k_L == 0)
                    {
                        N_N_I[i*2*dim_L + 2*j_L + k_L] = i + (lattice_size[j_L]-1)*dir_mult[j_L];
                    }
                    else if (site_to_dir_index[j_L]==lattice_size[j_L]-1 && k_L == 1)
                    {
                        N_N_I[i*2*dim_L + 2*j_L + k_L] = i - (lattice_size[j_L]-1)*dir_mult[j_L];
                    }
                    else
                    {
                        N_N_I[i*2*dim_L + 2*j_L + k_L] = i + (k_L*2-1)*dir_mult[j_L];
                    }
                    
                    // N_N_I[i*2*dim_L + 2*j_L + k_L] = i - (long int)(site_to_dir_index[j_L]) * dir_mult[j_L];
                    // N_N_I[i*2*dim_L + 2*j_L + k_L] = N_N_I[i*2*dim_L + 2*j_L + k_L] + (long int)((site_to_dir_index[j_L] + ((k_L*2)-1) + lattice_size[j_L])%lattice_size[j_L]) * dir_mult[j_L];
                    
                    if (BC[j_L] == 0)
                    {
                        if ( site_to_dir_index[j_L] == k_L*(lattice_size[j_L]-1) )
                        {
                            N_N_I[i*2*dim_L + 2*j_L + k_L] = no_of_sites;
                        }
                    }

                    // if (N_N_I[i*2*dim_L + 2*j_L + k_L] != nn)
                    // {
                    //     printf("%ld,%ld,%d,%d\n", N_N_I[i*2*dim_L + 2*j_L + k_L], i, j_L, k_L);
                    //     // return -1;
                    // }
                }
            }
        }

        printf("Nearest neighbor initialized. \n");
        return 0;
    }

    int initialize_nearest_neighbor_index()
    {
        long int i; 
        int j_L, k_L;
        // no_of_sites = 1;
        // for (j_L=0; j_L<dim_L; j_L++)
        // {
        //     no_of_sites = no_of_sites*lattice_size[j_L];
        // }
        // N_N_I = (long int*)malloc(2*dim_L*no_of_sites*sizeof(long int));  
        
        #pragma omp parallel for private(j_L,k_L) //memcheck
        for(i=0; i<no_of_sites; i++)
        {
            // direction_index(i);
            for(j_L=0; j_L<dim_L; j_L++)
            {
                for(k_L=0; k_L<2; k_L++)
                {
                    N_N_I[i*2*dim_L + 2*j_L + k_L] = nearest_neighbor(i, j_L, k_L);
                    
                    if (BC[j_L] == 0)
                    {
                        int site_to_dir_index[dim_L];
                        long int dir_mult[dim_L];
                        direction_index(i, site_to_dir_index, dir_mult);
                        if ( site_to_dir_index[j_L] == k_L*(lattice_size[j_L]-1) )
                        {
                            N_N_I[i*2*dim_L + 2*j_L + k_L] = no_of_sites;
                        }
                    }
                }
            }
        }

        printf("Nearest neighbor initialized. \n");
        return 0;
    }

    int initialize_checkerboard_sites_2()
    {
        long int i = 0;
        int j_L=0, k_L=1;
        
        long int black_white_index[2] = { 0, 0 };
        
        long int i_1=0, i_2=1;
        // #pragma omp parallel for
        for (i=0; i<no_of_sites; i++)
        {
            // black_white_checkerboard[i]=-1;
            cluster[i] = -1;
        }
        // long int next_site = nucleation_sites[0];
        nucleation_sites[i_1] = 0;
        black_white_checkerboard[0] = i_1;
        cluster[i_1] = 0;
        cluster[no_of_sites] = 2;
        // black_white_index[0] = 1;
        while (i_1!=i_2)
        {
            // #pragma omp parallel private(k_L)
            for (j_L=0; j_L<dim_L; j_L++)
            {
                // for (k_L=0; k_L<2; k_L++)
                {
                    long int next_site = N_N_I[2*dim_L*nucleation_sites[i_1] + 2*j_L + 1];
                    if ( cluster[next_site]==-1 )
                    {
                        cluster[next_site] = 1-cluster[nucleation_sites[i_1]];
                        long int i_2_temp;
                        // #pragma omp atomic capture
                            {i_2_temp = i_2; i_2++;}
                        nucleation_sites[i_2_temp] = next_site;
                    }
                }
            }
            i_1++;
        }
        printf("%ld\n\n",i_2);
        for (i=0; i<no_of_sites; i++)
        {
            long int check;
            {check = black_white_index[cluster[i]]; black_white_index[cluster[i]]+=1;}

            black_white_checkerboard[cluster[i]*no_of_black_white_sites[0]+check] = i;
            
        }
        printf("Checkerboard sites initialized. \n");
        return 0;
    }

    int initialize_checkerboard_sites()
    {
        long int i = 0;
        int j_L;
        
        // int dir_index_sum;
        
        // int black = 0;
        // int white = 1;
        int black_white[2] = { 0, 1 };
        
        // no_of_sites = 1;
        // for (j_L=0; j_L<dim_L; j_L++)
        // {
        //     no_of_sites = no_of_sites*lattice_size[j_L];
        // }
        
        /* if (no_of_sites % 2 == 1)
        {
            // no_of_black_sites = (no_of_sites + 1) / 2;
            no_of_black_white_sites[0] = (no_of_sites + 1) / 2;
            // no_of_white_sites = (no_of_sites - 1) / 2;
            no_of_black_white_sites[1] = (no_of_sites - 1) / 2;
        }
        else
        {
            // no_of_black_sites = no_of_sites / 2;
            no_of_black_white_sites[0] = no_of_sites / 2;
            // no_of_white_sites = no_of_sites / 2;
            no_of_black_white_sites[1] = no_of_sites / 2;
        }
        black_white_checkerboard[0] = (long int*)malloc(no_of_black_white_sites[0]*sizeof(long int));
        black_white_checkerboard[1] = (long int*)malloc(no_of_black_white_sites[1]*sizeof(long int));
         */

        // no_of_black_white_sites[0] = no_of_black_sites;
        // no_of_black_white_sites[1] = no_of_white_sites;
        
        // black_checkerboard = (long int*)malloc(no_of_black_sites*sizeof(long int));
        // white_checkerboard = (long int*)malloc(no_of_white_sites*sizeof(long int));
        // black_white_checkerboard[0] = black_checkerboard;
        // black_white_checkerboard[1] = white_checkerboard;


        // long int black_index = 0;
        // long int white_index = 0;
        long int black_white_index[2] = { 0, 0 };
        
        // omp_lock_t writelock0, writelock1;
        // omp_init_lock(&writelock0);
        // omp_init_lock(&writelock1);
        
        // #pragma omp parallel for private(j_L)
        for (i=0; i<no_of_sites; i++)
        {
            int site_to_dir_index[dim_L];
            long int dir_mult[dim_L];
            direction_index(i, site_to_dir_index, dir_mult);
            int dir_index_sum = 0;
            for (j_L=0; j_L<dim_L; j_L++)
            {
                dir_index_sum = dir_index_sum + site_to_dir_index[j_L];
            }
            dir_index_sum = dir_index_sum%2;
            // omp_set_lock(&writelock0);
            // #pragma omp critical
            long int check;
            // #pragma omp atomic capture
                {check = black_white_index[dir_index_sum]; black_white_index[dir_index_sum]++;}
            black_white_checkerboard[dir_index_sum*no_of_black_white_sites[0]+check] = i;
            /* {
                // black_checkerboard[black_index] = i;
                black_white_checkerboard[0+black_white_index[0]] = i;
                // if ( black_checkerboard[black_white_index[0]] - black_white_checkerboard[0][black_white_index[0]] != 0 )
                // {
                //     printf("black_checkerboard[i] = %ld, black_white_checkerboard[0][i] = %ld; %ld\n", black_checkerboard[black_white_index[0]], black_white_checkerboard[0][black_white_index[0]], black_checkerboard[black_white_index[0]] - black_white_checkerboard[0][black_white_index[0]] );
                // }
                black_white_index[0]++;
                // black_index++;
            } */

            // omp_unset_lock(&writelock0);
        }
        // omp_destroy_lock(&writelock0);
        // omp_destroy_lock(&writelock1);
        printf("Checkerboard sites initialized. \n");
        return 0;
    }

    int initialize_ordered_spin_config()
    {
        long int i;
        int j_S;

        #pragma omp parallel for private(j_S)
        for(i=0; i<no_of_sites; i=i+1)
        {
            for(j_S=0; j_S<dim_S; j_S=j_S+1)
            {
                spin[dim_S*i+j_S] = order[j_S];
            }
        }
        // initialize_nearest_neighbor_spin_sum();
        h_order = 0;
        r_order = 0;
        return 0;
    }

    int initialize_h_ordered_spin_config()
    {
        long int i;
        int j_S;

        initialize_ordered_spin_config();
        #pragma omp parallel for private(j_S)
        for(i=0; i<no_of_sites; i=i+1)
        {
            double h_mod = 0.0;
            for(j_S=0; j_S<dim_S; j_S=j_S+1)
            {
                h_mod = h_mod + h_random[dim_S*i+j_S]*h_random[dim_S*i+j_S];
            }
            if (h_mod != 0)
            {
                h_mod = sqrt(h_mod);
                for(j_S=0; j_S<dim_S; j_S=j_S+1)
                {
                    spin[dim_S*i+j_S] = h_random[dim_S*i+j_S] / h_mod;
                }
            }
            else
            {
                double h_dev_mod = 0.0;
                for(j_S=0; j_S<dim_S; j_S=j_S+1)
                {
                    h_dev_mod = h_dev_mod + h_dev_avg[j_S]*h_dev_avg[j_S];
                }
                if(h_dev_mod != 0)
                {
                    h_dev_mod = sqrt(h_dev_mod);
                    for(j_S=0; j_S<dim_S; j_S=j_S+1)
                    {
                        spin[dim_S*i+j_S] = h_dev_avg[j_S] / h_dev_mod;
                    }
                }
            }
        }
        // initialize_nearest_neighbor_spin_sum();
        h_order = 1;
        r_order = 0;
        return 0;
    }

    int initialize_random_spin_config()
    {
        long int i;
        int j_S;
        double limit = 0.01 * dim_S;
        #pragma omp parallel for private(j_S)
        for(i=0; i<no_of_sites; i=i+1)
        {
            double s_mod = 0.0;
            do
            {
                s_mod = 0.0;
                for(j_S=0; j_S<dim_S; j_S=j_S+1)
                {
                    // spin[dim_S*i+j_S] = (-1.0 + 2.0 * (double)rand_r(&random_seed[cache_size*thread_num_if_parallel()])/(double)(RAND_MAX));
                    spin[dim_S*i+j_S] = (-1.0 + 2.0 * genrand64_real1(thread_num_if_parallel()));
                    
                    s_mod = s_mod + spin[dim_S*i+j_S]*spin[dim_S*i+j_S];
                }
            }
            while(s_mod >= 1 || s_mod <=limit);

            s_mod = sqrt(s_mod);
            for(j_S=0; j_S<dim_S; j_S=j_S+1)
            {
                spin[dim_S*i+j_S] = spin[dim_S*i+j_S] / s_mod;
            }
        }
        // initialize_nearest_neighbor_spin_sum();
        
        h_order = 0;
        r_order = 1;
        return 0;
    }

    int initialize_spin_config()
    {
        int j_S;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            spin[dim_S*no_of_sites+j_S] = 0.0;
        }
        if (r_order == 1)
        {
            initialize_random_spin_config();
        }
        else
        {
            if (h_order == 1)
            {
                initialize_h_ordered_spin_config();
            }
            else
            {
                initialize_ordered_spin_config();
            }
        }
        return 0;
    }

    
//====================  Save J, h, Spin               ====================//

    int save_spin_config(char append_string[], char write_mode[], int reqd_to_print)
    {
        #ifdef enable_CUDA_CODE
            #ifdef CUDA_with_managed
            cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
            #else
            cudaMemcpyFromSymbol(spin, "dev_spin", dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
            #endif
        #endif

        long int i;
        int j_S, j_L;

        if (reqd_to_print*reqd_to_print == 1)
        {
            char output_file_1[256];
            char *pos = output_file_1;
            pos += sprintf(pos, "Spin(%d)_", dim_S);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%.3f", T);
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S) 
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            #endif
            pos += sprintf(pos, "_");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S) 
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h[j_S]);
            }
            strcat(output_file_1, append_string);
            strcat(output_file_1, ".dat");
                
            pFile_2 = fopen(output_file_1, write_mode); // opens new file for writing

            for (i = 0; i < no_of_sites; i++)
            {
                for (j_S = 0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_2, "%.17e\t", spin[dim_S*i + j_S]);
                }
                fprintf(pFile_2, "\n");
            }
            fclose(pFile_2);

            if (reqd_to_print == 1)
            {
                printf("Saved spin config. Output file name: %s", output_file_1);
                reqd_to_print = -reqd_to_print;
            }
        }
        else
        {
            if (reqd_to_print == 0)
            {
                char output_file_phase[256];
                char *pos_phase = output_file_phase;
                pos_phase += sprintf(pos_phase, "XY_Angle_");
                for (j_L = 0 ; j_L != dim_L ; j_L++) 
                {
                    if (j_L) 
                    {
                        pos_phase += sprintf(pos_phase, "x");
                    }
                    pos_phase += sprintf(pos_phase, "%d", lattice_size[j_L]);
                }
                pos_phase += sprintf(pos_phase, "_T%.3f", T);
                #ifdef RANDOM_FIELD
                pos_phase += sprintf(pos_phase, "_Rh");
                for (j_S = 0 ; j_S != dim_S ; j_S++) 
                {
                    if (j_S) 
                    {
                        pos_phase += sprintf(pos_phase, ",");
                    }
                    pos_phase += sprintf(pos_phase, "%.3f", sigma_h[j_S]);
                }
                #endif
                pos_phase += sprintf(pos_phase, "_h");
                for (j_S = 0 ; j_S != dim_S ; j_S++) 
                {
                    if (j_S) 
                    {
                        pos_phase += sprintf(pos_phase, ",");
                    }
                    pos_phase += sprintf(pos_phase, "%lf", h[j_S]);
                }
                
                strcat(output_file_phase, append_string);
                strcat(output_file_phase, ".dat");
                    
                pFile_phase = fopen(output_file_phase, write_mode); // opens new file for writing

                int ix, iy;
                double angle;
                for (ix = 0; ix < lattice_size[0]; ix++)
                {
                    for (iy = 0; iy < lattice_size[1]; iy++)
                    {
                        angle = ( (1.0-custom_double_sign(spin[dim_S*(lattice_size[0]*iy + ix) + 1])) + custom_double_sign(spin[dim_S*(lattice_size[0]*iy + ix) + 1]) * acos(spin[dim_S*(lattice_size[0]*iy + ix) + 0]) / pie ) / 2.0; 
                        fprintf(pFile_phase, "%.8e\t", angle );
                    }
                    fprintf(pFile_phase, "\n");
                }
                fclose(pFile_phase);
            }
            else
            {
                if (reqd_to_print == 2 && dim_L==2)
                {
                    char output_file_ising_spin[256];
                    char *pos_ising_spin = output_file_ising_spin;
                    pos_ising_spin += sprintf(pos_ising_spin, "Ising_Spin_");
                    for (j_L = 0 ; j_L != dim_L ; j_L++) 
                    {
                        if (j_L) 
                        {
                            pos_ising_spin += sprintf(pos_ising_spin, "x");
                        }
                        pos_ising_spin += sprintf(pos_ising_spin, "%d", lattice_size[j_L]);
                    }
                    pos_ising_spin += sprintf(pos_ising_spin, "_T%.3f", T);
                    #ifdef RANDOM_FIELD
                    pos_ising_spin += sprintf(pos_ising_spin, "_Rh");
                    for (j_S = 0 ; j_S != dim_S ; j_S++) 
                    {
                        if (j_S) 
                        {
                            pos_ising_spin += sprintf(pos_ising_spin, ",");
                        }
                        pos_ising_spin += sprintf(pos_ising_spin, "%.3f", sigma_h[j_S]);
                    }
                    #endif
                    pos_ising_spin += sprintf(pos_ising_spin, "_h");
                    for (j_S = 0 ; j_S != dim_S ; j_S++) 
                    {
                        if (j_S) 
                        {
                            pos_ising_spin += sprintf(pos_ising_spin, ",");
                        }
                        pos_ising_spin += sprintf(pos_ising_spin, "%lf", h[j_S]);
                    }
                        
                    int ix, iy;
                    int ising_spin;
                    #ifdef TRAINING_DATA
                        int it;
                        for (it=0; it<16; it++)
                        {
                            sprintf(pos_ising_spin, "%s_%g%g_%d.csv", append_string, BC[0], BC[1], it);
                            pFile_train_data[it] = fopen(output_file_ising_spin,"a");
                        }
                        
                        
                        for (iy = 0; iy < lattice_size[1]; iy++)
                        {
                            for (ix = 0; ix < lattice_size[0]; ix++)
                            {
                                int iy_f = (lattice_size[1]-1-iy);
                                int ix_f = (lattice_size[0]-1-ix);
                                ising_spin = (int)( spin[dim_S*(lattice_size[0]*iy + ix)] + 1) / 2; 
                                fprintf(pFile_train_data[0], "%d,", ising_spin );
                                fprintf(pFile_train_data[1], "%d,", 1-ising_spin );
                                ising_spin = (int)( spin[dim_S*(lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                fprintf(pFile_train_data[2], "%d,", ising_spin );
                                fprintf(pFile_train_data[3], "%d,", 1-ising_spin );
                                ising_spin = (int)( spin[dim_S*(lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                fprintf(pFile_train_data[4], "%d,", ising_spin );
                                fprintf(pFile_train_data[5], "%d,", 1-ising_spin );
                                ising_spin = (int)( spin[dim_S*(lattice_size[0]*iy_f + ix_f)] + 1) / 2; 
                                fprintf(pFile_train_data[6], "%d,", ising_spin );
                                fprintf(pFile_train_data[7], "%d,", 1-ising_spin );
                            }
                        }
                        for (ix = 0; ix < lattice_size[0]; ix++)
                        {
                            for (iy = 0; iy < lattice_size[1]; iy++)
                            {
                                int iy_f = (lattice_size[1]-1-iy);
                                int ix_f = (lattice_size[0]-1-ix);
                                ising_spin = (int)( spin[dim_S*(lattice_size[0]*iy + ix)] + 1) / 2; 
                                fprintf(pFile_train_data[8+0], "%d,", ising_spin );
                                fprintf(pFile_train_data[8+1], "%d,", 1-ising_spin );
                                ising_spin = (int)( spin[dim_S*(lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                fprintf(pFile_train_data[8+2], "%d,", ising_spin );
                                fprintf(pFile_train_data[8+3], "%d,", 1-ising_spin );
                                ising_spin = (int)( spin[dim_S*(lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                fprintf(pFile_train_data[8+4], "%d,", ising_spin );
                                fprintf(pFile_train_data[8+5], "%d,", 1-ising_spin );
                                ising_spin = (int)( spin[dim_S*(lattice_size[0]*iy_f + ix_f)] + 1) / 2; 
                                fprintf(pFile_train_data[8+6], "%d,", ising_spin );
                                fprintf(pFile_train_data[8+7], "%d,", 1-ising_spin );
                            }
                        }

                        for (it=0; it<16; it++)
                        {
                            #ifdef C_IM
                                fprintf(pFile_train_data[it], "0,1,0,0,0,0,0\n"); // Clean Ising 2D
                            #endif
                            #ifdef RFIM
                                fprintf(pFile_train_data[it], "0,0,0,1,0,0,0\n"); // RFIM 2D
                            #endif
                            fclose(pFile_train_data[it]);
                        }
                    #else
                        sprintf(pos_ising_spin, "%s.dat", append_string);
                        pFile_ising_spin = fopen(output_file_ising_spin, write_mode); // opens new file for writing
                        for (iy = 0; iy < lattice_size[1]; iy++)
                        {
                            for (ix = 0; ix < lattice_size[0]; ix++)
                            {
                                ising_spin = (int)( spin[dim_S*(lattice_size[0]*iy + ix)] + 1) / 2; 
                                fprintf(pFile_ising_spin, "%d\t", ising_spin );
                            }
                            fprintf(pFile_ising_spin, "\n");
                        }
                        fclose(pFile_ising_spin);
                    
                    #endif
                }
                if (reqd_to_print == 2 && dim_L==3)
                {
                    char output_file_ising_spin[256];
                    char *pos_ising_spin = output_file_ising_spin;
                    pos_ising_spin += sprintf(pos_ising_spin, "Ising_Spin_");
                    for (j_L = 0 ; j_L != dim_L ; j_L++) 
                    {
                        if (j_L) 
                        {
                            pos_ising_spin += sprintf(pos_ising_spin, "x");
                        }
                        pos_ising_spin += sprintf(pos_ising_spin, "%d", lattice_size[j_L]);
                    }
                    pos_ising_spin += sprintf(pos_ising_spin, "_T%.3f", T);
                    #ifdef RANDOM_FIELD
                    pos_ising_spin += sprintf(pos_ising_spin, "_Rh");
                    for (j_S = 0 ; j_S != dim_S ; j_S++) 
                    {
                        if (j_S) 
                        {
                            pos_ising_spin += sprintf(pos_ising_spin, ",");
                        }
                        pos_ising_spin += sprintf(pos_ising_spin, "%.3f", sigma_h[j_S]);
                    }
                    #endif
                    pos_ising_spin += sprintf(pos_ising_spin, "_h");
                    for (j_S = 0 ; j_S != dim_S ; j_S++) 
                    {
                        if (j_S) 
                        {
                            pos_ising_spin += sprintf(pos_ising_spin, ",");
                        }
                        pos_ising_spin += sprintf(pos_ising_spin, "%lf", h[j_S]);
                    }

                    int ix, iy, iz;
                    int ising_spin;
                    #ifdef TRAINING_DATA
                        { // if (BC[0]==0)
                            int it;
                            for (it=0; it<16; it++)
                            {
                                sprintf(pos_ising_spin, "%s_b0_%g%g%g_%d.csv", append_string, BC[0], BC[1], BC[2], it);
                                pFile_train_data[it] = fopen(output_file_ising_spin,"a");
                            }

                            int ix_min = 0;
                            double m_min = 1;
                            for (ix=(lattice_size[0]/4)*(1-BC[0]); ix<lattice_size[0]; ix += lattice_size[0]/4){
                                double m_tmp = 0;
                                #pragma omp parallel for reduction(+:m_tmp)
                                for (iz = 0; iz < lattice_size[2]; iz++)
                                {
                                    for (iy = 0; iy < lattice_size[1]; iy++)
                                    {
                                        m_tmp += spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)]; 
                                    }
                                }
                                m_tmp = fabs(m_tmp)/no_of_sites;
                                if (m_tmp<m_min){
                                    m_min = m_tmp;
                                    ix_min = ix;
                                }
                            }
                            for (ix=ix_min; ix<lattice_size[0]; ix += lattice_size[0]){
                                for (iz = 0; iz < lattice_size[2]; iz++)
                                {
                                    for (iy = 0; iy < lattice_size[1]; iy++)
                                    {
                                        int iz_f = (lattice_size[2]-1-iz);
                                        int iy_f = (lattice_size[1]-1-iy);
                                        // int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (iy = 0; iy < lattice_size[1]; iy++)
                                {
                                    for (iz = 0; iz < lattice_size[2]; iz++)
                                    {
                                        int iz_f = (lattice_size[2]-1-iz);
                                        int iy_f = (lattice_size[1]-1-iy);
                                        // int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (it=0; it<16; it++)
                                {
                                    #ifdef C_IM
                                        fprintf(pFile_train_data[it], "0,0,1,0,0,0,0\n"); // Clean Ising 3D
                                    #endif
                                    #ifdef RFIM
                                        fprintf(pFile_train_data[it], "0,0,0,0,1,0,0\n"); // RFIM 3D
                                    #endif
                                }
                            }

                            for (it=0; it<16; it++)
                            {
                                fclose(pFile_train_data[it]);
                            }
                        }
                        { // if (BC[1]==0)
                            int it;
                            for (it=0; it<16; it++)
                            {
                                sprintf(pos_ising_spin, "%s_b1_%g%g%g_%d.csv", append_string, BC[0], BC[1], BC[2], it);
                                pFile_train_data[it] = fopen(output_file_ising_spin,"a");
                            }

                            int iy_min = 0;
                            double m_min = 1.0;
                            for (iy=(lattice_size[1]/4)*(1-BC[1]); iy<lattice_size[1]; iy += lattice_size[1]/4){
                                double m_tmp = 0;
                                #pragma omp parallel for reduction(+:m_tmp)
                                for (ix = 0; ix < lattice_size[0]; ix++)
                                {
                                    for (iz = 0; iz < lattice_size[2]; iz++)
                                    {
                                        m_tmp += spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)]; 
                                    }
                                }
                                m_tmp = fabs(m_tmp)/no_of_sites;
                                if (m_tmp<m_min){
                                    m_min = m_tmp;
                                    iy_min = iy;
                                }
                            }
                            for (iy=iy_min; iy<lattice_size[1]; iy += lattice_size[1]){
                                for (ix = 0; ix < lattice_size[0]; ix++)
                                {
                                    for (iz = 0; iz < lattice_size[2]; iz++)
                                    {
                                        int iz_f = (lattice_size[2]-1-iz);
                                        // int iy_f = (lattice_size[1]-1-iy);
                                        int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (iz = 0; iz < lattice_size[2]; iz++)
                                {
                                    for (ix = 0; ix < lattice_size[0]; ix++)
                                    {
                                        int iz_f = (lattice_size[2]-1-iz);
                                        // int iy_f = (lattice_size[1]-1-iy);
                                        int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (it=0; it<16; it++)
                                {
                                    #ifdef C_IM
                                        fprintf(pFile_train_data[it], "0,0,1,0,0,0,0\n"); // Clean Ising 3D
                                    #endif
                                    #ifdef RFIM
                                        fprintf(pFile_train_data[it], "0,0,0,0,1,0,0\n"); // RFIM 3D
                                    #endif
                                }
                            }

                            for (it=0; it<16; it++)
                            {
                                fclose(pFile_train_data[it]);
                            }
                        }
                        { // if (BC[2]==0)
                            int it;
                            for (it=0; it<16; it++)
                            {
                                sprintf(pos_ising_spin, "%s_b2_%g%g%g_%d.csv", append_string, BC[0], BC[1], BC[2], it);
                                pFile_train_data[it] = fopen(output_file_ising_spin,"a");
                            }

                            int iz_min = 0;
                            double m_min = 1.0;
                            for (iz=(lattice_size[2]/4)*(1-BC[2]); iz<lattice_size[2]; iz += lattice_size[2]/4){
                                double m_tmp = 0;
                                #pragma omp parallel for reduction(+:m_tmp)
                                for (iy = 0; iy < lattice_size[1]; iy++)
                                {
                                    for (ix = 0; ix < lattice_size[0]; ix++)
                                    {
                                        m_tmp += spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)]; 
                                    }
                                }
                                m_tmp = fabs(m_tmp)/no_of_sites;
                                if (m_tmp<m_min){
                                    m_min = m_tmp;
                                    iz_min = iz;
                                }
                            }
                            for (iz=iz_min; iz<lattice_size[2]; iz += lattice_size[2]){
                                for (iy = 0; iy < lattice_size[1]; iy++)
                                {
                                    for (ix = 0; ix < lattice_size[0]; ix++)
                                    {
                                        // int iz_f = (lattice_size[2]-1-iz);
                                        int iy_f = (lattice_size[1]-1-iy);
                                        int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (ix = 0; ix < lattice_size[0]; ix++)
                                {
                                    for (iy = 0; iy < lattice_size[1]; iy++)
                                    {
                                        // int iz_f = (lattice_size[2]-1-iz);
                                        int iy_f = (lattice_size[1]-1-iy);
                                        int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (it=0; it<16; it++)
                                {
                                    #ifdef C_IM
                                        fprintf(pFile_train_data[it], "0,0,1,0,0,0,0\n"); // Clean Ising 3D
                                    #endif
                                    #ifdef RFIM
                                        fprintf(pFile_train_data[it], "0,0,0,0,1,0,0\n"); // RFIM 3D
                                    #endif
                                }
                            }

                            for (it=0; it<16; it++)
                            {
                                fclose(pFile_train_data[it]);
                            }
                        }

                    
                        if (BC[0]==0){
                            int it;
                            for (it=0; it<16; it++)
                            {
                                sprintf(pos_ising_spin, "%s_s0_%g%g%g_%d.csv", append_string, BC[0], BC[1], BC[2], it);
                                pFile_train_data[it] = fopen(output_file_ising_spin,"a");
                            }

                            int ix_min = 0;
                            double m_min = 1.0;
                            for (ix=0; ix<lattice_size[0]; ix += lattice_size[0]-1){
                                double m_tmp = 0;
                                #pragma omp parallel for reduction(+:m_tmp)
                                for (iz = 0; iz < lattice_size[2]; iz++)
                                {
                                    for (iy = 0; iy < lattice_size[1]; iy++)
                                    {
                                        m_tmp += spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)]; 
                                    }
                                }
                                m_tmp = fabs(m_tmp)/no_of_sites;
                                if (m_tmp<m_min){
                                    m_min = m_tmp;
                                    ix_min = ix;
                                }
                            }
                            for (ix=ix_min; ix<lattice_size[0]; ix += lattice_size[0]){
                                for (iz = 0; iz < lattice_size[2]; iz++)
                                {
                                    for (iy = 0; iy < lattice_size[1]; iy++)
                                    {
                                        int iz_f = (lattice_size[2]-1-iz);
                                        int iy_f = (lattice_size[1]-1-iy);
                                        // int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (iy = 0; iy < lattice_size[1]; iy++)
                                {
                                    for (iz = 0; iz < lattice_size[2]; iz++)
                                    {
                                        int iz_f = (lattice_size[2]-1-iz);
                                        int iy_f = (lattice_size[1]-1-iy);
                                        // int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (it=0; it<16; it++)
                                {
                                    #ifdef C_IM
                                        fprintf(pFile_train_data[it], "0,0,1,0,0,0,0\n"); // Clean Ising 3D
                                    #endif
                                    #ifdef RFIM
                                        fprintf(pFile_train_data[it], "0,0,0,0,1,0,0\n"); // RFIM 3D
                                    #endif
                                }
                            }

                            for (it=0; it<16; it++)
                            {
                                fclose(pFile_train_data[it]);
                            }
                        }
                        if (BC[1]==0){
                            int it;
                            for (it=0; it<16; it++)
                            {
                                sprintf(pos_ising_spin, "%s_s1_%g%g%g_%d.csv", append_string, BC[0], BC[1], BC[2], it);
                                pFile_train_data[it] = fopen(output_file_ising_spin,"a");
                            }

                            int iy_min = 0;
                            double m_min = 1.0;
                            for (iy=0; iy<lattice_size[1]; iy += lattice_size[1]-1){
                                double m_tmp = 0;
                                #pragma omp parallel for reduction(+:m_tmp)
                                for (ix = 0; ix < lattice_size[0]; ix++)
                                {
                                    for (iz = 0; iz < lattice_size[2]; iz++)
                                    {
                                        m_tmp += spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)]; 
                                    }
                                }
                                m_tmp = fabs(m_tmp)/no_of_sites;
                                if (m_tmp<m_min){
                                    m_min = m_tmp;
                                    iy_min = iy;
                                }
                            }
                            for (iy=iy_min; iy<lattice_size[1]; iy += lattice_size[1]){
                                for (ix = 0; ix < lattice_size[0]; ix++)
                                {
                                    for (iz = 0; iz < lattice_size[2]; iz++)
                                    {
                                        int iz_f = (lattice_size[2]-1-iz);
                                        // int iy_f = (lattice_size[1]-1-iy);
                                        int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (iz = 0; iz < lattice_size[2]; iz++)
                                {
                                    for (ix = 0; ix < lattice_size[0]; ix++)
                                    {
                                        int iz_f = (lattice_size[2]-1-iz);
                                        // int iy_f = (lattice_size[1]-1-iy);
                                        int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz_f+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (it=0; it<16; it++)
                                {
                                    #ifdef C_IM
                                        fprintf(pFile_train_data[it], "0,0,1,0,0,0,0\n"); // Clean Ising 3D
                                    #endif
                                    #ifdef RFIM
                                        fprintf(pFile_train_data[it], "0,0,0,0,1,0,0\n"); // RFIM 3D
                                    #endif
                                }
                            }

                            for (it=0; it<16; it++)
                            {
                                fclose(pFile_train_data[it]);
                            }
                        }
                        if (BC[2]==0){
                            int it;
                            for (it=0; it<16; it++)
                            {
                                sprintf(pos_ising_spin, "%s_s2_%g%g%g_%d.csv", append_string, BC[0], BC[1], BC[2], it);
                                pFile_train_data[it] = fopen(output_file_ising_spin,"a");
                            }

                            int iz_min = 0;
                            double m_min = 1.0;
                            for (iz=0; iz<lattice_size[2]; iz += lattice_size[2]-1){
                                double m_tmp = 0;
                                #pragma omp parallel for reduction(+:m_tmp)
                                for (iy = 0; iy < lattice_size[1]; iy++)
                                {
                                    for (ix = 0; ix < lattice_size[0]; ix++)
                                    {
                                        m_tmp += spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)]; 
                                    }
                                }
                                m_tmp = fabs(m_tmp)/no_of_sites;
                                if (m_tmp<m_min){
                                    m_min = m_tmp;
                                    iz_min = iz;
                                }
                            }
                            for (iz=iz_min; iz<lattice_size[2]; iz += lattice_size[2]){
                                for (iy = 0; iy < lattice_size[1]; iy++)
                                {
                                    for (ix = 0; ix < lattice_size[0]; ix++)
                                    {
                                        // int iz_f = (lattice_size[2]-1-iz);
                                        int iy_f = (lattice_size[1]-1-iy);
                                        int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (ix = 0; ix < lattice_size[0]; ix++)
                                {
                                    for (iy = 0; iy < lattice_size[1]; iy++)
                                    {
                                        // int iz_f = (lattice_size[2]-1-iz);
                                        int iy_f = (lattice_size[1]-1-iy);
                                        int ix_f = (lattice_size[0]-1-ix);
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+0], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+1], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+2], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+3], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+4], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+5], "%d,", 1-ising_spin );
                                        ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy_f + ix_f)] + 1) / 2; 
                                        fprintf(pFile_train_data[8+6], "%d,", ising_spin );
                                        fprintf(pFile_train_data[8+7], "%d,", 1-ising_spin );
                                    }
                                }
                                for (it=0; it<16; it++)
                                {
                                    #ifdef C_IM
                                        fprintf(pFile_train_data[it], "0,0,1,0,0,0,0\n"); // Clean Ising 3D
                                    #endif
                                    #ifdef RFIM
                                        fprintf(pFile_train_data[it], "0,0,0,0,1,0,0\n"); // RFIM 3D
                                    #endif
                                }
                            }

                            for (it=0; it<16; it++)
                            {
                                fclose(pFile_train_data[it]);
                            }
                        }
                    #else
                        for (iz = 0; iz < lattice_size[2]-1; iz+=SLICE_STRIDE)
                        {
                            sprintf(pos_ising_spin, "%s_s%d.dat", append_string, iz);
                            pFile_ising_spin = fopen(output_file_ising_spin, write_mode); // opens new file for writing
                            for (iy = 0; iy < lattice_size[1]; iy++)
                            {
                                for (ix = 0; ix < lattice_size[0]; ix++)
                                {
                                    ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                    fprintf(pFile_ising_spin, "%d\t", ising_spin );
                                }
                                fprintf(pFile_ising_spin, "\n");
                            }
                            fclose(pFile_ising_spin);
                        }

                        iz = lattice_size[2]-1;
                        sprintf(pos_ising_spin, "%s_s%d.dat", append_string, iz);
                        pFile_ising_spin = fopen(output_file_ising_spin, write_mode); // opens new file for writing
                        for (iy = 0; iy < lattice_size[1]; iy++)
                        {
                            for (ix = 0; ix < lattice_size[0]; ix++)
                            {
                                ising_spin = (int)( spin[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] + 1) / 2; 
                                fprintf(pFile_ising_spin, "%d\t", ising_spin );
                            }
                            fprintf(pFile_ising_spin, "\n");
                        }
                        fclose(pFile_ising_spin);

                    #endif
                }
            }
            
        }
        
        /* for (i = 0; i < no_of_sites; i++)
        {
            for (j_S = 0; j_S<dim_S; j_S++)
            {
                printf("|%le|", spin[dim_S*i + j_S]);
            }
            printf("\n");
        }
        printf("\n"); */
        
        return 0;
    }

    int save_h_config(char append_string[])
    {
        long int i;
        int j_S, j_L;
        
        #ifdef RANDOM_FIELD
        char output_file_1[256];
        char *pos = output_file_1;
        pos += sprintf(pos, "h_config_");
        for (j_S = 0 ; j_S != dim_S ; j_S++) 
        {
            if (j_S) 
            {
                pos += sprintf(pos, ",");
            }
            pos += sprintf(pos, "%lf", sigma_h[j_S]);
        }
        pos += sprintf(pos, "_");
        for (j_L = 0 ; j_L != dim_L ; j_L++) 
        {
            if (j_L) 
            {
                pos += sprintf(pos, "x");
            }
            pos += sprintf(pos, "%d", lattice_size[j_L]);
        }

        #ifdef BIMODAL_FIELD
        pos += sprintf(pos, "_bimodal");
        #endif
        #ifdef GAUSSIAN_FIELD
        pos += sprintf(pos, "_gaussian");
        #endif
        
        strcat(output_file_1, append_string);
        strcat(output_file_1, ".dat");
            
        pFile_1 = fopen(output_file_1, "a"); // opens new file for writing
        
        fprintf(pFile_1, "%.12e\t", h_i_min);
        printf( "\nh_i_min=%lf ", h_i_min);
        // fprintf(pFile_1, "\n");
        fprintf(pFile_1, "%.12e\t", h_i_max);
        printf( "h_i_max=%lf \n", h_i_max);
        fprintf(pFile_1, "\n");

        for (j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", h[j_S]);
            fprintf(pFile_1, "%.12e\t", sigma_h[j_S]);
            printf( "sigma_h[%d]=%lf \n", j_S, sigma_h[j_S]);
            fprintf(pFile_1, "%.12e\t", h_dev_avg[j_S]);
            printf( "h_dev_avg[%d]=%lf \n", j_S, h_dev_avg[j_S]);
            fprintf(pFile_1, "\n");
        }
        fprintf(pFile_1, "\n");

        for (i = 0; i < no_of_sites; i++)
        {
            for (j_S = 0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", h_random[dim_S*i + j_S]);
            }
            fprintf(pFile_1, "\n");
            
        }
        fclose(pFile_1);

        if (dim_L>=2 && dim_S==1)
        {
            char output_file_ising_h[256];
            char *pos_ising_h = output_file_ising_h;
            pos_ising_h += sprintf(pos_ising_h, "Ising_h_");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S) 
                {
                    pos_ising_h += sprintf(pos_ising_h, ",");
                }
                pos_ising_h += sprintf(pos_ising_h, "%.3f", sigma_h[j_S]);
            }
            pos_ising_h += sprintf(pos_ising_h, "_");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos_ising_h += sprintf(pos_ising_h, "x");
                }
                pos_ising_h += sprintf(pos_ising_h, "%d", lattice_size[j_L]);
            }

            #ifdef BIMODAL_FIELD
            pos += sprintf(pos, "_bimodal");
            #endif
            #ifdef GAUSSIAN_FIELD
            pos += sprintf(pos, "_gaussian");
            #endif
            
            strcat(output_file_ising_h, append_string);
            strcat(output_file_ising_h, ".dat");
                
            pFile_ising_h = fopen(output_file_ising_h, "w"); // opens new file for writing

            if (dim_L>=3)
            {
                int ix, iy, iz;
                for (ix = 0; ix < lattice_size[0]; ix++)
                {
                    for (iy = 0; iy < lattice_size[1]; iy++)
                    {
                        for (iz = 0; iz < lattice_size[2]; iz++)
                        {
                            fprintf(pFile_ising_h, "%.12e\t", h_random[dim_S*(lattice_size[0]*lattice_size[1]*iz+lattice_size[0]*iy + ix)] );
                        }
                        fprintf(pFile_ising_h, "\n");
                    }
                    fprintf(pFile_ising_h, "\n");
                }
            }
            else
            {
                int ix, iy;
                for (ix = 0; ix < lattice_size[0]; ix++)
                {
                    for (iy = 0; iy < lattice_size[1]; iy++)
                    {
                        fprintf(pFile_ising_h, "%.12e\t", h_random[dim_S*(lattice_size[0]*iy + ix)] );
                    }
                    fprintf(pFile_ising_h, "\n");
                }
                fclose(pFile_ising_h);
            }
        }
        /* for (i = 0; i < no_of_sites; i++)
        {
            for (j_S = 0; j_S<dim_S; j_S++)
            {
                printf("|%lf|", h_random[dim_S*i + j_S]);
            }
            printf("\n");
        }
        printf("\n"); */
        #endif

        return 0;
    }

    int save_J_config(char append_string[])
    {
        long int i;
        int j_L, k_L;
        
        #ifdef RANDOM_BOND
        char output_file_1[256];
        char *pos = output_file_1;
        pos += sprintf(pos, "J_config_");
        for (j_L = 0 ; j_L != dim_L ; j_L++) 
        {
            if (j_L) 
            {
                pos += sprintf(pos, ",");
            }
            pos += sprintf(pos, "%lf", sigma_J[j_L]);
        }
        pos += sprintf(pos, "_");
        for (j_L = 0 ; j_L != dim_L ; j_L++) 
        {
            if (j_L) 
            {
                pos += sprintf(pos, "x");
            }
            pos += sprintf(pos, "%d", lattice_size[j_L]);
        }

        #ifdef BIMODAL_BOND
        pos += sprintf(pos, "_bimodal");
        #endif
        #ifdef GAUSSIAN_BOND
        pos += sprintf(pos, "_gaussian");
        #endif
        
        strcat(output_file_1, append_string);
        strcat(output_file_1, ".dat");
        
        pFile_1 = fopen(output_file_1, "a"); // opens new file for writing
        
        fprintf(pFile_1, "%.12e\t", J_i_min);
        // fprintf(pFile_1, "\n");
        fprintf(pFile_1, "%.12e\t", J_i_max);
        fprintf(pFile_1, "\n");

        for (j_L=0; j_L<dim_L; j_L++)
        {
            fprintf(pFile_1, "%.12e\t", J[j_L]);
            fprintf(pFile_1, "%.12e\t", sigma_J[j_L]);
            fprintf(pFile_1, "%.12e\t", J_dev_avg[j_L]);
            fprintf(pFile_1, "\n");
        }
        fprintf(pFile_1, "\n");

        for (i = 0; i < no_of_sites; i++)
        {
            for (j_L = 0; j_L<dim_L; j_L++)
            {
                for (k_L = 0; k_L<2; k_L++)
                {
                    fprintf(pFile_1, "%.12e\t", J_random[2*dim_L*i + 2*j_L + k_L]);
                }
            }
            fprintf(pFile_1, "\n");
        }
        fclose(pFile_1);

        /* for (i = 0; i < no_of_sites; i++)
        {
            for (j_L = 0; j_L<dim_L; j_L++)
            {
                for (k_L = 0; k_L<2; k_L++)
                {
                    printf("|%lf|", J_random[2*dim_L*i + 2*j_L + k_L]);
                }
            }
            printf("\n");
        }
        printf("\n"); */
        #endif

        return 0;
    }

//====================  Load J, h, Spin               ====================//

    int load_spin_config(char append_string[])
    {
        long int i;
        int j_S, j_L;

        char input_file_1[256];
        char *pos = input_file_1;
        pos += sprintf(pos, "Spin(%d)_", dim_S);
        for (j_L = 0 ; j_L != dim_L ; j_L++) 
        {
            if (j_L) 
            {
                pos += sprintf(pos, "x");
            }
            pos += sprintf(pos, "%d", lattice_size[j_L]);
        }
        pos += sprintf(pos, "_%.3f", T);
        #ifdef RANDOM_FIELD
        pos += sprintf(pos, "_");
        for (j_S = 0 ; j_S != dim_S ; j_S++) 
        {
            if (j_S) 
            {
                pos += sprintf(pos, ",");
            }
            pos += sprintf(pos, "%.3f", sigma_h[j_S]);
        }
        #endif
        pos += sprintf(pos, "_");
        for (j_S = 0 ; j_S != dim_S ; j_S++) 
        {
            if (j_S) 
            {
                pos += sprintf(pos, ",");
            }
            pos += sprintf(pos, "%lf", h[j_S]);
        }
        strcat(input_file_1, append_string);
        strcat(input_file_1, ".dat");
        // pos += sprintf(pos, ".dat");
            
        pFile_2 = fopen(input_file_1, "r"); // opens old file for reading
        
        if (pFile_2 == NULL)
        {
            initialize_spin_config();
            printf("Initialized spin config. No Input file name: %s\n", input_file_1);
        }
        else
        {
            
            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin[dim_S*no_of_sites+j_S] = 0.0;
            }

            for (i = 0; i < no_of_sites; i++)
            {
                for (j_S = 0; j_S<dim_S; j_S++)
                {
                    fscanf(pFile_2, "%le", &spin[dim_S*i + j_S]);
                }
            }
            fclose(pFile_2);
            printf("Loaded spin config. Input file name: %s\n", input_file_1);
        }


        /* for (i = 0; i < no_of_sites; i++)
        {
            for (j_S = 0; j_S<dim_S; j_S++)
            {
                printf("|%le|", spin[dim_S*i + j_S]);
            }
            printf("\n");
        }
        printf("\n"); */

        #ifdef enable_CUDA_CODE
            #ifdef CUDA_with_managed
            cudaMemcpy(dev_spin, spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #else
            cudaMemcpyToSymbol("dev_spin", spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
        #endif

        return 0;
    }

    int load_h_config(char append_string[])
    {
        //---------------------------------------------------------------------------------------//
        
        // h_random = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        long int i;
        int j_S, j_L;
        #ifdef RANDOM_FIELD
        char input_file_1[256];
        char *pos = input_file_1;
        pos += sprintf(pos, "h_config_");
        
        for (j_S = 0 ; j_S != dim_S ; j_S++) 
        {
            if (j_S) 
            {
                pos += sprintf(pos, ",");
            }
            pos += sprintf(pos, "%lf", sigma_h[j_S]);
        }
        pos += sprintf(pos, "_");
        for (j_L = 0 ; j_L != dim_L ; j_L++) 
        {
            if (j_L) 
            {
                pos += sprintf(pos, "x");
            }
            pos += sprintf(pos, "%d", lattice_size[j_L]);
        }

        #ifdef BIMODAL_FIELD
        pos += sprintf(pos, "_bimodal");
        #endif
        #ifdef GAUSSIAN_FIELD
        pos += sprintf(pos, "_gaussian");
        #endif
        
        strcat(input_file_1, append_string);
        strcat(input_file_1, ".dat");
        
        pFile_1 = fopen(input_file_1, "r"); // opens file for reading

        if (pFile_1 == NULL)
        {
            // h_random = (double*)malloc(dim_S*no_of_sites*sizeof(double));
            #ifdef GAUSSIAN_FIELD // for random gaussian fields
            initialize_h_random_gaussian();
            #endif
            #ifdef BIMODAL_FIELD // for random bimodal fields
            initialize_h_random_bimodal();
            #endif
            save_h_config(append_string); // creates file for later
            printf("Initialized h_random config. Output file name: %s\n", input_file_1);
        }
        else
        {
            // h_random = (double*)malloc(dim_S*no_of_sites*sizeof(double));
            fscanf(pFile_1, "%le", &h_i_min);
            fscanf(pFile_1, "%le", &h_i_max);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fscanf(pFile_1, "%le", &h[j_S]);
                fscanf(pFile_1, "%le", &sigma_h[j_S]);
                fscanf(pFile_1, "%le", &h_dev_avg[j_S]);
            }
            
            for (i = 0; i < no_of_sites; i++)
            {
                for (j_S = 0; j_S<dim_S; j_S++)
                {
                    fscanf(pFile_1, "%le", &h_random[dim_S*i + j_S]);
                }
            }
            fclose(pFile_1);
            printf("Loaded h_random config. Input file name: %s\n", input_file_1);
        }
        //---------------------------------------------------------------------------------------//
        /*
        for (i = 0; i < no_of_sites; i++)
        {
            for (j_S = 0; j_S<dim_S; j_S++)
            {
                printf("|%lf|", h_random[dim_S*i + j_S]);
            }
            printf("\n");
        }
        printf("\n");
        */
        #endif
        return 0;
    }

    int load_J_config(char append_string[])
    {
        //---------------------------------------------------------------------------------------//
        
        // J_random = (double*)malloc(2*dim_L*no_of_sites*sizeof(double));
        long int i;
        int j_L, k_L;
        #ifdef RANDOM_BOND
        char input_file_1[256];
        char *pos = input_file_1;
        pos += sprintf(pos, "J_config_");

        for (j_L = 0 ; j_L != dim_L ; j_L++) 
        {
            if (j_L) 
            {
                pos += sprintf(pos, ",");
            }
            pos += sprintf(pos, "%lf", sigma_J[j_L]);
        }
        pos += sprintf(pos, "_");
        for (j_L = 0 ; j_L != dim_L ; j_L++) 
        {
            if (j_L) 
            {
                pos += sprintf(pos, "x");
            }
            pos += sprintf(pos, "%d", lattice_size[j_L]);
        }

        #ifdef BIMODAL_BOND
        pos += sprintf(pos, "_bimodal");
        #endif
        #ifdef GAUSSIAN_BOND
        pos += sprintf(pos, "_gaussian");
        #endif

        strcat(input_file_1, append_string);
        strcat(input_file_1, ".dat");
        
        pFile_1 = fopen(input_file_1, "r"); // opens file for reading

        if (pFile_1 == NULL)
        {
            // J_random = (double*)malloc(2*dim_L*no_of_sites*sizeof(double));
            #ifdef GAUSSIAN_BOND // for random gaussian bonds
            initialize_J_random_gaussian();
            #endif
            #ifdef BIMODAL_BOND // for random bimodal bonds
            initialize_J_random_bimodal();
            #endif

            save_J_config(append_string); // creates file for later
            printf("Initialized J_random config. Output file name: %s\n", input_file_1);
        }
        else
        {
            // J_random = (double*)malloc(2*dim_L*no_of_sites*sizeof(double));
            
            // for (j_L=0; j_L<dim_S; j_L++)
            // {
            //     for (k_L=0; k_L<dim_S; k_L++)
            //     {
            //         J_random[2*dim_L*no_of_sites+2*j_L+k_L] = 0.0;
            //     }
            // }

            fscanf(pFile_1, "%le", &J_i_min);
            fscanf(pFile_1, "%le", &J_i_max);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fscanf(pFile_1, "%le", &J[j_L]);
                fscanf(pFile_1, "%le", &sigma_J[j_L]);
                fscanf(pFile_1, "%le", &J_dev_avg[j_L]);
            }
            
            for (i = 0; i < no_of_sites; i++)
            {
                for (j_L = 0; j_L<dim_L; j_L++)
                {
                    for (k_L = 0; k_L<2; k_L++)
                    {
                        fscanf(pFile_1, "%le", &J_random[2*dim_L*i + 2*j_L + k_L]);
                    }
                }
            }
            fclose(pFile_1);
            printf("Loaded J_random config. Input file name: %s\n", input_file_1);
        }
        //---------------------------------------------------------------------------------------//
        /*
        for (i = 0; i < no_of_sites; i++)
        {
            for (j_L = 0; j_L<dim_L; j_L++)
            {
                for (k_L = 0; k_L<2; k_L++)
                {
                    printf("|%lf|", J_random[2*dim_L*i + 2*j_L + k_L]);
                }
            }
            printf("\n");
        }
        printf("\n");
        */
        #endif

        return 0;
    }

//====================  Checkpoint                    ====================//

    int restore_checkpoint(int startif, int array_type, int array_length, void *voidarray, int stopif)
    {
        int j_arr, j_L, j_S;
        static int restore_point_exist = 1;
        if (startif == -1)
        {
            restore_point_exist = 1;
            return 0;
        }
        if (restore_point_exist == 0)
        {
            return 0;
        }
        if (startif == 1)
        {
            char chkpt_file_1[256];
            char *pos = chkpt_file_1;
            pos += sprintf(pos, "Data_");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%.3f", T);
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S) 
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            #endif
            
            if (array_type == TYPE_INT)
            {
                pos += sprintf(pos, "_");
                int *array = (int *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, ",");
                    }
                    pos += sprintf(pos, "%d", array[j_arr]);
                }
            }
            if (array_type == TYPE_LONGINT)
            {
                pos += sprintf(pos, "_");
                long int *array = (long int *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, ",");
                    }
                    pos += sprintf(pos, "%ld", array[j_arr]);
                }
            }
            if (array_type == TYPE_FLOAT)
            {
                pos += sprintf(pos, "_");
                float *array = (float *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, ",");
                    }
                    pos += sprintf(pos, "%f", array[j_arr]);
                }
            }
            if (array_type == TYPE_DOUBLE)
            {
                pos += sprintf(pos, "_");
                double *array = (double *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, ",");
                    }
                    pos += sprintf(pos, "%lf", array[j_arr]);
                }
            }
            // strcat(chkpt_file_1, append_string);
            strcat(chkpt_file_1, "_chkpt.dat");
                
            pFile_chkpt = fopen(chkpt_file_1, "r"); // opens new file for reading
            if (pFile_chkpt == NULL)
            {
                restore_point_exist = 0;
                printf("\n---- Starting from Initial Conditions ----\n");
                return 0;
            }
            else
            {
                load_spin_config("_chkpt");
            }
        }
        else
        {
            if (array_type == TYPE_INT)
            {
                int *array = (int *)voidarray;
                for (j_arr=0; j_arr<array_length; j_arr++)
                {
                    fscanf(pFile_chkpt, "%d", &array[j_arr]);
                }
            }
            if (array_type == TYPE_LONGINT)
            {
                long int *array = (long int *)voidarray;
                for (j_arr=0; j_arr<array_length; j_arr++)
                {
                    fscanf(pFile_chkpt, "%ld", &array[j_arr]);
                }
            }
            if (array_type == TYPE_FLOAT)
            {
                float *array = (float *)voidarray;
                for (j_arr=0; j_arr<array_length; j_arr++)
                {
                    fscanf(pFile_chkpt, "%e", &array[j_arr]);
                }
            }
            if (array_type == TYPE_DOUBLE)
            {
                double *array = (double *)voidarray;
                for (j_arr=0; j_arr<array_length; j_arr++)
                {
                    fscanf(pFile_chkpt, "%le", &array[j_arr]);
                }
            }
            // if (array_type != TYPE_VOID)
            // {
            //     fprintf(pFile_chkpt, "\n");
            // }
            
        }

        if (stopif == 1)
        {
            fclose(pFile_chkpt);
            printf("\n---- Resuming from Checkpoint ----\n");
        }

        return 1;
    }

    int checkpoint_backup(int startif, int array_type, int array_length, void *voidarray, int stopif)
    {
        int j_arr, j_L, j_S;
        if (startif == 1)
        {
            char chkpt_file_1[256];
            char *pos = chkpt_file_1;
            pos += sprintf(pos, "Data_");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%.3f", T);
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S) 
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            #endif

            if (array_type == TYPE_INT)
            {
                pos += sprintf(pos, "_");
                int *array = (int *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, ",");
                    }
                    pos += sprintf(pos, "%d", array[j_arr]);
                }
            }
            if (array_type == TYPE_LONGINT)
            {
                pos += sprintf(pos, "_");
                long int *array = (long int *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, ",");
                    }
                    pos += sprintf(pos, "%ld", array[j_arr]);
                }
            }
            if (array_type == TYPE_FLOAT)
            {
                pos += sprintf(pos, "_");
                float *array = (float *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, ",");
                    }
                    pos += sprintf(pos, "%f", array[j_arr]);
                }
            }
            if (array_type == TYPE_DOUBLE)
            {
                pos += sprintf(pos, "_");
                double *array = (double *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, ",");
                    }
                    pos += sprintf(pos, "%lf", array[j_arr]);
                }
            }
            // strcat(chkpt_file_1, append_string);
            strcat(chkpt_file_1, "_chkpt.dat");
                
            pFile_chkpt = fopen(chkpt_file_1, "w"); // opens new file for writing
            
            save_spin_config("_chkpt", "w", 1);
        }
        else
        {
            if (array_type == TYPE_INT)
            {
                int *array = (int *)voidarray;
                for (j_arr=0; j_arr<array_length; j_arr++)
                {
                    fprintf(pFile_chkpt, "%d\t", array[j_arr]);
                }
            }
            if (array_type == TYPE_LONGINT)
            {
                long int *array = (long int *)voidarray;
                for (j_arr=0; j_arr<array_length; j_arr++)
                {
                    fprintf(pFile_chkpt, "%ld\t", array[j_arr]);
                }
            }
            if (array_type == TYPE_FLOAT)
            {
                float *array = (float *)voidarray;
                for (j_arr=0; j_arr<array_length; j_arr++)
                {
                    fprintf(pFile_chkpt, "%.9e\t", array[j_arr]);
                }
            }
            if (array_type == TYPE_DOUBLE)
            {
                double *array = (double *)voidarray;
                for (j_arr=0; j_arr<array_length; j_arr++)
                {
                    fprintf(pFile_chkpt, "%.17e\t", array[j_arr]);
                }
            }
            if (array_type != TYPE_VOID)
            {
                fprintf(pFile_chkpt, "\n");
            }
        }

        if (stopif == 1)
        {
            fclose(pFile_chkpt);
            double time_now = get_time_if_parallel();
            printf("\n---- Checkpoint after %lf seconds ----\n", time_now - start_time);
            return 1;
        }

        return 0;
    }

//====================  Print Output                  ====================//

    int output_param_file(char output_file_name[])
    {
        int j_L, j_S;
        printf("\nOutput file name: %s\n", output_file_name);
        pFile_output = fopen(output_file_name, "a");
        fprintf(pFile_output, "##################################################################################\n");
        
        fprintf(pFile_output, "dim_{Spin} = %d ,\n", dim_S);

        fprintf(pFile_output, "dim_{Lat} = %d ,\n", dim_L);

        fprintf(pFile_output, "N = (");
        for (j_L=0; j_L<dim_L; j_L++)
        {
            if (j_L)
            {
                fprintf(pFile_output, "x");
            }
            fprintf(pFile_output, " %d ", lattice_size[j_L]);
        }
        fprintf(pFile_output, ") ,\n");
        
        fprintf(pFile_output, "Boundary Condition(s) = (");
        for (j_L=0; j_L<dim_L; j_L++)
        {
            if (j_L)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.1e ", BC[j_L]);
        }
        fprintf(pFile_output, ") ,\n");

        fprintf(pFile_output, "T = %.12e ,\n", T);
        fprintf(pFile_output, "T_min = %.12e ,\n", Temp_min);
        fprintf(pFile_output, "T_max = %.12e ,\n", Temp_max);
        
        fprintf(pFile_output, "delta_T = %.12e ,\n", delta_T);

        fprintf(pFile_output, "J = (");
        for (j_L=0; j_L<dim_L; j_L++)
        {
            if (j_L)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.12e ", J[j_L]);
        }
        fprintf(pFile_output, ") ,\n");

        #ifdef RANDOM_BOND
        fprintf(pFile_output, "sigma_J = (");
        for (j_L=0; j_L<dim_L; j_L++)
        {
            if (j_L)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.12e ", sigma_J[j_L]);
        }
        fprintf(pFile_output, ") , ");
        fprintf(pFile_output, "<J_{ij}> = (");
        for (j_L=0; j_L<dim_L; j_L++)
        {
            if (j_L)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.12e ", J_dev_avg[j_L]);
        }
        fprintf(pFile_output, ") , ");
        #ifdef BIMODAL_BOND
        fprintf(pFile_output, " [Bimodal distribution] ,\n");
        #endif
        #ifdef GAUSSIAN_BOND
        fprintf(pFile_output, " [Gaussian distribution] ,\n");
        #endif
        #endif

        fprintf(pFile_output, "h = (");
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.12e ", h[j_S]);
        }
        fprintf(pFile_output, ") ,\n");
        
        #ifdef RANDOM_FIELD
        fprintf(pFile_output, "sigma_h = (");
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.12e ", sigma_h[j_S]);
        }
        fprintf(pFile_output, ") , ");
        fprintf(pFile_output, "<h_i> = (");
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.12e ", h_dev_avg[j_S]);
        }
        fprintf(pFile_output, ") , ");
        #ifdef BIMODAL_FIELD
        fprintf(pFile_output, " [Bimodal distribution] ,\n");
        #endif
        #ifdef GAUSSIAN_FIELD
        fprintf(pFile_output, " [Gaussian distribution] ,\n");
        #endif
        #endif

        fprintf(pFile_output, "Spin Initialization - ");
        if( r_order==1 )
        {
            fprintf(pFile_output, "Random ,\n");
        }
        else
        {
            if ( h_order==1 )
            {
                fprintf(pFile_output, "Along Field ,\n");
            }
            else if ( h_order==0 && r_order==0 )
            {
                fprintf(pFile_output, "Ordered along = (");
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    if (j_S)
                    {
                        fprintf(pFile_output, ",");
                    }
                    fprintf(pFile_output, " %.12e ", order[j_S]);
                }
                fprintf(pFile_output, ") ,\n");
            }
        }
        

        fprintf(pFile_output, "Thermalizing-MCS = %ld ,\n", thermal_i);

        if (MC_algo_type_thrm == 2)
        {
            if (MC_update_type_thrm == 0)
            {
                fprintf(pFile_output, "Swendsen-Wang cluster update ,\n");
            }
            else
            {
                fprintf(pFile_output, "Wolff cluster update ,\n");
            }
        }
        else if (MC_algo_type_thrm == 1)
        {
            fprintf(pFile_output, "Metropolis - ");
            if (MC_update_type_thrm == 0)
            {
                fprintf(pFile_output, "Checkerboard site update ,\n");
            }
            else if (MC_update_type_thrm == 1)
            {
                fprintf(pFile_output, "Random site update ,\n");
            }
            else if (MC_update_type_thrm == 2)
            {
                fprintf(pFile_output, "Linear site update ,\n");
            }
        }
        else if (MC_algo_type_thrm == 0)
        {
            fprintf(pFile_output, "Glauber - ");
            if (MC_update_type_thrm == 0)
            {
                fprintf(pFile_output, "Checkerboard site update ,\n");
            }
            else if (MC_update_type_thrm == 1)
            {
                fprintf(pFile_output, "Random site update ,\n");
            }
            else if (MC_update_type_thrm == 2)
            {
                fprintf(pFile_output, "Linear site update ,\n");
            }
        }

        fprintf(pFile_output, "Averaging-MCS = %ld x [1-%ld] ,\n", average_j, sampling_inter);

        if (MC_algo_type_avg == 2)
        {
            if (MC_update_type_avg == 0)
            {
                fprintf(pFile_output, "Swendsen-Wang cluster update ,\n");
            }
            else
            {
                fprintf(pFile_output, "Wolff cluster update ,\n");
            }
        }
        else if (MC_algo_type_avg == 1)
        {
            fprintf(pFile_output, "Metropolis - ");
            if (MC_update_type_avg == 0)
            {
                fprintf(pFile_output, "Checkerboard site update ,\n");
            }
            else if (MC_update_type_avg == 1)
            {
                fprintf(pFile_output, "Random site update ,\n");
            }
            else if (MC_update_type_avg == 2)
            {
                fprintf(pFile_output, "Linear site update ,\n");
            }
        }
        else if (MC_algo_type_avg == 0)
        {
            fprintf(pFile_output, "Glauber - ");
            if (MC_update_type_avg == 0)
            {
                fprintf(pFile_output, "Checkerboard site update ,\n");
            }
            else if (MC_update_type_avg == 1)
            {
                fprintf(pFile_output, "Random site update ,\n");
            }
            else if (MC_update_type_avg == 2)
            {
                fprintf(pFile_output, "Linear site update ,\n");
            }
        }

        fprintf(pFile_output, "hysteresis-MCS/step = %ld, \n", hysteresis_MCS);
        
        // fprintf(pFile_output, "del_h = range_h x %.12e ,\n", del_h);
        fprintf(pFile_output, "del_h = %.12e [ cutoff_{d_h} = %.12e ] ,\n", del_h, del_h_cutoff);
        fprintf(pFile_output, "delta_{phi} = %.12e [ cutoff_{d_phi} = %.12e ] ,\n", del_phi, del_phi_cutoff);

        fprintf(pFile_output, "##################################################################################\n");

        // for (i=0; i<whatever; i++)
        // {
        //     fprintf(pFile_output, "%s", column_head[i]);
        // }

        // fprintf(pFile_output, "----------------------------------------------------------------------------------\n");

        
        fclose(pFile_output);
        pFile_output = NULL;
        return 0;
    }

    int output_label(char output_file_name[], char prepend_column[], char append_column[])
    {
        int j_L, j_S, j_SS;
        // printf("\nOutput file name: %s\n", output_file_name);
        pFile_output = fopen(output_file_name, "a");
        
        if (output_prepend == 1)
        {
            fprintf(pFile_output, "%s", prepend_column);
        }
        if (output_J == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "J[%d]\t", j_L);
            }
        }
        if (output_sigma_J == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "sigma{J[%d]}\t", j_L);
            }
        }
        if (output_h == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "h[%d]\t", j_S);
            }
        }
        if (output_sigma_h == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "sigma{h[%d]}\t", j_S);
            }
        }
        if (output_T == 1)
        {
            fprintf(pFile_output, "T\t");
        }
        
        if (output_abs_m == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "<<|S[%d]|>>\t", j_S);
            }
        }

        if (output_m == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "m[%d]\t", j_S);
            }
        }
        if (output_m_avg == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "<m[%d]>\t", j_S);
            }
        }
        if (output_m_abs_avg == 1)
        {
            fprintf(pFile_output, "<|m|>\t");
        }
        if (output_m_2_avg == 1)
        {
            fprintf(pFile_output, "<m^2>\t");
        }
        if (output_m_4_avg == 1)
        {
            fprintf(pFile_output, "<m^4>\t");
        }

        if (output_m_abs_vec_avg == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "<|m[%d]|>\t", j_S);
            }
        }
        if (output_m_2_vec_avg == 1 && output_m_ab_avg != 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "<m[%d]^2>\t", j_S);
            }
        }
        if (output_m_4_vec_avg == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "<m[%d]^4>\t", j_S);
            }
        }

        if (output_m_ab_avg == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    fprintf(pFile_output, "<m[%d]m[%d]>\t", j_S, j_SS);
                }
            }
        }

        if (output_delta_S_squared == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "<delta{S[%d]^2}>\t", j_S);
            }
        }
        if (output_delta_S_abs == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "<|delta{S[%d]}|>\t", j_S);
            }
        }
        if (output_delta_S_max == 1)
        {
                fprintf(pFile_output, "max(|delta{S}|)\t");
        }

        if (output_delta_M == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "delta{m[%d]}\t", j_S);
            }
        }

        if (output_X == 1)
        {
            fprintf(pFile_output, "X\t");
        }
        if (output_X_ab == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    fprintf(pFile_output, "X[%d][%d]\t", j_S, j_SS);
                }
            }
        }

        if (output_B == 1)
        {
            fprintf(pFile_output, "B\t");
        }
        if (output_B_a == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "B[%d]\t", j_S);
            }
        }

        if (output_E == 1)
        {
            fprintf(pFile_output, "E\t");
        }
        if (output_E_avg == 1)
        {
            fprintf(pFile_output, "<E>\t");
        }
        if (output_E_2_avg == 1)
        {
            fprintf(pFile_output, "<E^2>\t");
        }
        
        if (output_delta_E == 1)
        {
            fprintf(pFile_output, "delta{E}\t");
        }

        if (output_Cv == 1)
        {
            fprintf(pFile_output, "Cv\t");
        }

        if (output_Y_1 == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "Y_1{%d}\t", j_L);
            }
        }
        if (output_Y_2 == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "Y_2{%d}\t", j_L);
            }
        }
        if (output_Y_1_avg == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "<Y_1{%d}>\t", j_L);
            }
        }
        if (output_Y_2_avg == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "<Y_2{%d}>\t", j_L);
            }
        }
        if (output_Y_x_mu == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "Y_mu{%d}\t", j_L);
            }
        }
        if (output_append == 1)
        {
            fprintf(pFile_output, "%s", append_column);
        }
        fprintf(pFile_output, "\n==================================================================================\n");

        fclose(pFile_output);
        pFile_output = NULL;
        return 0;
    }
    
    int output_data(char output_file_name[], char prepend_column[], char append_column[])
    {
        int j_L, j_S, j_SS;
        // printf("\nOutput file name: %s\n", output_file_name);
        int close_file = 0;
        if (pFile_output == NULL)
        {
            close_file = 1;
            pFile_output = fopen(output_file_name, "a");
        }
        
        if (output_prepend == 1)
        {
            fprintf(pFile_output, "%s", prepend_column);
        }
        if (output_J == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "%.17e\t", J[j_L]);
            }
        }
        if (output_sigma_J == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "%.17e\t", sigma_J[j_L]);
            }
        }
        if (output_h == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", h[j_S]);
            }
        }
        if (output_sigma_h == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", sigma_h[j_S]);
            }
        }
        if (output_T == 1)
        {
            fprintf(pFile_output, "%.17e\t", T);
        }
        
        if (output_abs_m == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", abs_m[j_S]);
            }
        }

        if (output_m == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", m[j_S]);
            }
        }
        if (output_m_avg == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", m_avg[j_S]);
            }
        }
        if (output_m_abs_avg == 1)
        {
            fprintf(pFile_output, "%.17e\t", m_abs_avg);
        }
        if (output_m_2_avg == 1)
        {
            fprintf(pFile_output, "%.17e\t", m_2_avg);
        }
        if (output_m_4_avg == 1)
        {
            fprintf(pFile_output, "%.17e\t", m_4_avg);
        }

        if (output_m_abs_vec_avg == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", m_abs_vec_avg[j_S]);
            }
        }
        if (output_m_2_vec_avg == 1 && output_m_ab_avg != 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", m_2_vec_avg[j_S]);
            }
        }
        if (output_m_4_vec_avg == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", m_4_vec_avg[j_S]);
            }
        }

        if (output_m_ab_avg == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    fprintf(pFile_output, "%.17e\t", m_ab_avg[j_S*dim_S + j_SS]);
                }
            }
        }

        if (output_delta_S_squared == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", delta_S_squared[j_S]);
            }
        }
        if (output_delta_S_abs == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", delta_S_abs[j_S]);
            }
        }
        if (output_delta_S_max == 1)
        {
                fprintf(pFile_output, "%.17e\t", delta_S_max);
        }

        if (output_delta_M == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", delta_M[j_S]);
            }
        }

        if (output_X == 1)
        {
            fprintf(pFile_output, "%.17e\t", X);
        }
        if (output_X_ab == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    fprintf(pFile_output, "%.17e\t", X_ab[j_S*dim_S + j_SS]);
                }
            }
        }

        if (output_B == 1)
        {
            fprintf(pFile_output, "%.17e\t", B);
        }
        if (output_B_a == 1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_output, "%.17e\t", B_a[j_S]);
            }
        }

        if (output_E == 1)
        {
            fprintf(pFile_output, "%.17e\t", E);
        }
        if (output_E_avg == 1)
        {
            fprintf(pFile_output, "%.17e\t", E_avg);
        }
        if (output_E_2_avg == 1)
        {
            fprintf(pFile_output, "%.17e\t", E_2_avg);
        }

        if (output_delta_E == 1)
        {
            fprintf(pFile_output, "%.17e\t", delta_E);
        }
        
        if (output_Cv == 1)
        {
            fprintf(pFile_output, "%.17e\t", Cv);
        }
        if (output_Y_1 == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "%.17e\t", Y_1[j_L]);
            }
        }
        if (output_Y_2 == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "%.17e\t", Y_2[j_L]);
            }
        }
        if (output_Y_1_avg == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "%.17e\t", Y_1_avg[j_L]);
            }
        }
        if (output_Y_2_avg == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "%.17e\t", Y_2_avg[j_L]);
            }
        }
        if (output_Y_x_mu == 1)
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fprintf(pFile_output, "%.17e\t", Y_x_mu[j_L]);
            }
        }
        if (output_append == 1)
        {
            fprintf(pFile_output, "%s", append_column);
        }
        fprintf(pFile_output, "\n");

        if (close_file == 1)
        {
            fclose(pFile_output);
            pFile_output = NULL;
        }
        
        return 0;
    }

    int reset_output_variable_name_0()
    {
        output_J = 0;
        output_sigma_J = 0;
        output_h = 0;
        output_sigma_h = 0;
        output_T = 0;
        output_abs_m = 0;
        output_m = 0;
        output_m_avg = 0;
        output_m_abs_avg = 0;
        output_m_2_avg = 0;
        output_m_4_avg = 0;
        output_m_abs_vec_avg = 0;
        output_m_2_vec_avg = 0;
        output_m_4_vec_avg = 0;
        output_m_ab_avg = 0;
        output_delta_S_squared = 0;
        output_delta_S_abs = 0;
        output_delta_S_max = 0;
        output_delta_M = 0;
        output_X = 0;
        output_X_ab = 0;
        output_B = 0;
        output_B_a = 0;
        output_E = 0;
        output_E_avg = 0;
        output_E_2_avg = 0;
        output_delta_E = 0;
        output_Cv = 0;
        output_Y_1 = 0;
        output_Y_2 = 0;
        output_Y_1_avg = 0;
        output_Y_2_avg = 0;
        output_Y_x_mu = 0;
        output_prepend = 1;
        output_append = 1;

        return 0;
    }

//====================  Avalanche delta(S)            ====================//

    #ifdef enable_CUDA_CODE
        __global__ void copy_delta_spin_squared_max(long int sites)
        {
            int index = threadIdx.x + blockIdx.x*blockDim.x;
            int xyzi = index;
            if (index < sites)
            {
                int j_S;
                dev_delta_spin_max_reduce[ xyzi ] = 0.0;
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    double delta_S_temp = dev_spin[ xyzi*dim_S + j_S ] - dev_spin_bkp[ xyzi*dim_S + j_S ];
                    
                    dev_delta_spin_abs_reduce[ xyzi*dim_S + j_S ] = fabs(delta_S_temp);
                    dev_delta_spin_squared_reduce[ xyzi*dim_S + j_S ] = (delta_S_temp)*(delta_S_temp);
                    
                    dev_delta_spin_max_reduce[ xyzi ] += dev_delta_spin_squared_reduce[ xyzi*dim_S + j_S ];
                }
            }
            return; 
        }

        __global__ void ensemble_delta_S_squared_max_cuda_reduce(long int max_sites, long int stride_second_site)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            long int xyzi = index;
            if (index < max_sites)
            {
                int j_S;

                for (j_S=0; j_S<dim_S; j_S++)
                {
                    dev_delta_spin_abs_reduce[ dim_S*xyzi + j_S ] += dev_delta_spin_abs_reduce[ dim_S*(xyzi+stride_second_site) + j_S ];
                    dev_delta_spin_squared_reduce[ dim_S*xyzi + j_S ] += dev_delta_spin_squared_reduce[ dim_S*(xyzi+stride_second_site) + j_S ];
                }
                if (dev_delta_spin_max_reduce[ xyzi ] < dev_delta_spin_squared_reduce[ (xyzi+stride_second_site) ])
                {
                    dev_delta_spin_max_reduce[ xyzi ] = dev_delta_spin_squared_reduce[ (xyzi+stride_second_site) ];
                }
            }

            return;
        }

        int ensemble_delta_S_squared_max()
        {
            long int i; 
            int j_S;
            copy_delta_spin_squared_max<<< dim_S*no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites);
            cudaDeviceSynchronize();

            ensemble_delta_S_squared_max_cuda_reduce<<< no_of_sites_remaining_power_2/gpu_threads + 1, gpu_threads >>>(no_of_sites_remaining_power_2, no_of_sites_max_power_2);
            cudaDeviceSynchronize();

            long int no_of_sites_halved = no_of_sites_max_power_2;
            do
            {
                no_of_sites_halved = no_of_sites_halved/2;
                ensemble_delta_S_squared_max_cuda_reduce<<< no_of_sites_halved/gpu_threads + 1, gpu_threads >>>(no_of_sites_halved, no_of_sites_halved);
                
                cudaDeviceSynchronize();
            }
            while (no_of_sites_halved != 1);

            // copy_delta_S_squared_max_ensemble<<< 1, 1 >>>(no_of_sites);
            // cudaDeviceSynchronize();

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(delta_S_squared, dev_delta_spin_squared_reduce, dim_S*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(delta_S_abs, dev_delta_spin_abs_reduce, dim_S*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(&delta_S_max, dev_delta_spin_max_reduce, sizeof(double), cudaMemcpyDeviceToHost);
                #else
                cudaMemcpyFromSymbol(delta_S_squared, "dev_delta_spin_squared_reduce", dim_S*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpyFromSymbol(delta_S_abs, "dev_delta_spin_abs_reduce", dim_S*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpyFromSymbol(&delta_S_max, "dev_delta_spin_max_reduce", sizeof(double), cudaMemcpyDeviceToHost);
                #endif
            #endif

            return 0;
        }

        __global__ void copy_delta_spin_squared(long int sites)
        {
            int index = threadIdx.x + blockIdx.x*blockDim.x;
            int xyzi = index;
            if (index < sites)
            {
                int j_S;
                
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    double delta_S_temp = dev_spin[ xyzi*dim_S + j_S ] - dev_spin_bkp[ xyzi*dim_S + j_S ];
                    
                    dev_delta_spin_abs_reduce[ xyzi*dim_S + j_S ] = fabs(delta_S_temp);
                    dev_delta_spin_squared_reduce[ xyzi*dim_S + j_S ] = (delta_S_temp)*(delta_S_temp);
                    
                }
            }
            return; 
        }

        __global__ void ensemble_delta_S_squared_cuda_reduce(long int max_sites, long int stride_second_site)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            long int xyzi = index;
            if (index < max_sites)
            {
                int j_S;

                for (j_S=0; j_S<dim_S; j_S++)
                {
                    dev_delta_spin_abs_reduce[ dim_S*xyzi + j_S ] += dev_delta_spin_abs_reduce[ dim_S*(xyzi+stride_second_site) + j_S ];
                    dev_delta_spin_squared_reduce[ dim_S*xyzi + j_S ] += dev_delta_spin_squared_reduce[ dim_S*(xyzi+stride_second_site) + j_S ];
                }
                
            }

            return;
        }

        int ensemble_delta_S_squared()
        {
            long int i; 
            int j_S;
            copy_delta_spin_squared<<< dim_S*no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites);
            cudaDeviceSynchronize();

            ensemble_delta_S_squared_cuda_reduce<<< no_of_sites_remaining_power_2/gpu_threads + 1, gpu_threads >>>(no_of_sites_remaining_power_2, no_of_sites_max_power_2);
            cudaDeviceSynchronize();

            long int no_of_sites_halved = no_of_sites_max_power_2;
            do
            {
                no_of_sites_halved = no_of_sites_halved/2;
                ensemble_delta_S_squared_cuda_reduce<<< no_of_sites_halved/gpu_threads + 1, gpu_threads >>>(no_of_sites_halved, no_of_sites_halved);
                
                cudaDeviceSynchronize();
            }
            while (no_of_sites_halved != 1);

            // copy_delta_S_squared_ensemble<<< 1, 1 >>>(no_of_sites);
            // cudaDeviceSynchronize();

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(delta_S_squared, dev_delta_spin_squared_reduce, dim_S*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(delta_S_abs, dev_delta_spin_abs_reduce, dim_S*sizeof(double), cudaMemcpyDeviceToHost);
                #else
                cudaMemcpyFromSymbol(delta_S_squared, "dev_delta_spin_squared_reduce", dim_S*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpyFromSymbol(delta_S_abs, "dev_delta_spin_abs_reduce", dim_S*sizeof(double), cudaMemcpyDeviceToHost);
                #endif
            #endif

            return 0;
        }

        __global__ void copy_delta_spin_max(long int sites)
        {
            int index = threadIdx.x + blockIdx.x*blockDim.x;
            int xyzi = index;
            if (index < sites)
            {
                int j_S;
                dev_delta_spin_max_reduce[ xyzi ] = 0.0;
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    double delta_S_temp = dev_spin[ xyzi*dim_S + j_S ] - dev_spin_bkp[ xyzi*dim_S + j_S ];
                    
                    dev_delta_spin_max_reduce[ xyzi ] += (delta_S_temp)*(delta_S_temp);
                }
            }
            return; 
        }

        __global__ void ensemble_delta_S_max_cuda_reduce(long int max_sites, long int stride_second_site)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            long int xyzi = index;
            if (index < max_sites)
            {

                if (dev_delta_spin_max_reduce[ xyzi ] < dev_delta_spin_squared_reduce[ (xyzi+stride_second_site) ])
                {
                    dev_delta_spin_max_reduce[ xyzi ] = dev_delta_spin_squared_reduce[ (xyzi+stride_second_site) ];
                }
            }

            return;
        }

        int find_delta_S_max()
        {
            long int i; 
            int j_S;
            copy_delta_spin_max<<< dim_S*no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites);
            cudaDeviceSynchronize();

            ensemble_delta_S_max_cuda_reduce<<< no_of_sites_remaining_power_2/gpu_threads + 1, gpu_threads >>>(no_of_sites_remaining_power_2, no_of_sites_max_power_2);
            cudaDeviceSynchronize();

            long int no_of_sites_halved = no_of_sites_max_power_2;
            do
            {
                no_of_sites_halved = no_of_sites_halved/2;
                ensemble_delta_S_max_cuda_reduce<<< no_of_sites_halved/gpu_threads + 1, gpu_threads >>>(no_of_sites_halved, no_of_sites_halved);
                
                cudaDeviceSynchronize();
            }
            while (no_of_sites_halved != 1);

            // copy_delta_S_max_ensemble<<< 1, 1 >>>(no_of_sites);
            // cudaDeviceSynchronize();

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(&delta_S_max, dev_delta_spin_max_reduce, sizeof(double), cudaMemcpyDeviceToHost);
                #else
                cudaMemcpyFromSymbol(&delta_S_max, "dev_delta_spin_max_reduce", sizeof(double), cudaMemcpyDeviceToHost);
                #endif
            #endif

            return 0;
        }
    #else
        int ensemble_delta_S_squared_max()
        {
            long int i; 
            int j_S;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                delta_S_squared[j_S] = 0;
                delta_S_abs[j_S] = 0;
            }
            delta_S_max = 0.0;
            
            #ifdef OLD_COMPILER
            #pragma omp parallel for private(i,j_S) reduction(max:delta_S_max)
            for(i=0; i<no_of_sites; i++)
            {
                double delta_S_sq_temp = 0.0;
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    double delta_S_temp = spin_bkp[dim_S*i + j_S]-spin[dim_S*i + j_S];
                    // #pragma omp critical
                    delta_S_sq_temp += delta_S_temp*delta_S_temp;
                }
                if (delta_S_max < delta_S_sq_temp)
                {
                    delta_S_max = delta_S_sq_temp;
                }
            }
            for (j_S=0; j_S<dim_S; j_S++)
            {
                double delta_S_squared_j_S = 0;
                double delta_S_abs_j_S = 0;
                #pragma omp parallel for private(i,j_S) reduction(+:delta_S_squared_j_S,delta_S_abs_j_S)
                for(i=0; i<no_of_sites; i++)
                {
                    double delta_S_temp = spin_bkp[dim_S*i + j_S]-spin[dim_S*i + j_S];
                    delta_S_squared_j_S += delta_S_temp*delta_S_temp;
                    delta_S_abs_j_S += fabs(delta_S_temp);
                }
                delta_S_squared[j_S] = delta_S_squared_j_S;
                delta_S_abs[j_S] = delta_S_abs_j_S;
            }
            #else
            #pragma omp parallel for private(i,j_S) reduction(+:delta_S_squared[:dim_S],delta_S_abs[:dim_S]) reduction(max:delta_S_max)
            for(i=0; i<no_of_sites; i++)
            {
                double delta_S_sq_temp = 0.0;
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    double delta_S_temp = spin_bkp[dim_S*i + j_S]-spin[dim_S*i + j_S];
                    delta_S_squared[j_S] += delta_S_temp*delta_S_temp;
                    delta_S_abs[j_S] += fabs(delta_S_temp);
                    // #pragma omp critical
                    delta_S_sq_temp += delta_S_temp*delta_S_temp;
                }
                if (delta_S_max < delta_S_sq_temp)
                {
                    delta_S_max = delta_S_sq_temp;
                }
            }
            #endif
            // delta_S_max = sqrt(delta_S_max);
            
            return 0;
        }

        int ensemble_delta_S_squared()
        {
            long int i; 
            int j_S;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                delta_S_squared[j_S] = 0;
                delta_S_abs[j_S] = 0;
            }
            #ifdef OLD_COMPILER
            for (j_S=0; j_S<dim_S; j_S++)
            {
                double delta_S_squared_j_S = 0;
                double delta_S_abs_j_S = 0;
                #pragma omp parallel for private(i,j_S) reduction(+:delta_S_squared_j_S,delta_S_abs_j_S)
                for(i=0; i<no_of_sites; i++)
                {
                    double delta_S_temp = spin_bkp[dim_S*i + j_S]-spin[dim_S*i + j_S];
                    delta_S_squared_j_S += delta_S_temp*delta_S_temp;
                    delta_S_abs_j_S += fabs(delta_S_temp);
                }
                delta_S_squared[j_S] = delta_S_squared_j_S;
                delta_S_abs[j_S] = delta_S_abs_j_S;
            }
            #else
            #pragma omp parallel for private(i,j_S) reduction(+:delta_S_squared[:dim_S],delta_S_abs[:dim_S]) 
            for(i=0; i<no_of_sites; i++)
            {
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    double delta_S_temp = spin_bkp[dim_S*i + j_S]-spin[dim_S*i + j_S];
                    delta_S_squared[j_S] += delta_S_temp*delta_S_temp;
                    delta_S_abs[j_S] += fabs(delta_S_temp);
                }
            }
            #endif
            
            return 0;
        }

        int find_delta_S_max()
        {
            long int i; 
            int j_S;

            delta_S_max = 0.0;
            
            #pragma omp parallel for private(i,j_S) reduction(max:delta_S_max)
            for(i=0; i<no_of_sites; i++)
            {
                // #pragma omp critical
                {
                    double delta_S_sq_temp = 0.0;
                    for (j_S=0; j_S<dim_S; j_S++)
                    {
                        double delta_S_temp = spin_bkp[dim_S*i + j_S]-spin[dim_S*i + j_S];
                        delta_S_sq_temp += delta_S_temp*delta_S_temp;
                    }
                    if (delta_S_max < delta_S_sq_temp)
                    {
                        delta_S_max = delta_S_sq_temp;
                    }
                }
            }
            
            // delta_S_max = sqrt(delta_S_max);
            
            return 0;
        }
    #endif

//====================  Magnetization                 ====================//

    int ensemble_abs_m()
    {
        long int i; 
        int j_S;
        for (j_S=0; j_S<dim_S; j_S=j_S+1)
        {
            abs_m[j_S] = 0;
        }
        
        #ifndef OLD_COMPILER
        #pragma omp parallel for private(j_S) reduction(+:abs_m[:dim_S])
        for(i=0; i<no_of_sites; i=i+1)
        {
            for (j_S=0; j_S<dim_S; j_S=j_S+1)
            {
                abs_m[j_S] += fabs( spin[dim_S*i + j_S]);
            }
        }
        #else
        for (j_S=0; j_S<dim_S; j_S=j_S+1)
        {
            double abs_m_j_S = 0.0;
            #pragma omp parallel for private(i) reduction(+:abs_m_j_S)
            for(i=0; i<no_of_sites; i=i+1)
            {
                abs_m_j_S += fabs( spin[dim_S*i + j_S]);
            }
            abs_m[j_S] = abs_m_j_S;
        }
        #endif
        
        for (j_S=0; j_S<dim_S; j_S=j_S+1)
        {
            abs_m[j_S] = abs_m[j_S] / no_of_sites;
        }
        
        return 0;
    }

    /* int set_sum_of_moment_m_0()
    {
        int j_S;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_sum[j_S] = 0;
            m_avg[j_S] = 1;
        }

        return 0;
    } */

    #ifdef enable_CUDA_CODE
        __global__ void ensemble_m_cuda_reduce(long int max_sites, long int stride_second_site)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            long int xyzi = index;
            if (index < max_sites)
            {
                int j_S;
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    dev_spin_reduce[ dim_S*xyzi + j_S ] += dev_spin_reduce[ dim_S*(xyzi+stride_second_site) + j_S ];
                }
            }
            // do
            // {
            //     stride_second_site /= 2;
            //     ensemble_m_cuda_reduce <<< stride_second_site/dev_gpu_threads + 1, dev_gpu_threads >>>(stride_second_site, stride_second_site);
            // }
            // while (stride_second_site != 1);

            return;
        }

        __global__ void copy_m_ensemble(long int sites)
        {
            int index_j_S = threadIdx.x + blockIdx.x*blockDim.x;
            if (index_j_S < dim_S)
            {
                dev_m[index_j_S] = dev_spin_reduce[index_j_S] / sites;
            }
            return; 
        }
        
        __global__ void copy_spin(long int sites)
        {
            int index = threadIdx.x + blockIdx.x*blockDim.x;
            if (index < sites*dim_S)
            {
                dev_spin_reduce[index] = dev_spin[index];
            }
            return; 
        }

        int ensemble_m()
        {
            long int i; 
            int j_S;
            
            copy_spin<<< dim_S*no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites);
            // copy_spin<<< 1, dim_S*no_of_sites >>>(no_of_sites);

            // copy_in_device_var<<< dim_S*no_of_sites/gpu_threads + 1, gpu_threads >>>(dev_spin_reduce, dev_spin_temp, dim_S*no_of_sites);
            cudaDeviceSynchronize();
            ensemble_m_cuda_reduce<<< no_of_sites_remaining_power_2/gpu_threads + 1, gpu_threads >>>(no_of_sites_remaining_power_2, no_of_sites_max_power_2);
            // ensemble_m_cuda_reduce<<< 1, no_of_sites_remaining_power_2 >>>(no_of_sites_remaining_power_2, no_of_sites_max_power_2);
            cudaDeviceSynchronize();
            
            long int no_of_sites_halved = no_of_sites_max_power_2;
            do
            {
                no_of_sites_halved = no_of_sites_halved/2;
                ensemble_m_cuda_reduce<<< no_of_sites_halved/gpu_threads + 1, gpu_threads >>>(no_of_sites_halved, no_of_sites_halved);
                // ensemble_m_cuda_reduce<<< 1, no_of_sites_halved >>>(no_of_sites_halved, no_of_sites_halved);
                cudaDeviceSynchronize();
            }
            while (no_of_sites_halved != 1);

            copy_m_ensemble<<< 1, dim_S >>>(no_of_sites);
            cudaDeviceSynchronize();

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(m, dev_m, dim_S*sizeof(double), cudaMemcpyDeviceToHost);
                #else
                cudaMemcpyFromSymbol(m, "dev_m", dim_S*sizeof(double), cudaMemcpyDeviceToHost);
                #endif
            #endif
                    
            return 0;
        }
    #else
        int ensemble_m()
        {
            long int i; 
            int j_S;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                m[j_S] = 0;
            }
            
            #ifndef OLD_COMPILER
            #pragma omp parallel for private(i,j_S) reduction(+:m[:dim_S])
            for(i=0; i<no_of_sites; i++)
            {
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    m[j_S] += spin[dim_S*i + j_S];
                }
            }
            #else
            for (j_S=0; j_S<dim_S; j_S++)
            {
                double m_j_S = 0.0;
                #pragma omp parallel for private(i) reduction(+:m_j_S)
                for(i=0; i<no_of_sites; i++)
                {
                    m_j_S += spin[dim_S*i + j_S];
                }
                m[j_S] = m_j_S;
            }
            #endif

            for (j_S=0; j_S<dim_S; j_S++)
            {
                m[j_S] = m[j_S] / no_of_sites;
            }
            
            return 0;
        }
    #endif
    
    /* int sum_of_moment_m()
    {
        int j_S;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_sum[j_S] += m[j_S];
        }
        
        return 0;
    } */

    /* int average_of_moment_m(double MCS_counter)
    {
        int j_S;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_avg[j_S] = m_sum[j_S] / MCS_counter;
        }
        
        return 0;
    } */

    /* int set_sum_of_moment_m_abs_0()
    {
        m_abs_sum = 0;
        m_abs_avg = 1;

        return 0;
    } */

    /* int sum_of_moment_m_abs()
    {
        int j_S, j_SS, j_L;
        double m_2_persite = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_2_persite += m[j_S] * m[j_S];
        }

        m_abs_sum += sqrt(m_2_persite);

        return 0;
    } */

    /* int average_of_moment_m_abs(double MCS_counter)
    {
        m_abs_avg = m_abs_sum / MCS_counter;

        return 0;
    } */

    /* int set_sum_of_moment_m_2_0()
    {
        m_2_sum = 0;
        m_2_avg = 1;

        return 0;
    } */

    /* int sum_of_moment_m_2()
    {
        int j_S, j_SS, j_L;
        double m_2_persite = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_2_persite += m[j_S] * m[j_S];
        }

        m_2_sum += m_2_persite;

        return 0;
    } */

    /* int average_of_moment_m_2(double MCS_counter)
    {
        m_2_avg = m_2_sum / MCS_counter;

        return 0;
    } */

    /* int set_sum_of_moment_m_4_0()
    {
        m_4_sum = 0;
        m_4_avg = 1;

        return 0;
    } */

    /* int sum_of_moment_m_4()
    {
        int j_S, j_SS, j_L;
        double m_2_persite = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_2_persite += m[j_S] * m[j_S];
        }

        m_4_sum += m_2_persite * m_2_persite;

        return 0;
    } */

    /* int average_of_moment_m_4(double MCS_counter)
    {
        m_4_avg = m_4_sum / MCS_counter;

        return 0;
    } */

    /* int set_sum_of_moment_m_ab_0()
    {
        int j_S, j_SS;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                m_ab_sum[j_S*dim_S + j_SS] = 0;
                m_ab_avg[j_S*dim_S + j_SS] = 1;
            }
        }

        return 0;
    } */

    /* int sum_of_moment_m_ab()
    {
        int j_S, j_SS;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {    
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                m_ab_sum[j_S*dim_S + j_SS] += m[j_S] * m[j_SS];            
            }
        }

        return 0;
    } */

    /* int average_of_moment_m_ab(double MCS_counter)
    {
        int j_S, j_SS;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                m_ab_avg[j_S*dim_S + j_SS] = m_ab_sum[j_S*dim_S + j_SS] / MCS_counter;
            }
        }
        
        return 0;
    } */

    /* int set_sum_of_moment_m_higher_0()
    {
        int j_S, j_SS;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                m_ab_sum[j_S*dim_S + j_SS] = 0;
                m_ab_avg[j_S*dim_S + j_SS] = 1;
            }
        }

        m_2_sum = 0;
        m_2_avg = 1;
        m_4_sum = 0;
        m_4_avg = 1;

        return 0;
    } */

    /* int sum_of_moment_m_higher()
    {
        int j_S, j_SS, j_L;
        double m_2_persite = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                m_ab_sum[j_S*dim_S + j_SS] += m[j_S] * m[j_SS];            
            }
            m_2_persite += m[j_S] * m[j_S];
        }

        m_2_sum += m_2_persite;
        m_4_sum += m_2_persite * m_2_persite;

        return 0;
    } */

    /* int average_of_moment_m_higher(double MCS_counter)
    {
        int j_S, j_SS, j_L;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                m_ab_avg[j_S*dim_S + j_SS] = m_ab_sum[j_S*dim_S + j_SS] / MCS_counter;
            }
        }
        m_2_avg = m_2_sum / MCS_counter;
        m_4_avg = m_4_sum / MCS_counter;
        
        return 0;
    } */

    /* int set_sum_of_moment_m_all_0()
    {
        int j_S, j_SS;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                m_ab_sum[j_S*dim_S + j_SS] = 0;
                m_ab_avg[j_S*dim_S + j_SS] = 1;
            }
        }
        m_abs_sum = 0;
        m_abs_avg = 1;
        m_2_sum = 0;
        m_2_avg = 1;
        m_4_sum = 0;
        m_4_avg = 1;

        return 0;
    } */

    /* int sum_of_moment_m_all()
    {
        int j_S, j_SS, j_L;
        double m_2_persite = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                m_ab_sum[j_S*dim_S + j_SS] += m[j_S] * m[j_SS];            
            }
            m_2_persite += m[j_S] * m[j_S];
        }

        m_abs_sum += sqrt(m_2_persite);
        m_2_sum += m_2_persite;
        m_4_sum += m_2_persite * m_2_persite;

        return 0;
    } */

    /* int average_of_moment_m_all(double MCS_counter)
    {
        int j_S, j_SS, j_L;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                m_ab_avg[j_S*dim_S + j_SS] = m_ab_sum[j_S*dim_S + j_SS] / MCS_counter;
            }
        }
        m_abs_avg = m_abs_sum / MCS_counter;
        m_2_avg = m_2_sum / MCS_counter;
        m_4_avg = m_4_sum / MCS_counter;
        
        return 0;
    } */

    /* int set_sum_of_moment_m_vec_0()
    {
        int j_S, j_SS;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_abs_vec_sum[j_S] = 0;
            m_abs_vec_avg[j_S] = 1;
            m_2_vec_sum[j_S] = 0;
            m_2_vec_avg[j_S] = 1;
            m_4_vec_sum[j_S] = 0;
            m_4_vec_avg[j_S] = 1;
        }

        return 0;
    } */

    /* int sum_of_moment_m_vec()
    {
        int j_S, j_SS, j_L;
        double m_2_vec_persite = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_2_vec_persite = m[j_S] * m[j_S];
            m_abs_vec_sum[j_S] += fabs(m[j_S]);
            m_2_vec_sum[j_S] += m_2_vec_persite;
            m_4_vec_sum[j_S] += m_2_vec_persite * m_2_vec_persite;
        }

        return 0;
    } */

    /* int average_of_moment_m_vec(double MCS_counter)
    {
        int j_S, j_SS, j_L;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_abs_vec_avg[j_S] = m_abs_vec_sum[j_S] / MCS_counter;
            m_2_vec_avg[j_S] = m_2_vec_sum[j_S] / MCS_counter;
            m_4_vec_avg[j_S] = m_4_vec_sum[j_S] / MCS_counter;
        }
        
        return 0;
    } */


//====================  Binder Parameter              ====================//

    int calculate_B() // (double MCS_counter)
    {
        B = (1.0 / 2.0) * ( 3.0 - ( m_4_avg / (m_2_avg * m_2_avg) ) );
        
        return 0;
    }

    int calculate_B_a() // (double MCS_counter)
    {
        int j_S;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            B_a[j_S] = (1.0 / 2.0) * ( 3.0 - ( m_4_vec_avg[j_S] / (m_2_vec_avg[j_S] * m_2_vec_avg[j_S]) ) );
        }
        
        return 0;
    }

//====================  Susceptibity                  ====================//

    int calculate_X() // (double MCS_counter)
    {
        X = (m_2_avg - (m_abs_avg * m_abs_avg)) / T;
        
        return 0;
    }

//====================  Susceptibility tensor         ====================//

    int calculate_X_ab() // (double MCS_counter)
    {
        int j_S, j_SS, j_L;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                X_ab[j_S*dim_S + j_SS] = (m_ab_avg[j_S*dim_S + j_SS] - m_avg[j_S] * m_avg[j_SS]) / T;
            }
        }
        
        return 0;
    }

//====================  Energy                        ====================//

    /* int set_sum_of_moment_E_0()
    {
        E_sum = 0;
        E_avg = 1;

        return 0;
    } */

    int ensemble_E()
    {
        long int i;
        int j_L, j_S;
        E = 0;
        
        #pragma omp parallel for private(j_L, j_S) reduction(+:E)
        for(i=0; i<no_of_sites; i++)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    #ifdef RANDOM_BOND
                    E += - (J[j_L] + J_random[2*dim_L*i + 2*j_L]) * spin[dim_S * N_N_I[i*2*dim_L + 2*j_L] + j_S] * (spin[dim_S*i + j_S]);
                    #else
                    E += - (J[j_L]) * spin[dim_S * N_N_I[i*2*dim_L + 2*j_L] + j_S] * (spin[dim_S*i + j_S]);
                    #endif
                }
                #ifdef RANDOM_FIELD
                E += - (h[j_S] + h_random[dim_S*i + j_S]) * (spin[dim_S*i + j_S]);
                #else
                E += - (h[j_S]) * (spin[dim_S*i + j_S]);
                #endif
            }
        }
        // for (j_S=0; j_S<dim_S; j_S++)
        // {
            // #pragma omp parallel for private(j_L) reduction(+:E)
            // for(i=0; i<no_of_sites; i++)
            // {
                // for (j_L=0; j_L<dim_L; j_L++)
                // {
                    // E += - (J[j_L] + J_random[2*dim_L*i + 2*j_L])  * spin[dim_S * N_N_I[i*2*dim_L + 2*j_L] + j_S] * (spin[dim_S*i + j_S]);
                // }
                // E += - (h[j_S] + h_random[dim_S*i + j_S]) * (spin[dim_S*i + j_S]);
            // }
        // }
        E = E / no_of_sites;
        return 0;
    }

    /* int sum_of_moment_E()
    {
        E_sum += E;
        
        return 0;
    } */

    /* int average_of_moment_E(double MCS_counter)
    {
        E_avg = E_sum / MCS_counter;
        
        return 0;
    } */

    /* int set_sum_of_moment_E_2_0()
    {
        E_2_sum = 0;
        E_2_avg = 1;

        return 0;
    } */

    /* int sum_of_moment_E_2()
    {
        E_2_sum += E * E;
        
        return 0;
    } */

    /* int average_of_moment_E_2(double MCS_counter)
    {
        E_2_avg = E_2_sum / MCS_counter;
        
        return 0;
    } */

//====================  Specific Heat                 ====================//

    int calculate_Cv() // (double MCS_counter)
    {

        Cv = (E_2_avg - (E_avg * E_avg)) / (T * T);
        
        return 0;
    }

//====================  Helicity                      ====================//

    /* int set_sum_of_moment_Y_0()
    {
        int j_S, j_SS, j_L;
        
        for (j_L=0; j_L<dim_L; j_L++)
        {
            Y_1_sum[j_L] = 0;
            Y_1_avg[j_L] = 1;
            Y_2_sum[j_L] = 0;
            Y_2_avg[j_L] = 1;
        }
        // for (j_L_j_S_j_SS=0; j_L_j_S_j_SS < dim_S*dim_S*dim_L; j_L_j_S_j_SS++)
        // {
        //     Y_1_sum[j_L_j_S_j_SS] = 0;
        //     Y_1_avg[j_L_j_S_j_SS] = 1;
        //     Y_2_sum[j_L_j_S_j_SS] = 0;
        //     Y_2_avg[j_L_j_S_j_SS] = 1;
        // }

        return 0;
    } */

    int ensemble_Y()
    {
        // http://www.pas.rochester.edu/~stte/papers/FFXY4.pdf
        // http://samos.martech.fsu.edu/papers/pdf/PRB49-1.pdf
        long int i;
        int j_L, j_S, j_SS, j_L_j_S_j_SS;

        // for (j_L_j_S_j_SS=0; j_L_j_S_j_SS < dim_S*dim_S*dim_L; j_L_j_S_j_SS++)
        // {
        //     Y_1[j_L_j_S_j_SS] = 0;
        //     Y_2[j_L_j_S_j_SS] = 0;
        // }
        for (j_L=0; j_L < dim_L; j_L++)
        {
            Y_1[j_L] = 0;
            Y_2[j_L] = 0;
        }

        #ifndef OLD_COMPILER
        // #pragma omp parallel for private(j_L, j_S, j_SS) reduction(+:Y_1[:dim_S*dim_S*dim_L], Y_2[:dim_S*dim_S*dim_L])
        #pragma omp parallel for private(j_L, j_S, j_SS) reduction(+:Y_1[:dim_L], Y_2[:dim_L])
        for(i=0; i<no_of_sites; i++)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        #ifdef RANDOM_BOND
                        if (j_S==j_SS)
                        {
                            Y_1[j_L] += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        }
                        else
                        {
                            Y_2[j_L] += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S] *(double)(j_SS-j_S)/fabs(j_SS-j_S)*(double)(((j_SS+j_S)%2)*2-1);
                        }
                        #else
                        if (j_S==j_SS)
                        {
                            Y_1[j_L] += ( J[j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        }
                        else
                        {
                            Y_2[j_L] += ( J[j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S] *(double)(j_SS-j_S)/fabs(j_SS-j_S)*(double)(((j_SS+j_S)%2)*2-1);
                        }
                        #endif
                        // #ifdef RANDOM_BOND
                        // Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        // Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        // Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += - ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        // Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        // #else
                        // Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        // Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        // Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += - ( J[j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        // Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        // #endif
                    }
                }
            }
        }
        #else
        for (j_L=0; j_L<dim_L; j_L++)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    double Y_1_j_L = 0.0;
                    double Y_2_j_L = 0.0;
                    // double Y_1_j_L_j_S_j_SS = 0.0;
                    // double Y_2_j_L_j_S_j_SS = 0.0;
                    // #pragma omp parallel for private(i) reduction(+:Y_1_j_L_j_S_j_SS, Y_2_j_L_j_S_j_SS)
                    #pragma omp parallel for private(i) reduction(+:Y_1_j_L, Y_2_j_L)
                    for(i=0; i<no_of_sites; i++)
                    {
                        #ifdef RANDOM_BOND
                        if (j_S==j_SS)
                        {
                            Y_1_j_L += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        }
                        else
                        {
                            Y_2_j_L += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S] *(double)(j_SS-j_S)/fabs(j_SS-j_S)*(double)(((j_SS+j_S)%2)*2-1);
                        }
                        #else
                        if (j_S==j_SS)
                        {
                            Y_1_j_L += ( J[j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        }
                        else
                        {
                            Y_2_j_L += ( J[j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S] *(double)(j_SS-j_S)/fabs(j_SS-j_S)*(double)(((j_SS+j_S)%2)*2-1);
                        }
                        #endif
                        // #ifdef RANDOM_BOND
                        // Y_1_j_L_j_S_j_SS += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        // Y_1_j_L_j_S_j_SS += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        // Y_2_j_L_j_S_j_SS += - ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        // Y_2_j_L_j_S_j_SS += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        // #else
                        // Y_1_j_L_j_S_j_SS += ( J[j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        // Y_1_j_L_j_S_j_SS += ( J[j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        // Y_2_j_L_j_S_j_SS += - ( J[j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        // Y_2_j_L_j_S_j_SS += ( J[j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        // #endif
                    }
                    Y_1[j_L] += Y_1_j_L;
                    Y_2[j_L] += Y_2_j_L;
                    // Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_1_j_L_j_S_j_SS;
                    // Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_2_j_L_j_S_j_SS;
                }
            }
        }
        #endif

        for (j_L=0; j_L < dim_L; j_L++)
        {
            Y_1[j_L] = Y_1[j_L] / no_of_sites;
            Y_2[j_L] = (Y_2[j_L] * Y_2[j_L]) / no_of_sites;
        }
        // for (j_L_j_S_j_SS=0; j_L_j_S_j_SS < dim_S*dim_S*dim_L; j_L_j_S_j_SS++)
        // {
        //     Y_1[j_L_j_S_j_SS] = Y_1[j_L_j_S_j_SS] / no_of_sites;
        //     Y_2[j_L_j_S_j_SS] = (Y_2[j_L_j_S_j_SS] * Y_2[j_L_j_S_j_SS]) / no_of_sites;
        // }
        
        return 0;
    }

    /* int sum_of_moment_Y()
    {
        int j_S, j_SS, j_L;
        
        for (j_L=0; j_L<dim_L; j_L++)
        {
            Y_1_sum[j_L] += Y_1[j_L];
            Y_2_sum[j_L] += Y_2[j_L];
        }
        // for (j_L_j_S_j_SS=0; j_L_j_S_j_SS < dim_S*dim_S*dim_L; j_L_j_S_j_SS++)
        // {
        //     Y_1_sum[j_L_j_S_j_SS] += Y_1[j_L_j_S_j_SS];
        //     Y_2_sum[j_L_j_S_j_SS] += Y_2[j_L_j_S_j_SS];
        // }
        
        return 0;
    } */

    /* int average_of_moment_Y(double MCS_counter)
    {
        int j_S, j_SS, j_L;
        
        for (j_L=0; j_L<dim_L; j_L++)
        {
            Y_1_avg[j_L] = Y_1_sum[j_L] / MCS_counter;
            Y_2_avg[j_L] = Y_2_sum[j_L] / MCS_counter;
        }

        // for (j_L_j_S_j_SS=0; j_L_j_S_j_SS < dim_S*dim_S*dim_L; j_L_j_S_j_SS++)
        // {
        //     Y_1_avg[j_L_j_S_j_SS] = Y_1_sum[j_L_j_S_j_SS] / MCS_counter;
        //     Y_2_avg[j_L_j_S_j_SS] = Y_2_sum[j_L_j_S_j_SS] / MCS_counter;
        // }
        
        return 0;
    } */

//====================  Helicity Modulus              ====================//

    int calculate_Y_x_mu() // (double MCS_counter)
    {

        int j_S, j_SS, j_L;
        
        for (j_L=0; j_L<dim_L; j_L++)
        {
            Y_x_mu[j_L] = Y_1_avg[j_L] - Y_2_avg[j_L] / T;
        }

        // for (j_L_j_S_j_SS=0; j_L_j_S_j_SS < dim_S*dim_S*dim_L; j_L_j_S_j_SS++)
        // {
        //     Y_ab_mu[j_L_j_S_j_SS] = Y_1_avg[j_L_j_S_j_SS] - Y_2_avg[j_L_j_S_j_SS] / T;
        // }
        
        return 0;
    }

//====================  All moments                   ====================//

    int set_sum_of_moment_all_0()
    {
        int j_L, j_S, j_SS;
        if ( output_m_abs_avg==1 || output_X==1 )
        {
            m_abs_sum = 0;
            m_abs_avg = 0;
        }
        if ( output_m_2_avg==1 || output_B==1 || output_X==1 )
        {
            m_2_sum = 0;
            m_2_avg = 0;
        }
        if ( output_m_4_avg==1 || output_B==1 )
        {
            m_4_sum = 0;
            m_4_avg = 0;
        }
        
        if ( output_E_avg==1 || output_Cv==1 )
        {
            E_sum = 0;
            E_avg = 0;
        }
        if ( output_E_2_avg==1 || output_Cv==1 )
        {
            E_2_sum = 0;
            E_2_avg = 0;
        }

        for (j_S=0; j_S<dim_S; j_S++)
        {
            if ( output_m_avg==1 || output_X_ab==1 )
            {
                m_sum[j_S] = 0;
                m_avg[j_S] = 0;
            }
            if ( output_m_abs_vec_avg==1 )
            {
                m_abs_vec_sum[j_S] = 0;
                m_abs_vec_avg[j_S] = 0;
            }
            if ( output_m_2_vec_avg==1 || output_B_a==1 )
            {
                m_2_vec_sum[j_S] = 0;
                m_2_vec_avg[j_S] = 0;
            }
            if ( output_m_4_vec_avg==1 || output_B_a==1 )
            {
                m_4_vec_sum[j_S] = 0;
                m_4_vec_avg[j_S] = 0;
            }

            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                if ( output_m_ab_avg==1 || output_X_ab==1 )
                {
                    m_ab_sum[j_S*dim_S + j_SS] = 0;
                    m_ab_avg[j_S*dim_S + j_SS] = 0;
                }
                // for (j_L=0; j_L<dim_L; j_L++)
                // {
                //     Y_1_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = 0;
                //     Y_1_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = 1;
                //     Y_2_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = 0;
                //     Y_2_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = 1;
                // }
            }
        }
        for (j_L=0; j_L<dim_L; j_L++)
        {
            if ( output_Y_1_avg==1 || output_Y_x_mu==1)
            {
                Y_1_sum[j_L] = 0;
                Y_1_avg[j_L] = 0;
            }
            if ( output_Y_2_avg==1 || output_Y_x_mu==1)
            {
                Y_2_sum[j_L] = 0;
                Y_2_avg[j_L] = 0;
            }
        }

        return 0;
    }

    int ensemble_all()
    {
        if ( 
            output_m == 1 ||
            output_m_avg == 1 ||
            output_m_abs_avg == 1 ||
            output_m_2_avg == 1 ||
            output_m_4_avg == 1 ||
            output_m_abs_vec_avg == 1 ||
            output_m_2_vec_avg == 1 ||
            output_m_4_vec_avg == 1 ||
            output_m_ab_avg == 1 ||
            output_X == 1 ||
            output_X_ab == 1 ||
            output_B == 1 ||
            output_B_a == 1 
        ){
            ensemble_m();
        }
        if (
            output_E == 1 ||
            output_E_avg == 1 ||
            output_E_2_avg == 1 ||
            output_Cv == 1 
        ){
            ensemble_E();
        }
        if (
            output_Y_1 == 1 ||
            output_Y_2 == 1 ||
            output_Y_1_avg == 1 ||
            output_Y_2_avg == 1 ||
            output_Y_x_mu == 1 
        ){
            ensemble_Y();
        }

        return 0;
    }

    int sum_of_moment_all()
    {
        int j_L, j_S, j_SS;
        
        double m_2_persite = 0;
        double m_2_vec_persite = 0;

        if ( output_E_avg==1 || output_Cv==1 )
        {
            E_sum += E;
        }
        if ( output_E_2_avg==1 || output_Cv==1 )
        {
            E_2_sum += E * E;
        }

        for (j_S=0; j_S<dim_S; j_S++)
        {
            if ( output_m_avg==1 || output_X_ab==1 )
            {
                m_sum[j_S] += m[j_S];
            }
            if ( output_m_abs_vec_avg==1 )
            {
                m_abs_vec_sum[j_S] += fabs(m[j_S]);
            }
            if ( output_m_abs_avg==1 || output_m_2_avg==1 || output_m_4_avg==1 || output_X==1 || output_B==1 || output_m_2_vec_avg==1 || output_m_4_vec_avg==1 || output_B_a==1 )
            {
                m_2_vec_persite = m[j_S] * m[j_S];
            }
            if ( output_m_abs_avg==1 || output_m_2_avg==1 || output_m_4_avg==1 || output_X==1 || output_B==1 )
            {
                m_2_persite += m_2_vec_persite;
            }
            if ( output_m_2_vec_avg==1 || output_B_a==1 )
            {
                m_2_vec_sum[j_S] += m_2_vec_persite;
            }
            if ( output_m_4_vec_avg==1 || output_B_a==1 )
            {
                m_4_vec_sum[j_S] += m_2_vec_persite * m_2_vec_persite;
            }

            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                if ( output_m_ab_avg==1 || output_X_ab==1 )
                {
                    m_ab_sum[j_S*dim_S + j_SS] += m[j_S] * m[j_SS];
                }
                // for (j_L=0; j_L<dim_L; j_L++)
                // {
                //     Y_1_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS];
                //     Y_2_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS];
                // }
            }
        }
        for (j_L=0; j_L<dim_L; j_L++)
        {
            if ( output_Y_1_avg==1 || output_Y_x_mu==1)
            {
                Y_1_sum[j_L] += Y_1[j_L];
            }
            if ( output_Y_2_avg==1 || output_Y_x_mu==1)
            {
                Y_2_sum[j_L] += Y_2[j_L];
            }
        }

        if ( output_m_abs_avg==1 || output_X==1 )
        {
            m_abs_sum += sqrt(m_2_persite);
        }
        if ( output_m_2_avg==1 || output_B==1 || output_X==1 )
        {
            m_2_sum += m_2_persite;
        }
        if ( output_m_4_avg==1 || output_B==1 )
        {
            m_4_sum += m_2_persite * m_2_persite;
        }

        return 0;
    }

    int average_of_moment_all(double MCS_counter)
    {
        int j_L, j_S, j_SS;

        if ( output_E_avg==1 || output_Cv==1 )
        {
            E_avg = E_sum / MCS_counter;
        }
        if ( output_E_2_avg==1 || output_Cv==1 )
        {
            E_2_avg = E_2_sum / MCS_counter;
        }

        for (j_S=0; j_S<dim_S; j_S++)
        {
            if ( output_m_avg==1 || output_X_ab==1 )
            {
                m_avg[j_S] = m_sum[j_S] / MCS_counter;
            }
            if ( output_m_abs_vec_avg==1 )
            {
                m_abs_vec_sum[j_S] = m_abs_vec_sum[j_S] / MCS_counter;
            }
            if ( output_m_2_vec_avg==1 || output_B_a==1 )
            {
                m_2_vec_avg[j_S] = m_2_vec_sum[j_S] / MCS_counter;
            }
            if ( output_m_4_vec_avg==1 || output_B_a==1 )
            {
                m_4_vec_avg[j_S] = m_4_vec_sum[j_S] / MCS_counter;
            }

            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                if ( output_m_ab_avg==1 || output_X_ab==1 )
                {
                    m_ab_avg[j_S*dim_S + j_SS] = m_ab_sum[j_S*dim_S + j_SS] / MCS_counter;
                }
                // for (j_L=0; j_L<dim_L; j_L++)
                // {
                //     Y_1_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_1_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] / MCS_counter;
                //     Y_2_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_2_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] / MCS_counter;
                // }
            }
        }
        for (j_L=0; j_L<dim_L; j_L++)
        {
            if ( output_Y_1_avg==1 || output_Y_x_mu==1)
            {
                Y_1_avg[j_L] = Y_1_sum[j_L] / MCS_counter;
            }
            if ( output_Y_2_avg==1 || output_Y_x_mu==1)
            {
                Y_2_avg[j_L] = Y_2_sum[j_L] / MCS_counter;
            }
        }

        if ( output_m_abs_avg==1 || output_X==1 )
        {
            m_abs_avg = m_abs_sum / MCS_counter;
        }
        if ( output_m_2_avg==1 || output_B==1 || output_X==1 )
        {
            m_2_avg = m_2_sum / MCS_counter;
        }
        if ( output_m_4_avg==1 || output_B==1 )
        {
            m_4_avg = m_4_sum / MCS_counter;
        }
        
        if (output_B == 1)
        {
            calculate_B();
        }
        
        if (output_B_a == 1)
        {
            calculate_B_a();
        }
        
        if (output_X == 1)
        {
            calculate_X();
        }
        
        if (output_X_ab == 1)
        {
            calculate_X_ab();
        }
        
        if (output_Cv == 1)
        {
            calculate_Cv();
        }
        
        if (output_Y_x_mu == 1)
        {
            calculate_Y_x_mu();
        }

        return 0;
    }

//====================  MonteCarlo-tools              ====================//

    #ifdef enable_CUDA_CODE
        // __global__ void update_spin_all(long int sites, double* spin_local, double* cutoff_bool)
        __global__ void update_spin_all(long int sites, int* cutoff_bool)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            // long int stride = blockDim.x*gridDim.x;
            long int xyzi = index;
            
            if (index < sites)
            {
                int j_S;
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    // double diff_spin = fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]);

                    // if ( diff_spin > dev_CUTOFF_SPIN[0] )
                    // if ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] )
                    // {
                    //     atomicOr(cutoff_bool, 1);
                    //     // cutoff_bool[0] = 1;
                    // }
                    int temp = cutoff_bool[0] || atomicOr(cutoff_bool, (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0]) );
                    // atomicAdd(&cutoff_bool[0], (double) (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] ));
                    // cutoff_bool[0] += (double) ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] );

                    dev_spin[dim_S*xyzi + j_S] = dev_spin_temp[dim_S*xyzi + j_S];
                }
            }

            // return 0;
        }

        __global__ void update_spin_all_cutoff_check(long int sites, int* cutoff_bool)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            // long int stride = blockDim.x*gridDim.x;
            long int xyzi = index;

            if (index < sites)
            {
                int j_S;
                double delta_S = 0.0;
                
                
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    delta_S += (dev_spin_temp[xyzi*dim_S+j_S] - dev_spin_bkp[xyzi*dim_S+j_S])*(dev_spin_temp[xyzi*dim_S+j_S] - dev_spin_bkp[xyzi*dim_S+j_S]);
                    // double diff_spin = fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]);

                    // if ( diff_spin > dev_CUTOFF_SPIN[0] )
                    // if ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] )
                    // {
                    //     atomicOr(cutoff_bool, 1);
                    //     // cutoff_bool[0] = 1;
                    // }
                    int temp = cutoff_bool[0] || atomicOr(cutoff_bool, (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0]) );
                    // atomicAdd(&cutoff_bool[0], (double) (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] ));
                    // cutoff_bool[0] += (double) ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] );
                    
                    dev_spin[xyzi*dim_S+j_S] = dev_spin_temp[xyzi*dim_S+j_S];
                }

                // if ( delta_S > dev_CUTOFF_S_SQ[0] )
                // {
                //     atomicOr(cutoff_bool+1, 1);
                //     // cutoff_bool[1] = 1;
                // }
                int temp = cutoff_bool[1] || atomicOr(cutoff_bool+1, (delta_S > dev_CUTOFF_S_SQ[0]) );
                // atomicAdd(&cutoff_bool[1], (double) ( delta_S > dev_CUTOFF_S_SQ[0] ));
                // cutoff_bool[1] += (double) ( delta_S > dev_CUTOFF_S_SQ[0] );
            }

        }

        __global__ void update_spin_all_cutoff_continue(long int sites, int* cutoff_bool)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            // long int stride = blockDim.x*gridDim.x;
            long int xyzi = index;

            if (index < sites)
            {
                int j_S;

                for (j_S=0; j_S<dim_S; j_S++)
                {
                    // double diff_spin = fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]);

                    // if ( diff_spin > dev_CUTOFF_SPIN[0] )
                    // if ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] )
                    // {
                    //     atomicOr(cutoff_bool, 1);
                    //     // cutoff_bool[0] = 1;
                    // }
                    int temp = cutoff_bool[0] || atomicOr(cutoff_bool, (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0]) );
                    // atomicAdd(&cutoff_bool[0], (double) (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] ));
                    // cutoff_bool[0] += (double) ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] );
                    
                    dev_spin[xyzi*dim_S+j_S] = dev_spin_temp[xyzi*dim_S+j_S];
                }
                
            }

        }
    #else
        int update_spin_all(double* __restrict__ spin_local)
        {
            int j_S;
            long int xyzi;
            #pragma omp parallel for private(j_S)
            for (xyzi=0; xyzi<no_of_sites; xyzi++)
            {
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    spin[dim_S*xyzi + j_S] = spin_local[dim_S*xyzi + j_S];
                }
            }
            return 0;
        }
    #endif
    
    int update_spin_single(long int xyzi, double* __restrict__ spin_local)
    {
        int j_S;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            spin[dim_S*xyzi + j_S] = spin_local[j_S];
        }

        return 0;
    }

    double Energy_minimum(long int xyzi, double* __restrict__ spin_local, double* __restrict__ field_local)
    {
        int j_S, j_L, k_L;
        double Energy_min = 0.0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            field_local[j_S] = 0.0;
            for (j_L=0; j_L<dim_L; j_L++)
            {
                for (k_L=0; k_L<2; k_L++)
                {
                    #ifdef RANDOM_BOND
                    field_local[j_S] = field_local[j_S] - (J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                    #else
                    field_local[j_S] = field_local[j_S] - J[j_L] * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                    #endif
                }
            }
            #ifdef RANDOM_FIELD
            field_local[j_S] = field_local[j_S] - (h[j_S] + h_random[dim_S*xyzi + j_S]);
            #else
            field_local[j_S] = field_local[j_S] - h[j_S];
            #endif

            Energy_min = Energy_min + field_local[j_S] * field_local[j_S];
        }
        if(Energy_min==0)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin_local[j_S] = spin[dim_S*xyzi + j_S];
            }
        }
        else
        {
            Energy_min = -sqrt(Energy_min);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin_local[j_S] = field_local[j_S] / Energy_min;
            }
        }

        return Energy_min;
    }

    double Energy_old(long int xyzi, double* __restrict__ spin_local, double* __restrict__ field_local)
    {
        int j_S, j_L, k_L;
        double Energy_ol = 0.0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            field_local[j_S] = 0.0;
            for (j_L=0; j_L<dim_L; j_L++)
            {
                for (k_L=0; k_L<2; k_L++)
                {
                    // field_site[dim_S*xyzi + j_S] = field_site[dim_S*xyzi + j_S] - (J[j_L] ) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                    #ifdef RANDOM_BOND
                    field_local[j_S] = field_local[j_S] - (J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                    #else
                    field_local[j_S] = field_local[j_S] - J[j_L] * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                    #endif
                }
            }
            // field_site[dim_S*xyzi + j_S] = field_site[dim_S*xyzi + j_S] - (h[j_S]);
            #ifdef RANDOM_FIELD
            field_local[j_S] = field_local[j_S] - (h[j_S] + h_random[dim_S*xyzi + j_S]);
            #else
            field_local[j_S] = field_local[j_S] - h[j_S];
            #endif
            // field_site[dim_S*xyzi + j_S] = field_local[j_S];
            // spin_old[dim_S*xyzi + j_S] = spin[dim_S*xyzi + j_S];
            Energy_ol = Energy_ol + field_local[j_S] * spin[dim_S*xyzi + j_S];
        }
        
        return Energy_ol;
    }

    double Energy_new(long int xyzi, double* __restrict__ spin_local, double* __restrict__ field_local)
    {
        int j_S;
        double Energy_nu=0.0, s_mod=0.0;
        double limit = 0.01 * dim_S;

        if (dim_S == 1)
        {
            spin_local[0] = -spin[xyzi*dim_S+0];
            
            Energy_nu = Energy_nu + field_local[0] * spin_local[0];
        }
        else
        {
            do
            {
                s_mod=0.0;
                for(j_S=0; j_S<dim_S; j_S=j_S+1)
                {
                    // spin_local[j_S] = (1.0 - 2.0 * (double)rand_r(&random_seed[cache_size*thread_num_if_parallel()])/(double)(RAND_MAX));
                    spin_local[j_S] = (1.0 - 2.0 * genrand64_real1(thread_num_if_parallel()));

                    s_mod = s_mod + spin_local[j_S] * spin_local[j_S];
                }
            }
            while(s_mod >= 1 || s_mod <= limit);
            s_mod = sqrt(s_mod);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                spin_local[j_S] = spin_local[j_S] / s_mod;
                // spin_new[dim_S*xyzi + j_S] = spin_local[j_S];
                // Energy_nu = Energy_nu + field_site[dim_S*xyzi + j_S] * spin_local[j_S];
                Energy_nu = Energy_nu + field_local[j_S] * spin_local[j_S];
            }
        }
        
        return Energy_nu;
    }

//====================  MonteCarlo-Wolff/Cluster      ====================//

    int generate_random_axis(double *reflection_planes)
    {
        int j_S;
        double s_mod = 0.0;
        double limit = 0.01 * dim_S;
        do
        {
            s_mod = 0.0;
            for(j_S=0; j_S<dim_S; j_S=j_S+1)
            {
                // reflection_plane[j_S] = (-1.0 + 2.0 * (double)rand_r(&random_seed[cache_size*thread_num_if_parallel()])/(double)(RAND_MAX));
                // reflection_planes[j_S] = (-1.0 + 2.0 * (double)rand_r(&random_seed[cache_size*thread_num_if_parallel()])/(double)(RAND_MAX));
                reflection_planes[j_S] = (-1.0 + 2.0 * genrand64_real1(thread_num_if_parallel()));
                
                // s_mod = s_mod + (reflection_plane[j_S] * reflection_plane[j_S]);
                s_mod = s_mod + (reflection_planes[j_S] * reflection_planes[j_S]);
            }
        }
        while(s_mod >= 1 || s_mod <= limit);
        s_mod = sqrt(s_mod);
        
        for(j_S=0; j_S<dim_S; j_S++)
        {
            // reflection_plane[j_S] = reflection_plane[j_S] / s_mod;
            reflection_planes[j_S] = reflection_planes[j_S] / s_mod;
        }
        return 0;
    }

    // int matrix_from_reflection_axis()
    int matrix_from_reflection_axis(double *reflection_planes, double *reflection_matrices)
    {
        int j_S, jj_S;
        for (j_S; j_S<dim_S; j_S++)
        {
            for (jj_S; jj_S<dim_S; jj_S++)
            {
                // reflection_matrix[j_S*dim_S+jj_S] = reflection_plane[j_S] * reflection_plane[jj_S];
                reflection_matrices[j_S*dim_S+jj_S] = -2 * reflection_planes[j_S] * reflection_planes[jj_S];
            }
            reflection_matrices[j_S*dim_S+j_S] += 1;
        }
        return 0;
    }

    int transform_spin(long int xyzi, double* __restrict__ spin_local)
    {
        double Si_dot_ref = 0;
        int j_S;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            Si_dot_ref += spin[dim_S*xyzi + j_S] * reflection_plane[j_S];
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            spin_local[j_S] = spin[dim_S*xyzi + j_S] - 2 * Si_dot_ref * reflection_plane[j_S];
        }
        return 0;
    }

    double E_site_old(long int xyzi)
    {
        double energy_site = 0;
        int j_S;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            #ifdef RANDOM_FIELD
            energy_site = -(h[j_S] + h_random[xyzi*dim_S + j_S]) * spin[dim_S*xyzi + j_S];
            #else
            energy_site = -(h[j_S]) * spin[dim_S*xyzi + j_S];
            #endif

        }

        return energy_site;
    }

    double E_site_new(long int xyzi, double* __restrict__ spin_local)
    {
        double energy_site = 0;

        int j_S;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            #ifdef RANDOM_FIELD
            energy_site = -(h[j_S] + h_random[xyzi*dim_S + j_S]) * spin_local[j_S];
            #else
            energy_site = -(h[j_S]) * spin_local[j_S];
            #endif
        }

        return energy_site;
    }

    double E_bond_old(long int xyzi, int j_L, int k_L, long int xyzi_nn)
    {
        double energy_bond = 0;
        double Si_dot_ref = 0;
        int j_S;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            #ifdef RANDOM_BOND
            energy_bond = -(J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*xyzi + j_S] * spin[dim_S*xyzi_nn + j_S];
            #else
            energy_bond = -(J[j_L]) * spin[dim_S*xyzi + j_S] * spin[dim_S*xyzi_nn + j_S];
            #endif
        }

        return energy_bond;
    }

    double E_bond_new(long int xyzi, int j_L, int k_L, double* __restrict__ spin_local)
    {
        double energy_bond = 0;
        double Si_dot_ref = 0;
        int j_S;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            #ifdef RANDOM_BOND
            energy_bond = -(J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*xyzi + j_S] * spin_local[j_S];
            #else
            energy_bond = -(J[j_L]) * spin[dim_S*xyzi + j_S] * spin_local[j_S];
            #endif
        }

        return energy_bond;
    }

    double activation_probability_Wolff(long int xyzi, long int xyzi_nn, int j_LL, int init)
    {
        static int stat_init = 0;
        // static int *first_call = NULL;
        static int poss_config = 1;
        static double **exp_Si_Sj = NULL;
        if (init==0)
        {
            // printf("______ZZZZZZZZZZZZZZZZZZZZZ______%d", stat_init);
            // printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
            // fflush(stdout);
            int spin_config = 0;
            switch(stat_init){
                case 1: { // if ( stat_init==1 ) {
                    spin_config = ( -spin[xyzi] * spin[xyzi_nn] ) + 1;
                    spin_config /= 2;
                    break;
                }
                case 2: { // else if ( stat_init==2 ) {
                    spin_config = ( -spin[xyzi] * spin[xyzi_nn] ) + 1;
                    spin_config /= 2;
                    spin_config += (j_LL); // j_LL = 2*j_L+edge
                    break;
                }
                case 3: { // else if ( stat_init==3 ) {
                    spin_config = ( -spin[xyzi] * spin[xyzi_nn] ) + 1;
                    spin_config /= 2;
                    spin_config += ((j_LL%2)*(2+j_LL-(j_LL%2))); // j_LL = 2*j_L+edge
                    break;
                }
                case 4: { // else { // if ( stat_init==4 ) {
                    spin_config = ( -spin[xyzi] * spin[xyzi_nn] ) + 1;
                    spin_config /= 2;
                    spin_config += (2*j_LL); // j_LL = 2*j_L+edge

                    break;
                }
            }
            // spin_config *= (-spin[xyzi]);
            return exp_Si_Sj[thread_num_if_parallel()][spin_config];
        }

        else if ( stat_init == 0 || init==1 )
        {
            // printf("______XXXXXXXXXXXXXXXXXXXXX______%d", num_of_threads);
            // printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
            // if (first_call==NULL) { first_call = (int *)malloc(num_of_threads*sizeof(int)); }
            // memset(first_call, 1, num_of_threads);
            
            stat_init = 1;
            int t_i,j_L;
            for (j_L=1; j_L<dim_L; j_L++) {
                if ( J[0]!=J[j_L] ) { stat_init += 1; break; }
            }
            for (j_L=0; j_L<dim_L; j_L++) {
                if ( BC[0]!=1 ) { stat_init += 2; break; }
            }

            // printf("%d",stat_init);
            switch(stat_init){
                case 1: { // if ( stat_init==1 ) { 
                    poss_config = 2; 
                    break;
                } 
                case 2: { // else if ( stat_init==2 ) { 
                    poss_config = 2*dim_L; 
                    break;
                } 
                case 3: { // else if ( stat_init==3 ) { 
                    poss_config = 2*(1+dim_L); 
                    break;
                } 
                case 4: { // else { /* if ( stat_init==4 ) { */ 
                    poss_config = 2*2*dim_L; 
                    break;
                } 
            }

            if ( exp_Si_Sj==NULL ) {
                exp_Si_Sj = (double **)malloc(num_of_threads*sizeof(double *));
                for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i] = (double *)malloc(poss_config*sizeof(double)); }
            }
            else {
                for (t_i=0; t_i<num_of_threads; t_i++) {
                    free(exp_Si_Sj[t_i]);
                    exp_Si_Sj[t_i] = (double *)malloc(poss_config*sizeof(double));
                }
            }

            // printf("______YYYYYYYYYYYYYYYYYYYYY______%d", stat_init);
            // printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
            switch(stat_init){
                case 1: { // if (stat_init==1) {
                    double delta_E_IM;
                    int i;
                    for (i=0; i<poss_config; i++) {
                        int p_i=i;
                        delta_E_IM = 0.0;

                        delta_E_IM += (double)(p_i*2-1) * (-2*J[0]);

                        double prob = 1.0;
                        if (delta_E_IM*T < 0) { prob = 1.0 - exp(delta_E_IM/T); }
                        else if (delta_E_IM > 0) { prob = 0.0; }
                        for (t_i=0; t_i<num_of_threads; t_i++){ exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 2: { // else if (stat_init==2) {
                    double delta_E_IM;
                    int i;
                    for (i=0; i<poss_config; i++) {
                        int p_i=i;
                        delta_E_IM = 0.0; 
                        
                        delta_E_IM += (double)((p_i%2)*2-1) * (-2*J[(p_i-(p_i%2))/2]);
                        
                        double prob = 1.0;
                        if (delta_E_IM*T < 0) { prob = 1.0 - exp(delta_E_IM/T); }
                        else if (delta_E_IM > 0) { prob = 0.0; }
                        for (t_i=0; t_i<num_of_threads; t_i++){ exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 3: { // else if (stat_init==3) {
                    double delta_E_IM;
                    int i;
                    for (i=0; i<poss_config; i++) {
                        int p_i=i;
                        delta_E_IM = 0.0;

                        if (p_i<2) {
                            delta_E_IM += (double)((p_i%2)*2-1) * (-2*J[0]);
                        }
                        else {
                            p_i -= 2;
                            delta_E_IM += (double)((p_i%2)*2-1) * (BC[(p_i-(p_i%2))/2]) * (-2*J[0]);
                        }

                        double prob = 1.0;
                        if (delta_E_IM*T < 0) { prob = 1.0 - exp(delta_E_IM/T); }
                        else if (delta_E_IM > 0) { prob = 0.0; }
                        for (t_i=0; t_i<num_of_threads; t_i++){ exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 4: { // else { // if (stat_init==4) {
                    double delta_E_IM;
                    int i;
                    for (i=0; i<poss_config; i++) {
                        int p_i=i;
                        delta_E_IM = 0.0;
                        
                        delta_E_IM += (double)((p_i%2)*2-1) ;
                        p_i = (p_i-(p_i%2))/2;
                        int p_l = (p_i-(p_i%2))/2;
                        delta_E_IM *= ( 1 - (double)(p_i%2) * ( 1 - BC[p_l] ) ) ;
                        delta_E_IM *= (-2*J[p_l]);

                        double prob = 1.0;
                        if (delta_E_IM*T < 0) { prob = 1.0 - exp(delta_E_IM/T); }
                        else if (delta_E_IM > 0) { prob = 0.0; }
                        for (t_i=0; t_i<num_of_threads; t_i++){ exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
            }
            // memset(first_call, 0, num_of_threads);
            return 0;
        }
        return 1.0;
    }

    #ifdef UPDATE_WOLFF_GHOST
    int nucleate_from_site(long int xyzi)
    {
        
        // double p_cluster = 0;
        double spin_reflected[dim_S];
        transform_spin(xyzi, spin_reflected);
        delta_E_cluster -= E_site_old(xyzi);
        delta_E_cluster += E_site_new(xyzi, spin_reflected);
        update_spin_single(xyzi, spin_reflected);
        cluster[xyzi] = 1;
        /*     if (delta_E_cluster <= 0)
            {
                update_spin_single(xyzi, spin_reflected);
            }
            else
            {
                if (T > 0)
                {
                    p_cluster = exp(delta_E_cluster/T);
                }
                else
                {
                    p_cluster = 1;
                }
                // double r_cluster = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()]) / (double) RAND_MAX;
                double r_cluster = genrand64_real1(thread_num_if_parallel());
                
                if (r_cluster < p_cluster)
                {
                    update_spin_single(xyzi, spin_reflected);
                }
            }
            cluster[xyzi] = 1; 
        */
        
        /*     double delta_E_site = 0;
            double spin_reflected[dim_S];
            transform_spin(xyzi, spin_reflected);
            delta_E_site -= E_site_old(xyzi);
            delta_E_site += E_site_new(xyzi, spin_reflected);
            if (delta_E_site <= 0)
            {
                update_spin_single(xyzi, spin_reflected);
                cluster[xyzi] = 1;
            }
            else
            {
                // double r_site = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()]) / (double) RAND_MAX;
                double r_site = genrand64_real1(thread_num_if_parallel());
                
                if (r_site<exp(-delta_E_site/T))
                {
                    update_spin_single(xyzi, spin_reflected);
                    cluster[xyzi] = 1;
                }
                // else
                // {
                //     return 0;
                // }
            }
        */

        
        int j_L, k_L;

        for (j_L=0; j_L<dim_L; j_L++)
        {
            for (k_L=0; k_L<2; k_L++)
            {
                long int xyzi_nn = N_N_I[2*dim_L*xyzi + 2*j_L + k_L];
                if (cluster[xyzi_nn] == 0)
                {
                    double p_bond = 0;
                    double spin_reflected[dim_S];
                    transform_spin(xyzi_nn, spin_reflected);
                    double delta_E_bond = 0;
                    delta_E_bond -= E_bond_old(xyzi, j_L, k_L, xyzi_nn);
                    delta_E_bond += E_bond_new(xyzi, j_L, k_L, spin_reflected);
                    if (delta_E_bond < 0)
                    {
                        if (T > 0)
                        {
                            p_bond = 1 - exp(delta_E_bond/T);
                        }
                        else
                        {
                            p_bond = 1;
                        }
                        // double r_bond = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()]) / (double) RAND_MAX;
                        double r_bond = genrand64_real1(thread_num_if_parallel());
                        
                        if (r_bond < p_bond)
                        {
                            nucleate_from_site(xyzi_nn);
                        }
                    }

                }
            }
        }

        return 0;
    }
    #else
    #ifdef UPDATE_WOLFF_BFS
    int nucleate_from_site(long int xyzi)
    {
        int j_SS = 0, j_L, k_L;
        double spin_reflected[dim_S];
        transform_spin(xyzi, spin_reflected);
        double delta_E_cluster = 0;
        delta_E_cluster -= E_site_old(xyzi);
        delta_E_cluster += E_site_new(xyzi, spin_reflected);
        double delta_E_bond = 0;
        nucleation_sites[0] = xyzi;
        long int xyzi_nn = nucleation_sites[0];
        long int i_1=0, i_2=1;
        update_spin_single(xyzi, spin_reflected);
        i_2 = 1;
        cluster[xyzi] = 1;
        double p_bond = 0;
        double r_bond = 0;

        do
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                for (k_L=0; k_L<2; k_L++)
                {
                    xyzi_nn = N_N_I[2*dim_L*nucleation_sites[i_1] + 2*j_L + k_L];
                    if ( xyzi_nn < no_of_sites )
                    {
                        if (cluster[xyzi_nn]==0)
                        {
                            p_bond = 0;
                            #ifdef C_IM
                            p_bond = activation_probability_Wolff(nucleation_sites[i_1], xyzi_nn, j_L*2, 0);
                            #else
                            transform_spin(xyzi_nn, spin_reflected);
                            delta_E_bond = 0;
                            delta_E_bond -= E_bond_old(nucleation_sites[i_1], j_L, k_L, xyzi_nn);
                            delta_E_bond += E_bond_new(nucleation_sites[i_1], j_L, k_L, spin_reflected);
                            if (delta_E_bond < 0)
                            {
                                if (T > 0)
                                {
                                    p_bond = 1 - exp(delta_E_bond/T);
                                }
                                else
                                {
                                    p_bond = 1;
                                }
                            #endif
                                // r_bond = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()]) / (double) RAND_MAX;
                                r_bond = genrand64_real1(thread_num_if_parallel());
                                
                                if (r_bond < p_bond)
                                {
                                    nucleation_sites[i_2] = xyzi_nn;
                                    i_2++;
                                    #ifdef C_IM
                                    spin[xyzi_nn] = -spin[xyzi_nn];
                                    #else
                                    update_spin_single(xyzi_nn, spin_reflected);
                                    #endif
                                    cluster[xyzi_nn] = 1;
                                    // nucleate_from_site(xyzi_nn);
                                }
                            #ifdef C_IM
                            #else
                            }
                            #endif
                        }
                    }
                }
            }
            i_1++;
        }
        while(i_1 != i_2);

        return i_2;
    }
    #else
    int nucleate_from_site(long int xyzi)
    {
        // double p_cluster = 0;
        double spin_reflected[dim_S];
        transform_spin(xyzi, spin_reflected);
        double delta_E_cluster = 0;
        delta_E_cluster -= E_site_old(xyzi);
        delta_E_cluster += E_site_new(xyzi, spin_reflected);
        update_spin_single(xyzi, spin_reflected);
        cluster[xyzi] = 1;
        /*     if (delta_E_cluster <= 0)
            {
                update_spin_single(xyzi, spin_reflected);
            }
            else
            {
                if (T > 0)
                {
                    p_cluster = exp(delta_E_cluster/T);
                }
                else
                {
                    p_cluster = 1;
                }
                // double r_cluster = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()]) / (double) RAND_MAX;
                double r_cluster = genrand64_real1(thread_num_if_parallel());
                
                if (r_cluster < p_cluster)
                {
                    update_spin_single(xyzi, spin_reflected);
                }
            }
            cluster[xyzi] = 1; 
        */
        
        /*  double delta_E_site = 0;
            double spin_reflected[dim_S];
            transform_spin(xyzi, spin_reflected);
            delta_E_site -= E_site_old(xyzi);
            delta_E_site += E_site_new(xyzi, spin_reflected);
            if (delta_E_site <= 0)
            {
                update_spin_single(xyzi, spin_reflected);
                cluster[xyzi] = 1;
            }
            else
            {
                // double r_site = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()]) / (double) RAND_MAX;
                double r_site = genrand64_real1(thread_num_if_parallel());
                
                if (r_site<exp(-delta_E_site/T))
                {
                    update_spin_single(xyzi, spin_reflected);
                    cluster[xyzi] = 1;
                }
                // else
                // {
                //     return 0;
                // }
            }
        */

        
        int j_L, k_L;

        for (j_L=0; j_L<dim_L; j_L++)
        {
            for (k_L=0; k_L<2; k_L++)
            {
                long int xyzi_nn = N_N_I[2*dim_L*xyzi + 2*j_L + k_L];
                if (cluster[xyzi_nn] == 0)
                {
                    double p_bond = 0;
                    double spin_reflected[dim_S];
                    transform_spin(xyzi_nn, spin_reflected);
                    double delta_E_bond = 0;
                    delta_E_bond -= E_bond_old(xyzi, j_L, k_L, xyzi_nn);
                    delta_E_bond += E_bond_new(xyzi, j_L, k_L, spin_reflected);
                    if (delta_E_bond < 0)
                    {
                        if (T > 0)
                        {
                            p_bond = 1 - exp(delta_E_bond/T);
                        }
                        else
                        {
                            p_bond = 1;
                        }
                        // double r_bond = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()]) / (double) RAND_MAX;
                        double r_bond = genrand64_real1(thread_num_if_parallel());
                        if (r_bond < p_bond)
                        {
                            nucleate_from_site(xyzi_nn);
                        }
                    }

                }
            }
        }

        return 0;
    }
    #endif
    #endif

    int revert_cluster()
    {
        long int i;
        #pragma omp parallel for
        for (i=0; i<no_of_sites; i++)
        {
            if (cluster[i] != 0)
            {
                double spin_reflected[dim_S];
                transform_spin(i, spin_reflected);
                update_spin_single(i, spin_reflected);
            }
        }
        
        return 0;
    }

    int set_cluster_s(int s)
    {
        long int i;
        #pragma omp parallel for
        for (i=0; i<no_of_sites; i++)
        {
            cluster[i] = s;
        }
        // memset(cluster, s, no_of_sites);
        cluster[no_of_sites] = -1;
        return 0;
    }

    #ifdef UPDATE_WOLFF_GHOST
    int random_Wolff_sweep(long int iter)
    {
        long int xyzi;
        int i, j_S;
        for (i=0; i<iter; i++)
        {
            set_cluster_s(0);
            
            generate_random_axis(reflection_plane);
            matrix_from_reflection_axis(reflection_plane, reflection_matrix);
            generate_random_axis(ghost_spin);
            matrix_from_reflection_axis(ghost_spin, ghost_matrix);
            delta_E_cluster = 0;
            // xyzi = rand_r(&random_seed[cache_size*thread_num_if_parallel()]) % no_of_sites;
            xyzi = genrand64_int64(thread_num_if_parallel()) % no_of_sites;
            for (xyzi=0; xyzi<no_of_sites; xyzi++)
            {
                if (cluster[xyzi] == 0)
                {
                    nucleate_from_site(xyzi);
                }
            }
        }

        return 0;
    }
    #else
    #ifdef PARALLEL_WOLFF
    int random_Wolff_sweep(long int iter)
    {
        long int xyzi, i;
        int j_S;

        for (i=0; i<iter; i++)
        {
            printf(": Step = %ld ... ", i);
            printf("\b\b\b\b\b\b\b\b\b\b"       );
            printf(           "\b\b\b\b\b");
            long int i_iter = i/10;
            while (i_iter>0)
            {
                printf("\b");
                i_iter = i_iter/10;
            }
            fflush(stdout);
            set_cluster_s(0);
            
            // xyzi = rand_r(&random_seed[cache_size*thread_num_if_parallel()]) % no_of_sites;
            xyzi = genrand64_int64(thread_num_if_parallel()) % no_of_sites;
            nucleation_sites[0] = xyzi;

            #ifdef C_IM
            spin[xyzi] = -spin[xyzi];
            #else
            generate_random_axis(reflection_plane);
            double spin_reflected_0[dim_S];
            transform_spin(xyzi, spin_reflected_0);
            double delta_E_cluster = 0;
            delta_E_cluster -= E_site_old(xyzi);
            delta_E_cluster += E_site_new(xyzi, spin_reflected_0);
            update_spin_single(xyzi, spin_reflected_0);
            #endif
            cluster[xyzi] = 1;
            
            
            long int remaining_sites = no_of_sites;
            long int no_of_nuclei = 1;
            long int i_1 = no_of_sites-remaining_sites;
            long int i_2 = no_of_sites-remaining_sites+no_of_nuclei;
            long int i_3 = no_of_sites-remaining_sites+no_of_nuclei;
            long int ii;
            int threads_to_launch = (num_of_threads>2*dim_L?2*dim_L:num_of_threads);
            while (i_1!=i_2)
            {
                int j_L,k_L;
                #pragma omp parallel for private(j_L,k_L) collapse(3) num_threads(threads_to_launch)//schedule(static,i_2-i_1) // if(i_2-i_1>1) // private(j_L,k_L) 
                for (ii=i_1; ii<i_2; ii++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        for (k_L=0; k_L<2; k_L++)
                        {
                            long int next_site = N_N_I[2*dim_L*nucleation_sites[ii] + 2*j_L + k_L];
                            if (next_site<no_of_sites)
                            {
                                if (cluster[next_site] == 0)
                                {
                                    double p_bond = 0;
                                    #ifdef C_IM
                                    // printf("\rhere.");
                                    p_bond = activation_probability_Wolff(nucleation_sites[ii], next_site, j_L*2, 0);
                                    #else
                                    double spin_reflected[dim_S];
                                    transform_spin(next_site, spin_reflected);
                                    double delta_E_bond = 0;
                                    delta_E_bond -= E_bond_old(nucleation_sites[ii], j_L, k_L, next_site);
                                    delta_E_bond += E_bond_new(nucleation_sites[ii], j_L, k_L, spin_reflected);
                                    if (delta_E_bond < 0)
                                    {
                                        if (T > 0)
                                        {
                                            p_bond = 1 - exp(delta_E_bond/T);
                                        }
                                        else
                                        {
                                            p_bond = 1;
                                        }
                                    #endif
                                        // double r_bond = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()]) / (double) RAND_MAX;
                                        double r_bond = genrand64_real1(thread_num_if_parallel());
                                        if (r_bond < p_bond)
                                        {
                                            int update_this_thread = 0;
                                            // #pragma omp flush(cluster)
                                            #pragma omp critical
                                            if ( cluster[next_site] != 1 )
                                            {
                                                update_this_thread = 1+i_2;
                                                i_2++;
                                                // update_this_thread = 1+i_3;
                                                // i_3++;
                                                cluster[next_site] = 1;
                                            }
                                            
                                            if (update_this_thread)
                                            {
                                                // #pragma omp flush(cluster) 
                                                #ifdef C_IM
                                                spin[next_site] = -spin[next_site];
                                                #else
                                                update_spin_single(next_site, spin_reflected);
                                                #endif

                                                nucleation_sites[update_this_thread-1] = next_site;
                                                // i_3++;
                                            }
                                        }
                                    #ifdef C_IM
                                    #else
                                    }
                                    #endif
                                }
                            }
                        }
                    }
                }

                i_1 = i_2;
                i_2 = i_3;
            }
            
            // printf("%ld \n",i_3);
            
            double p_cluster;
            // if (r_cluster<exp(delta_E_cluster/T))
            // {
            //     revert_cluster();
            // }
            if (delta_E_cluster <= 0)
            {
                p_cluster = 1.0;
            }
            else 
            {
                if (T == 0)
                {
                    p_cluster = 0.0;
                }
                else 
                {
                    p_cluster = exp(-delta_E_cluster/T);
                }
            }
            // double r_cluster = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()])/ (double) RAND_MAX;
            double r_cluster = genrand64_real1(thread_num_if_parallel());
            if(r_cluster >= p_cluster)
            {
                revert_cluster();
            }
        }

        return 0;
    }
    #else
    int random_Wolff_sweep(long int iter)
    {
        long int xyzi, i;
        int j_S;

        for (i=0; i<iter; i++)
        {
            printf(": Step = %ld ... ", i);
            printf("\b\b\b\b\b\b\b\b\b\b"       );
            printf(           "\b\b\b\b\b");
            long int i_iter = i/10;
            while (i_iter>0)
            {
                printf("\b");
                i_iter = i_iter/10;
            }
            fflush(stdout);
            set_cluster_s(0);
            
            // do
            // {
                generate_random_axis(reflection_plane);
                double delta_E_cluster = 0;
                // xyzi = rand_r(&random_seed[cache_size*thread_num_if_parallel()]) % no_of_sites;
                xyzi = genrand64_int64(thread_num_if_parallel()) % no_of_sites;

                nucleate_from_site(xyzi);
                double p_cluster;
                // if (r_cluster<exp(delta_E_cluster/T))
                // {
                //     revert_cluster();
                // }
                if (delta_E_cluster <= 0)
                {
                    p_cluster = 1.0;
                }
                else 
                {
                    if (T == 0)
                    {
                        p_cluster = 0.0;
                    }
                    else 
                    {
                        p_cluster = exp(-delta_E_cluster/T);
                    }
                }
                // double r_cluster = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()])/ (double) RAND_MAX;
                double r_cluster = genrand64_real1(thread_num_if_parallel());
                if(r_cluster >= p_cluster)
                {
                    revert_cluster();
                }
            // }
            // while (cluster[xyzi] != 1);
        }

        return 0;
    }
    #endif
    #endif

//====================  Swendsen-Wang                 ====================//
    
    int populate_bonds()
    {
        long int i;
        int j_L;
        // for (i=0; i<no_of_sites; i++)
        // {
            
        // }

        return 0;
    }
    
    int break_into_clusters()
    {
        return 0;
    }

    int set_nucleation_site()
    {
        long int i;
        #pragma omp parallel for
        for (i=0; i<no_of_sites; i++)
        {
            nucleation_sites[i] = i;
        }
        
        return 0;
    }

    int full_Wolff_sweep(long int iter)
    {
        long int xyzi, i;
        int j_S;

        for (i=0; i<iter; i++)
        {
            printf(": Step = %ld ... ", i);
            printf("\b\b\b\b\b\b\b\b\b\b"       );
            printf(           "\b\b\b\b\b");
            long int i_iter = i/10;
            while (i_iter>0)
            {
                printf("\b");
                i_iter = i_iter/10;
            }
            fflush(stdout);
            set_cluster_s(0);
            set_nucleation_site();
            #ifdef C_IM
            #else
            generate_random_axis(reflection_plane);
            #endif
            
            long int remaining_sites = no_of_sites;
            long int no_of_nuclei = 1;
            long int i_0 = 0;
            long int i_1 = no_of_sites-remaining_sites;
            long int i_2 = no_of_sites-remaining_sites+no_of_nuclei;
            long int i_3 = no_of_sites-remaining_sites+no_of_nuclei;
            long int ii = 0;
            long int no_of_clusters = 0;
            long int next_to_visit=0;

            while ( remaining_sites>0 )
            {
                no_of_clusters++;
                // next_to_visit = 0;
                while ( cluster[next_to_visit]!=0 )
                {
                    next_to_visit++;
                }
                // if (next_to_visit>=no_of_sites)
                // {
                //     printf("%ld,%ld,%ld,%ld,%ld,%ld                               \n", next_to_visit,i_0,i_1,i_2,i_3,no_of_clusters);
                //     break;
                // }
                // i_0 = no_of_sites-remaining_sites;
                // i_1 = no_of_sites-remaining_sites;
                // i_2 = no_of_sites-remaining_sites+no_of_nuclei;
                // i_3 = no_of_sites-remaining_sites+no_of_nuclei;

                nucleation_sites[i_1] = next_to_visit;
                cluster[next_to_visit] = no_of_clusters;
                #ifdef C_IM
                spin[next_to_visit] = -spin[next_to_visit];
                #else
                double spin_reflected_0[dim_S];
                transform_spin(next_to_visit, spin_reflected_0);
                update_spin_single(next_to_visit, spin_reflected_0);
                #endif

                while (i_1!=i_2)
                {
                    int j_L,k_L;
                    #pragma omp parallel for private(j_L,k_L) collapse(3) //schedule(static,i_2-i_1) // if(i_2-i_1>1) 
                    for (ii=i_1; ii<i_2; ii++)
                    {
                        for (j_L=0; j_L<dim_L; j_L++)
                        {
                            for (k_L=0; k_L<2; k_L++)
                            {
                                long int next_site = N_N_I[2*dim_L*nucleation_sites[ii] + 2*j_L + k_L];
                                if (next_site<no_of_sites)
                                {
                                    if (cluster[next_site] == 0)
                                    {
                                        double p_bond = 0;
                                        #ifdef C_IM
                                        // printf("\rhere.");
                                        p_bond = activation_probability_Wolff(nucleation_sites[ii], next_site, j_L, 0);
                                        #else
                                        double spin_reflected[dim_S];
                                        transform_spin(next_site, spin_reflected);
                                        double delta_E_bond = 0;
                                        delta_E_bond -= E_bond_old(nucleation_sites[ii], j_L, k_L, next_site);
                                        delta_E_bond += E_bond_new(nucleation_sites[ii], j_L, k_L, spin_reflected);
                                        if (delta_E_bond < 0)
                                        {
                                            if (T > 0)
                                            {
                                                p_bond = 1 - exp(delta_E_bond/T);
                                            }
                                            else
                                            {
                                                p_bond = 1;
                                            }
                                        #endif
                                            // double r_bond = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()]) / (double) RAND_MAX;
                                            double r_bond = genrand64_real1(thread_num_if_parallel());
                                            if (r_bond < p_bond)
                                            {
                                                int update_this_thread = 0;
                                                // #pragma omp flush(cluster)
                                                #pragma omp critical
                                                if ( cluster[next_site] != no_of_clusters )
                                                {
                                                    update_this_thread = 1+i_3;
                                                    i_3++;
                                                    cluster[next_site] = no_of_clusters;
                                                }
                                                
                                                if (update_this_thread)
                                                {
                                                    // #pragma omp flush(cluster)
                                                    #ifdef C_IM
                                                    spin[next_site] = -spin[next_site];
                                                    #else
                                                    update_spin_single(next_site, spin_reflected);
                                                    #endif

                                                    nucleation_sites[update_this_thread-1] = next_site;
                                                    // i_3++;
                                                }
                                            }
                                        #ifdef C_IM
                                        #else
                                        }
                                        #endif
                                    }
                                }
                            }
                        }
                    }
                    i_1 = i_2;
                    i_2 = i_3;
                }
                
                if (genrand64_int64(thread_num_if_parallel()) % 2)
                {
                    #pragma omp parallel for
                    for (ii=i_0; ii<i_3; ii++)
                    {
                        #ifdef C_IM
                        spin[nucleation_sites[ii]] = -spin[nucleation_sites[ii]]; 
                        #else
                        double spin_reflected[dim_S];
                        transform_spin(ii, spin_reflected);
                        update_spin_single(ii, spin_reflected);
                        #endif
                    }
                }
                i_0=i_1;
                remaining_sites = no_of_sites-i_3;
                i_2++;
                i_3++;
            }

            // printf("%ld \n",no_of_clusters);

            // for (ii=i_0; ii<i_3; ii++)
            // {
            //     if (cluster[ii]==0)
            //     {
            //         printf("%d",cluster[ii]);
            //         return 0;
            //     }
            // }
        }
        return 0;
    }

//====================  Metropolis                    ====================//
    
    double activation_probability_Metropolis(long int xyzi, int init)
    {
        static int stat_init = 0;
        // static int *first_call = NULL;
        static int poss_config = 1;
        static double **exp_Si_Sj = NULL;
        if (init==0)
        {
            int j_L;
            int spin_config = 0;
            switch (stat_init) {
                case 1: { // if ( stat_init==1 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config += ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) ;
                    }
                    spin_config *= (-spin[xyzi]) ;
                    spin_config /= 2 ;
                    spin_config += dim_L ;
                    break;
                }
                case 2: { // else if ( stat_init==2 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config += ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] );
                    }
                    spin_config *= (-spin[xyzi]);
                    spin_config += (2*dim_L + (-spin[xyzi]+1)/2);
                    break;
                }
                case 3: { // else if ( stat_init==3 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config *= 3; 
                        spin_config += ( ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) * (-spin[xyzi]) + 2 );
                    }
                    spin_config /= 2;
                    break;
                }
                case 4: { // else if ( stat_init==4 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config *= 3;
                        spin_config += (( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) * (-spin[xyzi]) + 2);
                    }
                    spin_config += ((-spin[xyzi]+1)/2);
                    break;
                }
                case 5: { // else if ( stat_init==5 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config += ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) ;
                    }
                    spin_config *= (-spin[xyzi]) ;
                    spin_config += (2*dim_L) ;
                    break;
                }
                case 6: { // else if ( stat_init==6 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config += ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] );
                    }
                    spin_config *= (-spin[xyzi]*2);
                    spin_config += (4*dim_L + (int)(spin[xyzi]!=1));
                    break;
                }
                case 7: { // else if ( stat_init==7 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config *= (3+2*(BC[j_L]!=1));
                        spin_config += ( ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) * (-spin[xyzi]) + 2 ) * (1+(BC[j_L]!=1));
                    }
                    spin_config /= 2;
                    break;
                }
                case 8: { // else {  // if ( stat_init==8 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config *= (3+2*(BC[j_L]!=1));
                        spin_config += ( ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) * (-spin[xyzi]) + 2 ) * (1+(BC[j_L]!=1));
                    }
                    spin_config += (int)(spin[xyzi]!=1);
                    break;
                }
            }
            // spin_config *= (-spin[xyzi]);
            return exp_Si_Sj[thread_num_if_parallel()][spin_config];
        }

        else if ( stat_init == 0 || init==1 )
        {
            // if (first_call==NULL) { first_call = (int *)malloc(num_of_threads*sizeof(int)); }
            // memset(first_call, 1, num_of_threads);
            
            stat_init = 1;
            stat_init += (int)(h[0]!=0);
            int t_i,j_L;
            for (j_L=1; j_L<dim_L; j_L++) {
                if ( J[0]!=J[j_L] ) { stat_init += 2; break; }
            }
            for (j_L=0; j_L<dim_L; j_L++) {
                if ( BC[j_L]!=1 ) { stat_init += 4; break; }
            }

            switch (stat_init){
                case 1: { // if ( stat_init==1 ) { 
                    poss_config = 2*dim_L+1; 
                }
                case 2: { // else if ( stat_init==2 ) { 
                    poss_config = (2*dim_L+1)*2; 
                }
                case 3: { // else if ( stat_init==3 ) {
                    poss_config = 1;
                    for (j_L=0; j_L<dim_L; j_L++) { poss_config *= 3; }
                }
                case 4: { // else if ( stat_init==4 ) {
                    poss_config = 2;
                    for (j_L=0; j_L<dim_L; j_L++) { poss_config *= 3; }
                }
                case 5: { // else if ( stat_init==5 ) { 
                    poss_config = 4*dim_L+1; 
                }
                case 6: { // else if ( stat_init==6 ) { 
                    poss_config = (4*dim_L+1)*2; 
                }
                case 7: { // else if ( stat_init==7 ) {
                    poss_config = 1;
                    for (j_L=0; j_L<dim_L; j_L++) { poss_config *= (3+2*(1-BC[j_L])); }
                }
                case 8: { // else { // if ( stat_init==8 ) {
                    poss_config = 2;
                    for (j_L=0; j_L<dim_L; j_L++) { poss_config *= (3+2*(1-BC[j_L])); }
                }
            }
            if ( exp_Si_Sj==NULL ) {
                exp_Si_Sj = (double **)malloc(num_of_threads*sizeof(double *));
                for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i] = (double *)malloc(poss_config*sizeof(double)); }
            }
            else {
                for (t_i=0; t_i<num_of_threads; t_i++) {
                    free(exp_Si_Sj[t_i]);
                    exp_Si_Sj[t_i] = (double *)malloc(poss_config*sizeof(double));
                }
            }
            
            // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
            switch (stat_init){
                case 1: { // if (stat_init==1) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0; 
                        p_i = p_i*2;
                        delta_E_IM += (double)(p_i-dim_L*2) * (-2*J[0]);
                        
                        double prob=0.0;
                        if (delta_E_IM <= 0) { prob = 1.0; }
                        else if (T != 0) { prob = exp(-(delta_E_IM)/T); }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 2: { // else if (stat_init==2) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0; 
                        delta_E_IM += (double)((p_i%2)*2-1) * (-2*h[0]);
                        p_i = (p_i-(p_i%2));
                        delta_E_IM += (double)(p_i-2*dim_L) * (-2*J[0]);
                        
                        double prob=0.0;
                        if (delta_E_IM <= 0) { prob = 1.0; }
                        else if (T != 0) { prob = exp(-(delta_E_IM)/T); }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 3: { // else if (stat_init==3) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0;

                        for (j_config=0; j_config<dim_L; j_config++)
                        {
                            delta_E_IM += (double)(((p_i%3)-1) * 2) * (-2*J[j_config]);
                            p_i = (p_i-(p_i%3))/3;
                        }

                        double prob=0.0;
                        if (delta_E_IM <= 0) { prob = 1.0; }
                        else if (T != 0) { prob = exp(-(delta_E_IM)/T); }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 4: { // else if (stat_init==4) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0;
                        delta_E_IM += (double)((p_i%2)*2 - 1) * (-2*h[0]);
                        p_i = (p_i-(p_i%2))/2;

                        for (j_config=0; j_config<dim_L; j_config++)
                        {
                            delta_E_IM += (double)(((p_i%3)-1) * 2) * (-2*J[j_config]);
                            p_i = (p_i-(p_i%3))/3;
                        }

                        double prob=0.0;
                        if (delta_E_IM <= 0) { prob = 1.0; }
                        else if (T != 0) { prob = exp(-(delta_E_IM)/T); }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 5: { // else if (stat_init==5) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0; 
                        delta_E_IM += (double)(p_i-dim_L*2) * (-2*J[0]);
                        
                        double prob=0.0;
                        if (delta_E_IM <= 0) { prob = 1.0; }
                        else if (T != 0) { prob = exp(-(delta_E_IM)/T); }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 6: { // else if (stat_init==6) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0; 
                        delta_E_IM += (double)((p_i%2)*2-1) * (-2*h[0]);
                        p_i = (p_i-(p_i%2))/2;
                        delta_E_IM += (double)(p_i-2*dim_L) * (-2*J[0]);
                        
                        double prob=0.0;
                        if (delta_E_IM <= 0) { prob = 1.0; }
                        else if (T != 0) { prob = exp(-(delta_E_IM)/T); }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 7: { // else if (stat_init==7) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0;

                        for (j_config=0; j_config<dim_L; j_config++)
                        {
                            int bc_j = 3+2*(BC[j_config]!=1);
                            delta_E_IM += (double)(((p_i%bc_j)-1-(BC[j_config]!=1)) * (2-(BC[j_config]!=1))) * (-2*J[j_config]);
                            p_i = (p_i-(p_i%bc_j))/bc_j;
                        }

                        double prob=0.0;
                        if (delta_E_IM <= 0) { prob = 1.0; }
                        else if (T != 0) { prob = exp(-(delta_E_IM)/T); }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 8: { // else { // if (stat_init==8) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0;
                        delta_E_IM += (double)((p_i%2)*2 - 1) * (-2*h[0]);
                        p_i = (p_i-(p_i%2))/2;

                        for (j_config=0; j_config<dim_L; j_config++)
                        {
                            int bc_j = 3+2*(BC[j_config]!=1);
                            delta_E_IM += (double)(((p_i%bc_j)-1-(BC[j_config]!=1)) * (2-(BC[j_config]!=1))) * (-2*J[j_config]);
                            p_i = (p_i-(p_i%bc_j))/bc_j;
                        }

                        double prob=0.0;
                        if (delta_E_IM <= 0) { prob = 1.0; }
                        else if (T != 0) { prob = exp(-(delta_E_IM)/T); }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
            }
            // memset(first_call, 0, num_of_threads);
            return 0;
        }
        
        return 1.0;
    }

    double update_probability_Metropolis(long int xyzi)
    {
        double update_prob, r;
        #ifdef C_IM
            update_prob = activation_probability_Metropolis(xyzi, 0);
            // r = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()])/ (double) RAND_MAX;
            r = genrand64_real1(thread_num_if_parallel());
            
            if(r < update_prob)
            {
                spin[xyzi] = -spin[xyzi];
                
                return 1.0;
            }
        #else
            // double update_prob;
            // printf("\n______ZZZZZZZZZZZZZZZZZZZZZ______%d\n",thread_num_if_parallel());
            double spin_local[dim_S];
            double field_local[dim_S];
            double E_old = Energy_old(xyzi, spin_local, field_local);
            double E_new = Energy_new(xyzi, spin_local, field_local);
            if (E_new - E_old <= 0) { update_prob = 1.0; }
            else {
                if (T == 0) { update_prob = 0.0; }
                else { update_prob = exp(-(E_new-E_old)/T); }
            }
            // r = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()])/ (double) RAND_MAX;
            r = genrand64_real1(thread_num_if_parallel());
            
            if(r < update_prob)
            {
                update_spin_single(xyzi, spin_local);
                return 1.0;
            }
        #endif
        return 0.0;
    }

    int linear_Metropolis_sweep(long int iter)
    {
        long int site_i;
        do
        {
            for(site_i=0; site_i<no_of_sites; site_i++)
            {

                double update_prob = update_probability_Metropolis(site_i);

                // // double r = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()])/ (double) RAND_MAX;
                // double r = genrand64_real1(thread_num_if_parallel());
                // if(r < update_prob)
                // {
                //     update_spin_single(site_i, 1);
                // }
                // else
                // {
                //     update_spin_single(site_i, 0);
                // }
            }
            iter--;
        }
        while (iter > 0);
        return 0;
    }

    int random_Metropolis_sweep(long int iter)
    {
        // edit this
        #ifdef _OPENMP
            #ifdef PARALLEL_RANDOM_MC_SWEEP
            long int temp_i, t_i;
            int j_L;
            num_of_threads = omp_get_max_threads();
            while (iter>0)
            {
                set_cluster_s(0);
                for (t_i=0; t_i<num_of_threads; t_i++)
                {
                    do
                    {
                        // temp_i = rand_r(&random_seed[cache_size*thread_num_if_parallel()])%no_of_sites;
                        temp_i = genrand64_int64(thread_num_if_parallel())%no_of_sites;
                    } while (cluster[temp_i]==1);
                    iter--;
                    random_sites[t_i] = temp_i;
                }
                #pragma omp parallel for
                for (t_i=0; t_i<num_of_threads+(int)(iter<0)*iter; t_i++)
                {
                    update_probability_Metropolis(random_sites[t_i]);
                }
            }
            #else
            // long int *random_sites = (long int *)malloc(no_of_black_white_sites[0],sizeof(long int));
            /* int *cluster */// long int *blocked_sites = (long int *)calloc((dim_L*2)*no_of_black_white_sites[0],sizeof(long int));

            int j_L;
            
            // long int temp_i = rand_r(&random_seed[cache_size*thread_num_if_parallel()])%no_of_sites;
            long int temp_i = genrand64_int64(thread_num_if_parallel())%no_of_sites;
    
            while (iter)
            {
                long int batch size = 0;
                set_cluster_s(0);
                while (iter>0 && cluster[temp_i] == 0)
                {
                    random_sites[batch_size] = temp_i;
                    cluster[temp_i] = 1;
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        for (k_L=0; k_L<2; k_L++)
                        {
                            cluster[N_N_I[2*dim_L*temp_i + 2*j_L + k_L];] = 1;
                        }
                    }
                    iter--;
                    batch_size++;
                    // temp_i = rand_r(&random_seed[cache_size*thread_num_if_parallel()])%no_of_sites;
                    temp_i = genrand64_int64(thread_num_if_parallel())%no_of_sites;
                    
                }
                #pragma omp parallel for
                for(i_b=0; i_b<batch_size; i_b++)
                {
                    update_probability_Metropolis(random_sites[i_b]);
                }
            }
            // free(random_sites);
            #endif
        #else

            long int xyzi;
            do
            {
                // xyzi = rand_r(&random_seed[cache_size*thread_num_if_parallel()])%no_of_sites;
                xyzi = genrand64_int64(thread_num_if_parallel())%no_of_sites;
                
                double update_prob = update_probability_Metropolis(xyzi);

                // // double r = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()])/ (double) RAND_MAX;
                // double r = genrand64_real1(thread_num_if_parallel());
                // if(r < update_prob)
                // {
                //     // spin[i] = -spin[i];
                //     update_spin_single(xyzi, 1);
                // }
                // else
                // {
                //     update_spin_single(xyzi, 0);
                // }
                iter--;
            }
            while (iter > 0);
        
        #endif

        return 0;
    }

    int checkerboard_Metropolis_sweep(long int iter)
    {
        // THIS PART IS PARALLELIZABLE. *modifications may be needed
        static int black_or_white = BLACK_WHITE;
        long int i;
        while(iter > 0)
        {
            #pragma omp parallel for 
            for (i=0; i < no_of_black_white_sites[black_or_white]; i++)
            {
                long int site_index = black_white_checkerboard[no_of_black_white_sites[black_or_white]*black_or_white + i];

                double update_prob = update_probability_Metropolis(site_index);
            }
            
            black_or_white = !black_or_white;
            iter--;
            // printf(": Step = %ld ... ", iter);
            // printf("\b\b\b\b\b\b\b\b\b\b"       );
            // printf(           "\b\b\b\b\b");
            // long int i_iter = iter/10;
            // while (i_iter>0)
            // {
            //     printf("\b");
            //     i_iter = i_iter/10;
            // }
        } 
        
        
        return 0;
    }

//====================  Glauber                       ====================//

    double activation_probability_Glauber(long int xyzi, int init)
    {
        static int stat_init = 0;
        // static int *first_call = NULL;
        static int poss_config = 1;
        static double **exp_Si_Sj = NULL;
        if (init==0)
        {
            int j_L;
            int spin_config = 0;
            switch (stat_init) {
                case 1: { // if ( stat_init==1 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config += ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) ;
                    }
                    spin_config *= (-spin[xyzi]) ;
                    spin_config /= 2 ;
                    spin_config += dim_L ;
                    break;
                }
                case 2: { // else if ( stat_init==2 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config += ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] );
                    }
                    spin_config *= (-spin[xyzi]);
                    spin_config += (2*dim_L + (-spin[xyzi]+1)/2);
                    break;
                }
                case 3: { // else if ( stat_init==3 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config *= 3; 
                        spin_config += ( ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) * (-spin[xyzi]) + 2 );
                    }
                    spin_config /= 2;
                    break;
                }
                case 4: { // else if ( stat_init==4 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config *= 3;
                        spin_config += (( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) * (-spin[xyzi]) + 2);
                    }
                    spin_config += ((-spin[xyzi]+1)/2);
                    break;
                }
                case 5: { // else if ( stat_init==5 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config += ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) ;
                    }
                    spin_config *= (-spin[xyzi]) ;
                    spin_config += (2*dim_L) ;
                    break;
                }
                case 6: { // else if ( stat_init==6 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config += ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] );
                    }
                    spin_config *= (-spin[xyzi]*2);
                    spin_config += (4*dim_L + (int)(spin[xyzi]!=1));
                    break;
                }
                case 7: { // else if ( stat_init==7 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config *= (3+2*(BC[j_L]!=1));
                        spin_config += ( ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) * (-spin[xyzi]) + 2 ) * (1+(BC[j_L]!=1));
                    }
                    spin_config /= 2;
                    break;
                }
                case 8: { // else {  // if ( stat_init==8 ) {
                    for (j_L=dim_L-1; j_L>=0; j_L--) {
                        spin_config *= (3+2*(BC[j_L]!=1));
                        spin_config += ( ( spin[N_N_I[2*dim_L*xyzi+2*j_L+0]] + spin[N_N_I[2*dim_L*xyzi+2*j_L+1]] ) * (-spin[xyzi]) + 2 ) * (1+(BC[j_L]!=1));
                    }
                    spin_config += (int)(spin[xyzi]!=1);
                    break;
                }
            }
            // spin_config *= (-spin[xyzi]);
            return exp_Si_Sj[thread_num_if_parallel()][spin_config];
        }

        else if ( stat_init == 0 || init==1 )
        {
            // if (first_call==NULL) { first_call = (int *)malloc(num_of_threads*sizeof(int)); }
            // memset(first_call, 1, num_of_threads);
            
            stat_init = 1;
            stat_init += (int)(h[0]!=0);
            int t_i,j_L;
            for (j_L=1; j_L<dim_L; j_L++) {
                if ( J[0]!=J[j_L] ) { stat_init += 2; break; }
            }
            for (j_L=0; j_L<dim_L; j_L++) {
                if ( BC[j_L]!=1 ) { stat_init += 4; break; }
            }

            switch (stat_init){
                case 1: { // if ( stat_init==1 ) { 
                    poss_config = 2*dim_L+1; 
                }
                case 2: { // else if ( stat_init==2 ) { 
                    poss_config = (2*dim_L+1)*2; 
                }
                case 3: { // else if ( stat_init==3 ) {
                    poss_config = 1;
                    for (j_L=0; j_L<dim_L; j_L++) { poss_config *= 3; }
                }
                case 4: { // else if ( stat_init==4 ) {
                    poss_config = 2;
                    for (j_L=0; j_L<dim_L; j_L++) { poss_config *= 3; }
                }
                case 5: { // else if ( stat_init==5 ) { 
                    poss_config = 4*dim_L+1; 
                }
                case 6: { // else if ( stat_init==6 ) { 
                    poss_config = (4*dim_L+1)*2; 
                }
                case 7: { // else if ( stat_init==7 ) {
                    poss_config = 1;
                    for (j_L=0; j_L<dim_L; j_L++) { poss_config *= (3+2*(1-BC[j_L])); }
                }
                case 8: { // else { // if ( stat_init==8 ) {
                    poss_config = 2;
                    for (j_L=0; j_L<dim_L; j_L++) { poss_config *= (3+2*(1-BC[j_L])); }
                }
            }
            if ( exp_Si_Sj==NULL ) {
                exp_Si_Sj = (double **)malloc(num_of_threads*sizeof(double *));
                for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i] = (double *)malloc(poss_config*sizeof(double)); }
            }
            else {
                for (t_i=0; t_i<num_of_threads; t_i++) {
                    free(exp_Si_Sj[t_i]);
                    exp_Si_Sj[t_i] = (double *)malloc(poss_config*sizeof(double));
                }
            }
            
            // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
            switch (stat_init){
                case 1: { // if (stat_init==1) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0; 
                        p_i = p_i*2;
                        delta_E_IM += (double)(p_i-dim_L*2) * (-2*J[0]);
                        
                        double prob=0.0;
                        if (delta_E_IM > 0) { 
                            if (T==0) { prob = 0.0; }
                            else { double p=exp(-delta_E_IM/T); prob = p/(1+p); }
                        }
                        else if (delta_E_IM < 0) {
                            if (T==0) { prob = 1.0; }
                            else { double p=exp(delta_E_IM/T); prob = 1/(1+p); }
                        }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 2: { // else if (stat_init==2) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0; 
                        delta_E_IM += (double)((p_i%2)*2-1) * (-2*h[0]);
                        p_i = (p_i-(p_i%2));
                        delta_E_IM += (double)(p_i-2*dim_L) * (-2*J[0]);
                        
                        double prob=0.0;
                        if (delta_E_IM > 0) { 
                            if (T==0) { prob = 0.0; }
                            else { double p=exp(-delta_E_IM/T); prob = p/(1+p); }
                        }
                        else if (delta_E_IM < 0) {
                            if (T==0) { prob = 1.0; }
                            else { double p=exp(delta_E_IM/T); prob = 1/(1+p); }
                        }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 3: { // else if (stat_init==3) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0;

                        for (j_config=0; j_config<dim_L; j_config++)
                        {
                            delta_E_IM += (double)(((p_i%3)-1) * 2) * (-2*J[j_config]);
                            p_i = (p_i-(p_i%3))/3;
                        }

                        double prob=0.0;
                        if (delta_E_IM > 0) { 
                            if (T==0) { prob = 0.0; }
                            else { double p=exp(-delta_E_IM/T); prob = p/(1+p); }
                        }
                        else if (delta_E_IM < 0) {
                            if (T==0) { prob = 1.0; }
                            else { double p=exp(delta_E_IM/T); prob = 1/(1+p); }
                        }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 4: { // else if (stat_init==4) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0;
                        delta_E_IM += (double)((p_i%2)*2 - 1) * (-2*h[0]);
                        p_i = (p_i-(p_i%2))/2;

                        for (j_config=0; j_config<dim_L; j_config++)
                        {
                            delta_E_IM += (double)(((p_i%3)-1) * 2) * (-2*J[j_config]);
                            p_i = (p_i-(p_i%3))/3;
                        }

                        double prob=0.0;
                        if (delta_E_IM > 0) { 
                            if (T==0) { prob = 0.0; }
                            else { double p=exp(-delta_E_IM/T); prob = p/(1+p); }
                        }
                        else if (delta_E_IM < 0) {
                            if (T==0) { prob = 1.0; }
                            else { double p=exp(delta_E_IM/T); prob = 1/(1+p); }
                        }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 5: { // else if (stat_init==5) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0; 
                        delta_E_IM += (double)(p_i-dim_L*2) * (-2*J[0]);
                        
                        double prob=0.0;
                        if (delta_E_IM > 0) { 
                            if (T==0) { prob = 0.0; }
                            else { double p=exp(-delta_E_IM/T); prob = p/(1+p); }
                        }
                        else if (delta_E_IM < 0) {
                            if (T==0) { prob = 1.0; }
                            else { double p=exp(delta_E_IM/T); prob = 1/(1+p); }
                        }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 6: { // else if (stat_init==6) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0; 
                        delta_E_IM += (double)((p_i%2)*2-1) * (-2*h[0]);
                        p_i = (p_i-(p_i%2))/2;
                        delta_E_IM += (double)(p_i-2*dim_L) * (-2*J[0]);
                        
                        double prob=0.0;
                        if (delta_E_IM > 0) { 
                            if (T==0) { prob = 0.0; }
                            else { double p=exp(-delta_E_IM/T); prob = p/(1+p); }
                        }
                        else if (delta_E_IM < 0) {
                            if (T==0) { prob = 1.0; }
                            else { double p=exp(delta_E_IM/T); prob = 1/(1+p); }
                        }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 7: { // else if (stat_init==7) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0;

                        for (j_config=0; j_config<dim_L; j_config++)
                        {
                            int bc_j = 3+2*(BC[j_config]!=1);
                            delta_E_IM += (double)(((p_i%bc_j)-1-(BC[j_config]!=1)) * (2-(BC[j_config]!=1))) * (-2*J[j_config]);
                            p_i = (p_i-(p_i%bc_j))/bc_j;
                        }

                        double prob=0.0;
                        if (delta_E_IM > 0) { 
                            if (T==0) { prob = 0.0; }
                            else { double p=exp(-delta_E_IM/T); prob = p/(1+p); }
                        }
                        else if (delta_E_IM < 0) {
                            if (T==0) { prob = 1.0; }
                            else { double p=exp(delta_E_IM/T); prob = 1/(1+p); }
                        }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
                case 8: { // else { // if (stat_init==8) {
                    // printf("\n%d______XXXXXXXXXXXXXXXXXXXXX______%d\n", thread_num_if_parallel(), stat_init);
                    double delta_E_IM;
                    int j_config, k_config, h_config;
                    int i;
                    for (i=0; i<poss_config; i++)
                    {
                        int p_i=i;
                        delta_E_IM = 0.0;
                        delta_E_IM += (double)((p_i%2)*2 - 1) * (-2*h[0]);
                        p_i = (p_i-(p_i%2))/2;

                        for (j_config=0; j_config<dim_L; j_config++)
                        {
                            int bc_j = 3+2*(BC[j_config]!=1);
                            delta_E_IM += (double)(((p_i%bc_j)-1-(BC[j_config]!=1)) * (2-(BC[j_config]!=1))) * (-2*J[j_config]);
                            p_i = (p_i-(p_i%bc_j))/bc_j;
                        }

                        double prob=0.0;
                        if (delta_E_IM > 0) { 
                            if (T==0) { prob = 0.0; }
                            else { double p=exp(-delta_E_IM/T); prob = p/(1+p); }
                        }
                        else if (delta_E_IM < 0) {
                            if (T==0) { prob = 1.0; }
                            else { double p=exp(delta_E_IM/T); prob = 1/(1+p); }
                        }
                        for (t_i=0; t_i<num_of_threads; t_i++) { exp_Si_Sj[t_i][i] = prob; }
                    }
                    break;
                }
            }
            // memset(first_call, 0, num_of_threads);
            return 0;
        }
        
        return 1.0;
    }

    double update_probability_Glauber(long int xyzi)
    {
        #ifdef C_IM
            double update_prob = activation_probability_Glauber(xyzi, 0);
            // r = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()])/ (double) RAND_MAX;
            double r = genrand64_real1(thread_num_if_parallel());
            
            if(r < update_prob)
            {
                spin[xyzi] = -spin[xyzi];
                
                return 1.0;
            }
        #else
            double update_prob;
            double spin_local[dim_S];
            double field_local[dim_S];
            double E_old = Energy_old(xyzi, spin_local, field_local);
            double E_new = Energy_new(xyzi, spin_local, field_local);
            if (T == 0) {
                if (E_new - E_old > 0) { update_prob = 0.0; }
                else if (E_new - E_old < 0) { update_prob = 1.0; }
                else { update_prob = 0.5; }
            }
            else { update_prob = 1/(1+exp((E_new-E_old)/T)); }
            // double r = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()])/ (double) RAND_MAX;
            double r = genrand64_real1(thread_num_if_parallel());
            
            if(r < update_prob)
            {
                update_spin_single(xyzi, spin_local);
                return 1.0;
            }
        #endif
        return 0.0;
    }

    int linear_Glauber_sweep(long int iter)
    {
        long int site_i;
        do
        {
            for(site_i=0; site_i<no_of_sites; site_i++)
            {

                double update_prob = update_probability_Glauber(site_i);

                // // double r = (double) rand_r(&random_seed[cache_size*thread_num_if_parallel()])/ (double) RAND_MAX;
                // double r = genrand64_real1(thread_num_if_parallel());
                // if(r < update_prob)
                // {
                //     update_spin_single(site_i, 1);
                // }
                // else
                // {
                //     update_spin_single(site_i, 0);
                // }
            }
            iter--;
        }
        while (iter > 0);
        return 0;
    }

    int random_Glauber_sweep(long int iter)
    {
        // edit this
        #ifdef _OPENMP
            #ifdef PARALLEL_RANDOM_MC_SWEEP
            long int temp_i, t_i;
            int j_L;
            num_of_threads = omp_get_max_threads();
            while (iter>0)
            {
                set_cluster_s(0);
                for (t_i=0; t_i<num_of_threads; t_i++)
                {
                    do
                    {
                        // temp_i = rand_r(&random_seed[cache_size*thread_num_if_parallel()])%no_of_sites;
                        temp_i = genrand64_int64(thread_num_if_parallel())%no_of_sites;
                    } while (cluster[temp_i]==1);
                    iter--;
                    random_sites[t_i] = temp_i;
                }
                #pragma omp parallel for
                for (t_i=0; t_i<num_of_threads+(int)(iter<0)*iter; t_i++)
                {
                    update_probability_Glauber(random_sites[t_i]);
                }
            }
            #else
            // long int *random_sites = (long int *)malloc(no_of_black_white_sites[0],sizeof(long int));
            /* int *cluster */// long int *blocked_sites = (long int *)calloc((dim_L*2)*no_of_black_white_sites[0],sizeof(long int));

            int j_L;
            
            // long int temp_i = rand_r(&random_seed[cache_size*thread_num_if_parallel()])%no_of_sites;
            long int temp_i = genrand64_real1(thread_num_if_parallel())%no_of_sites;
    
            while (iter)
            {
                long int batch size = 0;
                set_cluster_s(0);
                while (iter>0 && cluster[temp_i] == 0)
                {
                    random_sites[batch_size] = temp_i;
                    cluster[temp_i] = 1;
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        for (k_L=0; k_L<2; k_L++)
                        {
                            cluster[N_N_I[2*dim_L*temp_i + 2*j_L + k_L];] = 1;
                        }
                    }
                    iter--;
                    batch_size++;
                    // temp_i = rand_r(&random_seed[cache_size*thread_num_if_parallel()])%no_of_sites;
                    temp_i = genrand64_int64(thread_num_if_parallel())%no_of_sites;
                    
                }
                #pragma omp parallel for
                for(i_b=0; i_b<batch_size; i_b++)
                {
                    update_probability_Glauber(random_sites[i_b]);
                }
            }
            // free(random_sites);
            #endif

        #else

        long int xyzi;
        do
        {
            // xyzi = rand_r(&random_seed[cache_size*thread_num_if_parallel()])%no_of_sites;
            xyzi = genrand64_int64(thread_num_if_parallel())%no_of_sites;
            
            double update_prob = update_probability_Glauber(xyzi);

            iter--;
        }
        while (iter > 0);
        
        #endif

        return 0;
    }

    int checkerboard_Glauber_sweep(long int iter)
    {
        // THIS PART IS PARALLELIZABLE. *modifications may be needed
        static int black_or_white = BLACK_WHITE;
        long int i;
        while(iter > 0)
        {
            #pragma omp parallel for 
            for (i=0; i < no_of_black_white_sites[black_or_white]; i++)
            {
                long int site_index = black_white_checkerboard[no_of_black_white_sites[black_or_white]*black_or_white + i];

                double update_prob = update_probability_Glauber(site_index);
            }

            black_or_white = !black_or_white;
            iter--;
        } 
        
        return 0;
    }

//====================  MonteCarlo-Sweep              ====================//

    int Monte_Carlo_Sweep(long int sweeps, int MC_algo_type_local, int MC_update_type_local)
    {
        if (MC_algo_type_local == 0)
        {
            if (MC_update_type_local == 0)
            {
                checkerboard_Glauber_sweep(2*sweeps);
            }
            else
            {
                if (MC_update_type_local == 1)
                {
                    random_Glauber_sweep(no_of_sites*sweeps);
                }
                else
                {
                    linear_Glauber_sweep(sweeps);
                }
            }
        }
        else if (MC_algo_type_local == 1)
        {
            if (MC_update_type_local == 0)
            {
                checkerboard_Metropolis_sweep(2*sweeps);
            }
            else
            {
                if (MC_update_type_local == 1)
                {
                    random_Metropolis_sweep(no_of_sites*sweeps);
                }
                else
                {
                    linear_Metropolis_sweep(sweeps);
                }
            }
        }
        else
        {
            if (MC_update_type_local == 0)
            {
                // #ifdef SWENDSEN_WANG
                full_Wolff_sweep(sweeps);
                // #endif
            }
            else
            {
                // #ifndef SWENDSEN_WANG
                random_Wolff_sweep(sweeps);
                // #endif
            }
        }
        return 0;
    }

//====================  T!=0                          ====================//

    int print_abs_m()
    {
        int j_S;

        ensemble_abs_m();
        
        printf("\n\n<|s_i|> = { %lf", abs_m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", abs_m[j_S]);
        }
        printf("}\n");

        return 0;
    }

    int thermalizing_iteration(long int thermal_iter, int MC_algo_type_local, int MC_update_type_local, int reqd_to_print, int init)
    {
        #ifdef C_IM
        activation_probability_Metropolis(0, init);
        activation_probability_Glauber(0, init);
        activation_probability_Wolff(0, 0, 0, init);
        #endif
        
        // printf("\n--------\n");
        // printf("%ld,%d,%d", thermal_iter, MC_algo_type_local, MC_update_type_local);
        // printf("\n--------\n");
        int j_S;
        if (reqd_to_print == 1)
        {
            printf("Thermalizing.. ");
            fflush(stdout);
        }

        Monte_Carlo_Sweep(thermal_iter, MC_algo_type_local, MC_update_type_local);
        
        if (reqd_to_print == 1)
        {
            // ensemble_all();
            ensemble_E();
            ensemble_m();
            printf(" m=(%.3f", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%.3f", m[j_S]);
            }
            printf("), E=%.3f. ", E);

            printf("Done. ");
            fflush(stdout);
        }

        return 0;
    }

    int averaging_iteration(long int average_iter, long int sampl_inter, int MC_algo_type_local, int MC_update_type_local, int reqd_to_print)
    {
        // printf("\n--------\n");
        // printf("%ld,%d,%d", average_iter, MC_algo_type_local, MC_update_type_local);
        // printf("\n--------\n");
        double MCS_counter = 0;
        int j_S, j_SS, j_L;
        
        set_sum_of_moment_all_0();
        // set_sum_of_moment_m_0();
        // set_sum_of_moment_m_higher_0();
        // set_sum_of_moment_m_vec_0();
        // set_sum_of_moment_m_abs_0();
        // set_sum_of_moment_E_0();
        // set_sum_of_moment_Y_ab_mu_0();

        // printf("Averaging iterations... h=%lf", h[0]);
        if (reqd_to_print == 1)
        {
            printf("Averaging.. ");
            fflush(stdout);
            // for (j_S=0; j_S<dim_S; j_S++)
            // {
            //     printf("%lf,", h[j_S]);
            // }
            // printf("\n");
        }

        while(average_iter)
        {
            if (reqd_to_print == 1)
            {
                printf(" Iteration = %ld ", average_iter );
                fflush(stdout);
            }
            Monte_Carlo_Sweep(sampl_inter-genrand64_int64(thread_num_if_parallel())%sampl_inter, MC_algo_type_local, MC_update_type_local);

            // random_Wolff_sweep(1);
            
            ensemble_all();
            // ensemble_m();
            // ensemble_m_vec_abs();
            // ensemble_B();
            // ensemble_E();
            // ensemble_Y_ab_mu();

            sum_of_moment_all();
            // sum_of_moment_m();
            // sum_of_moment_m_higher();
            // sum_of_moment_m_vec();
            // sum_of_moment_B();
            // sum_of_moment_m_abs();
            // sum_of_moment_E();
            // sum_of_moment_Y_ab_mu();
            MCS_counter = MCS_counter + 1;
            
            
            printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"       );
            // printf(           "\b\b\b\b\b");
            long int i_iter = average_iter/10;
            while (i_iter>0)
            {
                printf("\b");
                i_iter = i_iter/10;
            }
            fflush(stdout);
            average_iter = average_iter - 1;
        }

        average_of_moment_all(MCS_counter);
        // average_of_moment_m(MCS_counter);
        // average_of_moment_m_higher(MCS_counter);
        // average_of_moment_m_vec(MCS_counter);
        // average_of_moment_B(MCS_counter);
        // average_of_moment_m_abs(MCS_counter);
        // average_of_moment_E(MCS_counter);
        // average_of_moment_Y_ab_mu(MCS_counter);
        if (reqd_to_print == 1)
        {
            printf(" <m>=(%.3f", m_avg[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%.3f", m_avg[j_S]);
            }
            printf("), <E>=%.3f. ", E_avg);
            // printf(", Binder = %.3f \n", B);
            /* 
            for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        printf("Y[%d,%d][%d]=%.3f \n", j_S, j_SS, j_L, Y_ab_mu[dim_S*dim_S*j_L + dim_S*j_S + j_SS]);
                    }
                }
            } */

            // printf("<M>=%.3f, <|M|>=%.3f, <E>=%.3f, Cv=%.3f, X=%.3f, X_abs=%.3f, B=%.3f. ", m_avg, m_abs_avg, E_avg, Cv, X, X_abs, B);
            printf("Done.\n");
            fflush(stdout);
        }
        return 0;
    }

    int evolution_at_T(int ini_order, long int repeat_for_same_T)
    {
        printf("\n__________________________________________________________\n");
        double start_time_local = get_time_if_parallel();
        // repeat with different initial configurations
        int j_S, j_SS, j_L;
        //
            long int thermal_i_temp = thermal_i;
            long int average_j_temp = average_j;
            long int sampling_inter_temp = sampling_inter;
            int MC_algo_type_temp = MC_algo_type;
            int MC_update_type_temp = MC_update_type;
            double T_temp = T;
            double h_temp[dim_S];
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h_temp[j_S] = h[j_S];
            }

        // thermal_i = 50;
        // average_j = 50;
        // sampling_inter = 2;
        // MC_algo_type = 1; // Metropolis
        // MC_update_type = 0; // Checkerboard
        // int MC_algo_type_avg = 2; // Wolff
        // int MC_update_type_avg = 0; // irrelevant

        printf("\norder = ({%lf", order[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", order[j_S]);
        }
        printf("}, %d, %d)\n", h_order, r_order);
        
        if (ini_order==0)
        {
            h_order = 0;
            r_order = 0;
            initialize_spin_config();
        }
        else
        {
            if (ini_order==1)
            {
                h_order = 1;
                r_order = 0;
                initialize_spin_config();
            }
            else
            {
                if (ini_order==2)
                {
                    h_order = 0;
                    r_order = 1;
                    initialize_spin_config();
                }
                else
                {
                    if (ini_order = 3)
                    {
                        load_spin_config("_chkpt");
                    }
                }
            }
        }
        // ensemble_all();
        ensemble_E();
        ensemble_m();

        printf("Initial Magnetization = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);
        /* // set output_variable_name=1 here
        {
            output_prepend = 1;
            output_m_avg = 1;
            // output_m_abs_avg = 1;
            // output_E_avg = 1;
            output_append = 1;
        } */

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_evo(t)_T%.3f_%c%c_%c%c_", dim_S, dim_L, T, G_M_W[MC_algo_type_thrm], C_R_L[MC_update_type_thrm], G_M_W[MC_algo_type_avg], C_R_L[MC_update_type_avg]);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            // pos += sprintf(pos, "_(%lf,%lf)", Temp_min, Temp_max);
            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");    
            #ifdef RANDOM_BOND
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", h[j_S]);
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld_%ld.dat", h_order, r_order, thermal_i, average_j);
        }

        // column labels and parameters
        output_param_file(output_file_0);
        output_label(output_file_0, "Step\t", "Saved_Spin\t");

        long int i;
        
        thermalizing_iteration(0, 0, 0, 0, 1);
        for (i=0; i<repeat_for_same_T; i++)
        {
            
            thermalizing_iteration(thermal_i, MC_algo_type_thrm, MC_update_type_thrm, 0, 0);
            averaging_iteration(average_j, sampling_inter, MC_algo_type_avg, MC_update_type_avg, 0);
            printf("\r(%ld) m=%2.6f, <m>=%2.6f [t=%2.3e s]    ", i, m[0], m_avg[0], (double)get_time_if_parallel()-start_time_local);
            fflush(stdout);
            char str_prep[128];
            char *pos_prep = str_prep;
            pos_prep += sprintf(pos_prep, "%ld\t", i);
            #ifdef SAVE_SPIN_AFTER
            if (i%SAVE_SPIN_AFTER==0 /* && fabs(m[0])<0.5 */ )
            {
                char append_string[128];
                sprintf(append_string, "_i%ld", i/SAVE_SPIN_AFTER);
                save_spin_config(append_string, "a", 2);
                output_data(output_file_0, str_prep, "Yes");
            }
            else
            {
                output_data(output_file_0, str_prep, "No");
            }
            #else
                output_data(output_file_0, str_prep, "No");
            #endif
            
        }
        
        // reset_output_variable_name_0();

            thermal_i = thermal_i_temp;
            average_j = average_j_temp;
            sampling_inter = sampling_inter_temp;
            MC_algo_type = MC_algo_type_temp;
            MC_update_type = MC_update_type_temp;
            T = T_temp;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_temp[j_S];
            }
        
        printf("\n__________________________________________________________\n");
        save_spin_config("_chkpt", "w", 1);
        return 0;
    }

    int initialize_spin_and_evolve_at_T(int MC_algo_type_th, int MC_update_type_th, int MC_algo_type_av, int MC_update_type_av)
    {
        // repeat with different initial configurations
        int j_S, j_SS, j_L;
        double start_time_local = get_time_if_parallel();
        initialize_spin_config();
        // Monte_Carlo_Sweep(/* sweeps */2, /* MC_algo_type_local */2, /* MC_update_type_local */0);
        // printf("[t=%lf s] ", (double)get_time_if_parallel()-start_time_local);
        printf("\norder = ({%lf", order[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", order[j_S]);
        }
        printf("}, %d, %d)\n", h_order, r_order);

        // ensemble_all();
        ensemble_E();
        ensemble_m();

        printf("Initial Magnetization = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);

        thermalizing_iteration(thermal_i, MC_algo_type_th, MC_update_type_th, 1, 1);
        // Monte_Carlo_Sweep(/* sweeps */2, /* MC_algo_type_local */2, /* MC_update_type_local */0);
        char append_string1[128];
        char *pos_append_string1 = append_string1;
        pos_append_string1 += sprintf(pos_append_string1, "_r%d%da", h_order, r_order);
        save_spin_config(append_string1, "a", 2);
        printf("[t=%lf s] ", (double)get_time_if_parallel()-start_time_local);

        averaging_iteration(average_j, sampling_inter, MC_algo_type_av, MC_update_type_av, 1);
        char append_string2[128];
        char *pos_append_string2 = append_string2;
        pos_append_string2 += sprintf(pos_append_string2, "_r%d%db", h_order, r_order);
        save_spin_config(append_string2, "a", 2);
        printf("[t=%lf s] ", (double)get_time_if_parallel()-start_time_local);
        
        printf("------------------------\n");

        return 0;
    }

    int evo_diff_ini_config_temp(int ini_order)
    {
        int j_S, j_L, j_SS;
        //
            long int thermal_i_temp = thermal_i;
            long int average_j_temp = average_j;
            long int sampling_inter_temp = sampling_inter;
            int MC_algo_type_temp = MC_algo_type;
            int MC_update_type_temp = MC_update_type;
            double T_temp = T;
            double h_temp[dim_S];
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h_temp[j_S] = h[j_S];
            }
            int r_order_temp = r_order;
            int h_order_temp = h_order;
        
        // thermal_i = 1024;//1024*10;//thermal_i*lattice_size[0];
        // average_j = 1024;//1024;//average_j*lattice_size[0];
        // sampling_inter = 1;
        // MC_algo_type = 2; // 0-Glauber // 1-Metropolis // 2-Wolff
        // MC_update_type = 0; // 0-Checkerboard // 1-Random // 2-Linear // Wolff-irrelevant
        // int MC_algo_type_avg = 2; // 0-Glauber // 1-Metropolis // 2-Wolff
        // int MC_update_type_avg = 0; // 0-Checkerboard // 1-Random // 2-Linear // Wolff-irrelevant
    
        if (ini_order==0)
        {
            h_order = 0;
            r_order = 0;
        }
        else
        {
            if (ini_order==1)
            {
                h_order = 1;
                r_order = 0;
            }
            else
            {
                if (ini_order==2)
                {
                    h_order = 0;
                    r_order = 1;
                }
            }
        }
        

        #ifdef RANDOM_FIELD
        printf("%lf", sigma_h[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", sigma_h[j_S]);
        }
        printf("\n<delta_h> = %lf \n", h_dev_avg[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", h_dev_avg[j_S]);
        }

        printf("\n");
        #endif

        /* // set output_variable_name=1 here
        {
            // output_h = 1;
            output_T = 1;
            output_m = 1;
            output_m_avg = 1;
            // output_m_abs_avg = 1;
            // output_m_2_avg = 1;
            // output_m_4_avg = 1;

            // output_m_2_vec_avg = 1;
            // output_m_4_vec_avg = 1;
            // output_m_ab_avg = 1;
            
            // output_X = 1;
            
            // output_X_ab = 1;
            
            // output_B = 1;
            
            // output_B_a = 1;
            // output_E_avg = 1;
            // output_E_2_avg = 1;
            // output_Cv = 1;
            // output_Y_1_avg = 1;
            // output_Y_2_avg = 1;
            // output_Y_x_mu = 1;
        } */

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_eq_%c%c_%c%c_", dim_S, dim_L, G_M_W[MC_algo_type_thrm], C_R_L[MC_update_type_thrm], G_M_W[MC_algo_type_avg], C_R_L[MC_update_type_avg]);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_(%.3f,%.3f)", Temp_min, Temp_max);
            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_BOND
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", h[j_S]);
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld_%ld.dat", h_order, r_order, thermal_i, average_j);
        }
        // column labels and parameters
        output_param_file(output_file_0);
        output_label(output_file_0, "", "");
        
        for (T=Temp_max; T>=Temp_min; T=T-delta_T)
        {
            printf("\nT=%lf\t ", T);
            initialize_spin_and_evolve_at_T(MC_algo_type_thrm, MC_update_type_thrm, MC_algo_type_avg, MC_update_type_avg);

            output_data(output_file_0, "", "");
        }
        
        // reset_output_variable_name_0();

            thermal_i = thermal_i_temp;
            average_j = average_j_temp;
            sampling_inter = sampling_inter_temp;
            MC_algo_type = MC_algo_type_temp;
            MC_update_type = MC_update_type_temp;
            T = T_temp;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_temp[j_S];
            }
            r_order = r_order_temp;
            h_order = h_order_temp;

        return 0;
    }

    int cooling_heating(char output_file_name[], int cool_heat)
    {
        #ifdef _OPENMP
        #ifdef BUNDLE
            omp_set_num_threads(BUNDLE);
        #else
            omp_set_num_threads(num_of_threads);
        #endif
        #endif
        /* #ifdef _OPENMP
            if (num_of_threads<=16)
            {
                omp_set_num_threads(num_of_threads);
            }
            else 
            {
                if (num_of_threads<=24)
                {
                    omp_set_num_threads(16);
                }
                else
                {
                    omp_set_num_threads(num_of_threads-8);
                }
            }
        #endif */

        int j_S, j_SS, j_L;

        // ensemble_all();
        ensemble_E();
        ensemble_m();

        printf("Initial Magnetization = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);
        if (cool_heat%2==1)
        {
            printf("Heating... ");
        }
        else
        {
            printf("Cooling... ");
        }
        
        pFile_1 = fopen(output_file_name, "a");
        if (cool_heat%2==1)
        {
            fprintf(pFile_1, "Heating... \n");
        }
        else
        {
            fprintf(pFile_1, "Cooling... \n");
        }
        fclose(pFile_1);

        double Temp_start;
        double Temp_end;
        double del_T;
        if (cool_heat%2==1)
        {
            Temp_start = Temp_min;
            Temp_end = Temp_max;
            del_T = delta_T;
        }
        else
        {
            Temp_start = Temp_max;
            Temp_end = Temp_min;
            del_T = -delta_T;
        }

        for (T=Temp_start; T*del_T<=Temp_end*del_T; T=T+del_T)
        {
            if (T==Temp_start) {printf("\nT = %lf ", T);}
            else {printf("--> %lf ", T);}
            fflush(stdout);
            
            thermalizing_iteration(thermal_i, MC_algo_type_thrm, MC_update_type_thrm, 0, 1);
            
            char append_string1[128];
            char *pos_append_string1 = append_string1;
            if (cool_heat%2==1)
            {
                pos_append_string1 += sprintf(pos_append_string1, "_heat_cycle_%da", (int)(cool_heat/2));
            }
            else
            {
                pos_append_string1 += sprintf(pos_append_string1, "_cool_cycle_%da", (int)(cool_heat/2));
            }
            save_spin_config(append_string1, "a", 2);

            averaging_iteration(average_j, sampling_inter, MC_algo_type_avg, MC_update_type_avg, 0);

            output_data(output_file_name, "", "");

            char append_string2[128];
            char *pos_append_string2 = append_string2;
            if (cool_heat%2==1)
            {
                pos_append_string2 += sprintf(pos_append_string2, "_heat_cycle_%db", (int)(cool_heat/2));
            }
            else
            {
                pos_append_string2 += sprintf(pos_append_string2, "_cool_cycle_%db", (int)(cool_heat/2));
            }
            save_spin_config(append_string2, "a", 2);
        }

        // ensemble_all();
        ensemble_E();
        ensemble_m();

        printf("\n");
        printf("Final Magnetization = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);
        printf("------------------------\n");

        return 0;
    }

    int fc_fh_or_both(int c_h_ch_hc, int zero_or_finite)
    {
        int j_S, j_SS, j_L;
        //
            long int hysteresis_repeat_temp = hysteresis_repeat;
            long int thermal_i_temp = thermal_i;
            long int average_j_temp = average_j;
            long int sampling_inter_temp = sampling_inter;
            int MC_algo_type_temp = MC_algo_type;
            int MC_update_type_temp = MC_update_type;
            double T_temp = T;
            double h_temp[dim_S];
            if (zero_or_finite == 0)
            {
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    h_temp[j_S] = h[j_S];
                }
            }
        // hysteresis_repeat = 2;
        // thermal_i = 1000;
        // average_j = 100;
        // sampling_inter = 16;
        // MC_algo_type = 1; // Metropolis
        // MC_update_type = 1; // 0 - Checkerboard, 1 - Random, 2 - Linear

        if (zero_or_finite == 0)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = 0;
            }
        }

        if (c_h_ch_hc == 0 || c_h_ch_hc == 2)
        {
            T = Temp_max;
        }
        if (c_h_ch_hc == 1 || c_h_ch_hc == 3)
        {
            T = Temp_min;
        }

        printf("\nRepeat = %d\n", hysteresis_repeat); 
        printf("\n__________________________________________________________\n");

        printf("\nField Cooling/Heating with (thermalizing steps+averaging steps)/delta_T=(%ld+%ld)/%lf at h={%lf", thermal_i, average_j, delta_T, h[0]);
        for (j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", h[j_S]);
        }
        printf("}..\n");
        
        /* // set output_variable_name=1 here
        {
            // output_h = 1;
            output_T = 1;
            output_m_avg = 1;
            output_m_abs_avg = 1;
            output_m_2_avg = 1;
            output_m_4_avg = 1;
            // output_m_2_vec_avg = 1;
            // output_m_4_vec_avg = 1;
            // output_m_ab_avg = 1;
            output_X = 1;
            // output_X_ab = 1;
            output_B = 1;
            // output_B_a = 1;
            // output_E_avg = 1;
            // output_E_2_avg = 1;
            // output_Cv = 1;
            // output_Y_1_avg = 1;
            // output_Y_2_avg = 1;
            // output_Y_x_mu = 1;
        } */
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_FC-FH_%c%c_%c%c_", dim_S, dim_L, G_M_W[MC_algo_type_thrm], C_R_L[MC_update_type_thrm], G_M_W[MC_algo_type_avg], C_R_L[MC_update_type_avg]);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_(%.3f-%.3f)-[%.3f]", Temp_min, Temp_max, delta_T);

            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");
            
            #ifdef RANDOM_BOND
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            } 
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%f", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", h[j_S]);
            }
            pos += sprintf(pos, "}"); 
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld_%ld.dat", h_order, r_order, thermal_i, average_j);
        }
        // column labels and parameters
        output_param_file(output_file_0);
        output_label(output_file_0, "", "");
        
        printf("\n");
        initialize_spin_config();
        printf("\n");
        int i;
        for (i=0; i<hysteresis_repeat; i++)
        {
            if (c_h_ch_hc == 0 || c_h_ch_hc == 2)
            {
                cooling_heating(output_file_0, 0+2*i);
            }
            if (c_h_ch_hc == 1 || c_h_ch_hc == 2 || c_h_ch_hc == 3)
            {
                cooling_heating(output_file_0, 1+2*i);
            }
            if (c_h_ch_hc == 3)
            {
                cooling_heating(output_file_0, 0+2*i);
            }
        }
        
        // reset_output_variable_name_0();
            hysteresis_repeat = hysteresis_repeat_temp;
            thermal_i = thermal_i_temp;
            average_j = average_j_temp;
            sampling_inter = sampling_inter_temp;
            MC_algo_type = MC_algo_type_temp;
            MC_update_type = MC_update_type_temp;
            T = T_temp;
            if (zero_or_finite == 0)
            {
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    h[j_S] = h_temp[j_S];
                }
            }

        printf("\n__________________________________________________________\n");
        return 0;
    }

    int hysteresis_protocol(int jj_S, double order_start)
    {
        printf("\nhysteresis_MCS = %ld, Repeat = %d \n", hysteresis_MCS, hysteresis_repeat); 
        printf("\n__________________________________________________________\n");
        int j_S, j_L;
        //
            long int thermal_i_temp = thermal_i;
            long int average_j_temp = average_j;
            long int sampling_inter_temp = sampling_inter;
            int MC_algo_type_temp = MC_algo_type;
            int MC_update_type_temp = MC_update_type;
            double T_temp = T;
            double h_temp[dim_S];
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h_temp[j_S] = h[j_S];
            }

        // thermal_i = 0;
        // average_j = hysteresis_MCS;
        // sampling_inter = 1;
        // MC_algo_type = 0; // Glauber
        // MC_update_type = 0; // Checkerboard

        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[jj_S] = order_start;
            }
            else
            {
                order[j_S] = 0;
            }
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = 0;
        }
        double h_start = order[jj_S]*(h_max+h_i_max);
        if (fabs(h_i_max) < fabs(h_i_min))
        {
            h_start = order[jj_S]*(h_max+fabs(h_i_min));
        }
        else
        {
            h_start = order[jj_S]*(h_max+fabs(h_i_max));
        }
        double h_end = -h_start;
        double delta_h = del_h;
        
        h[jj_S] = h_start;
        // delta_h = (2*order[jj_S]-1)*delta_h;
        h_order = 0;
        r_order = 0;
        initialize_spin_config();
        ensemble_m();
        ensemble_E();
        printf("\nm = %lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("\norder = ({%lf", order[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", order[j_S]);
        }
        printf("}, %d, %d)\n", h_order, r_order);

        printf("Hysteresis looping %d-times at T=%lf.. with MCS/field = %ld \n", hysteresis_repeat, T, hysteresis_MCS);
        
        /* // set output_variable_name=1 here
        {
            output_h = 1;
            output_m_avg = 1;
            // output_m_abs_avg = 1;
            output_E_avg = 1;
        } */
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hysteresis_%c%c_", dim_S, dim_L, G_M_W[MC_update_type_avg], C_R_L[MC_update_type_avg]);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");    
            #ifdef RANDOM_BOND
            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); 
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.2f,%.2f)-[%.4f]", h_start, h_end, delta_h);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld.dat", h_order, r_order, hysteresis_MCS);
        }
        // column labels and parameters
        output_param_file(output_file_0);
        output_label(output_file_0, "", "");
        
        int i;
        printf("\nh = %lf(0)", h_start);
        for (i=0; i<hysteresis_repeat; i=i+1)
        {
            printf(" --> %lf", h_end);
            // printf("h = %lf --> %lf\t ", h_start, h_end);
            fflush(stdout);
            for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
            {
                thermalizing_iteration(/* thermal_i */0, /* MC_algo_type_thrm */0, /* MC_update_type_thrm */0, 0, 1);
                averaging_iteration(average_j, sampling_inter, MC_algo_type_avg, MC_update_type_avg, 0);

                output_data(output_file_0, "", "");
            }
            printf("(%d) --> %lf", i+1, h_start);
            // printf("..(%d) h = %lf <-- %lf\t ", i+1, h_start, h_end);
            fflush(stdout);
            for (h[jj_S] = h_end; order[jj_S] * h[jj_S] <= order[jj_S] * h_start; h[jj_S] = h[jj_S] + order[jj_S] * delta_h)
            {
                averaging_iteration(average_j, sampling_inter, MC_algo_type, MC_update_type, 0);

                output_data(output_file_0, "", "");
            }
            printf("(%d) ", i+1);
            fflush(stdout);
            h[jj_S] = 0;
        }
        
        // delta_h = (2*order[jj_S]-1)*delta_h;
        
        // reset_output_variable_name_0();

            thermal_i = thermal_i_temp;
            average_j = average_j_temp;
            sampling_inter = sampling_inter_temp;
            MC_algo_type = MC_algo_type_temp;
            MC_update_type = MC_update_type_temp;
            T = T_temp;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_temp[j_S];
            }
        
        printf("\nFinished. \n");
        printf("\n__________________________________________________________\n");
        return 0;
    }

//====================  backup-restore                ====================//

    #ifdef enable_CUDA_CODE
        __global__ void backing_up_spin_on_device(long int sites)
        {
            int index = threadIdx.x + blockIdx.x*blockDim.x;
            if (index < sites*dim_S)
            {
                dev_spin_bkp[index] = dev_spin[index];
            }
            return; 
        }
        
        __global__ void backing_up_m_on_device()
        {
            int index_j_S = threadIdx.x + blockIdx.x*blockDim.x;
            if (index_j_S < dim_S)
            {
                dev_m_bkp[index_j_S] = dev_m[index_j_S];
            }
            return; 
        }
    #endif

    int backing_up_spin()
    {
        int i, j_S;
        
        #ifdef enable_CUDA_CODE
        backing_up_spin_on_device<<< dim_S*no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites);
        // backing_up_spin_on_device<<< 1, dim_S*no_of_sites >>>(no_of_sites);
        backing_up_m_on_device<<< 1, dim_S >>>();
        /* backing_up_E_on_device<<< 1, 1 >>>(); */
        #else
        #pragma omp parallel for
        for(i=0; i<no_of_sites*dim_S; i++)
        {
            spin_bkp[i] = spin[i];
        }

        for(j_S=0; j_S<dim_S; j_S++)
        {
            m_bkp[j_S] = m[j_S];
        }
        #ifdef EXPLORE_ENERGY_LANDSCAPE
        E_bkp = E;
        #endif
        #endif
        return 0;
    }

    #ifdef enable_CUDA_CODE
        __global__ void restoring_spin_on_device(long int sites)
        {
            int index = threadIdx.x + blockIdx.x*blockDim.x;
            if (index < sites*dim_S)
            {
                dev_spin[index] = dev_spin_bkp[index];
            }
            return; 
        }
        
        __global__ void restoring_m_on_device()
        {
            int index_j_S = threadIdx.x + blockIdx.x*blockDim.x;
            if (index_j_S < dim_S)
            {
                dev_m[index_j_S] = dev_m_bkp[index_j_S];
            }
            return; 
        }
    #endif

    int restoring_spin(double h_text, double delta_text, int jj_S, double delta_m, char text[], int reqd_to_print)
    {
        int i, j_S;

        #ifdef enable_CUDA_CODE
        restoring_spin_on_device<<< dim_S*no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites);
        // restoring_spin_on_device<<< 1, dim_S*no_of_sites >>>(no_of_sites);
        restoring_m_on_device<<< 1, dim_S >>>();
        /* restoring_E_on_device<<< 1, 1 >>>(); */
        #else
        #pragma omp parallel for
        for(i=0; i<no_of_sites*dim_S; i++)
        {
            spin[i] = spin_bkp[i];
        }

        for(j_S=0; j_S<dim_S; j_S++)
        {
            m[j_S] = m_bkp[j_S];
        }
        #ifdef EXPLORE_ENERGY_LANDSCAPE
        E = E_bkp;
        #endif
        #endif
        #ifdef PRINT_OUTPUT
        // if(reqd_to_print == 1)
        // {
            // printf(  "\n============================\n");
            printf(  "\r=1= %s = %.3e ", text, h_text );
            #ifndef CHECK_AVALANCHE
            printf(    ", d_m = %.3e ", delta_m );
            #else
            printf(    ", d_S = %.3e ", delta_m );
            #endif
            #ifdef EXPLORE_ENERGY_LANDSCAPE
            printf(    ", d_E = %.3e ", delta_E );
            #endif
            printf(    ", d_%s = %.3e ", text, order[jj_S]*delta_text );
            printf(    " -[ restore ]- ");
            printf(    " Time = %.3e s | ", get_time_if_parallel() - start_time );
            fflush(stdout);
            // printf(  "\n============================\n");
        // }
        #endif

        return 0;
    }

//====================  RFIM ZTNE                     ====================//
    
    // Merges two subarrays of arr[]. 
    // First subarray is arr[l..m] 
    // Second subarray is arr[m+1..r] 
    void merge(double arr[], long int l, long int m, long int r) 
    { 
        long int i, j, k; 
        long int n1 = m - l + 1; 
        long int n2 = r - m; 

        /* create temp arrays */
        double L[n1], R[n2]; 

        /* Copy data to temp arrays L[] and R[] */
        for (i = 0; i < n1; i++) 
            L[i] = arr[l + i]; 
        for (j = 0; j < n2; j++) 
            R[j] = arr[m + 1+ j]; 

        /* Merge the temp arrays back into arr[l..r]*/
        i = 0; // Initial index of first subarray 
        j = 0; // Initial index of second subarray 
        k = l; // Initial index of merged subarray 
        while (i < n1 && j < n2) 
        { 
            if (L[i] <= R[j]) 
            { 
                arr[k] = L[i]; 
                i++; 
            } 
            else
            { 
                arr[k] = R[j]; 
                j++; 
            } 
            k++; 
        } 

        /* Copy the remaining elements of L[], if there 
        are any */
        while (i < n1) 
        { 
            arr[k] = L[i]; 
            i++; 
            k++; 
        } 

        /* Copy the remaining elements of R[], if there 
        are any */
        while (j < n2) 
        { 
            arr[k] = R[j]; 
            j++; 
            k++; 
        } 
    } 

    /* l is for left index and r is right index of the 
    sub-array of arr to be sorted */
    void mergeSort(double arr[], long int l, long int r) 
    { 
        if (l < r) 
        { 
            // Same as (l+r)/2, but avoids overflow for 
            // large l and h 
            long int m = l+(r-l)/2; 

            // Sort first and second halves 
            mergeSort(arr, l, m); 
            mergeSort(arr, m+1, r); 

            merge(arr, l, m, r); 
        } 
    } 

    long int searchInsert(double nums[], long int ind[], double target, long int size_array) 
    {
        long int pivot, left = 0, right = size_array - 1;
        while (left <= right) 
        {
            pivot = left + (right - left) / 2;
            if (nums[ind[pivot]] == target) return pivot;
            if (target < nums[ind[pivot]]) right = pivot - 1;
            else left = pivot + 1;
        }
        return left;
    }

    long int* sort_h_index(double h_sort[], long int size_array)
    {
        long int *sorted_indices = (long int *)malloc(size_array*sizeof(long int));
        long int i;
        sorted_indices[0] = 0;
        for(i=1; i<no_of_sites; i++)
        {
            long int pivot = searchInsert(h_sort, sorted_indices, h_sort[i], i);
            long int i_p = pivot;
            while(i_p<i)
            {
                sorted_indices[i_p+1] = sorted_indices[i_p];
            }
            sorted_indices[pivot] = i;
        }
        return sorted_indices;
    }

    long int mark_unstables(double s, long int remaining_sites, long int *nucleation_sites)
    {
        // if (remaining_sites<=0) { return 0; } // remove
        // h[0] = 0;
        long int i_1 = 0;
        
        long int no_of_nuclei = 0;
        #pragma omp parallel for reduction(+:no_of_nuclei)
        for (i_1=0; i_1<no_of_sites; i_1++)
        {
            if (spin[i_1]==s)
            {
                double field_local[dim_S];
                double spin_local[dim_S];
                double E_1 = Energy_old(i_1, spin_local, field_local);
                if (E_1>=0)
                {
                    spin[i_1] = -s;
                    long int i_2;
                    #pragma omp atomic capture
                    {
                        i_2 = remaining_sites; remaining_sites--;
                    }
                    nucleation_sites[no_of_sites-i_2] = i_1;
                    no_of_nuclei += 1;
                }
            }
        }
        return no_of_nuclei;
    }

    long int mark_extremes(double s, long int remaining_sites, long int *nucleation_sites)
    {
        // if (remaining_sites<=0) { return 0; } // remove
        h[0] = 0;
        long int i_1 = 0;
        
        long int no_of_nuclei = 0;
        double E_2 = -1-(double)dim_L*2*(1+fabs(J_i_min)+fabs(J_i_max))-(fabs(h_i_min)+fabs(h_i_max));
        #pragma omp parallel for reduction(max:E_2)
        for (i_1=0; i_1<no_of_sites; i_1++)
        {
            if (spin[i_1]==s)
            {
                double field_local[dim_S];
                double spin_local[dim_S];
                double E_1 = Energy_old(i_1, spin_local, field_local);
                // E_2 = max(E_1,E_2);
                if (E_1>E_2)
                {
                    E_2 = E_1;
                }
            }
        }
        #pragma omp parallel for reduction(+:no_of_nuclei)
        for (i_1=0; i_1<no_of_sites; i_1++)
        {
            if (spin[i_1]==s)
            {
                double field_local[dim_S];
                double spin_local[dim_S];
                double E_1 = Energy_old(i_1, spin_local, field_local);
                if (E_1==E_2)
                {
                    long int i_2;
                    #pragma omp atomic capture
                    {
                        i_2 = remaining_sites; remaining_sites--;
                    }
                    nucleation_sites[no_of_sites-i_2] = i_1;
                    spin[i_1] = -s;
                    no_of_nuclei += 1;
                }
            }
        }
        h[0] = s*E_2;
        return no_of_nuclei;
    }
    
    long int flip_marked_parallel(double s, long int remaining_sites, long int *nucleation_sites, long int no_of_nuclei)
    {
        long int i_1 = no_of_sites-remaining_sites;
        long int i_2 = no_of_sites-remaining_sites+no_of_nuclei;
        long int i_3 = no_of_sites-remaining_sites+no_of_nuclei;
        long int i;

        while (i_1!=i_2)
        {
            #pragma omp parallel for
            for (i=i_1; i<i_2; i++)
            {
                int j_L,k_L;
                
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    for (k_L=0; k_L<2; k_L++)
                    {
                        long int next_site = N_N_I[2*dim_L*nucleation_sites[i] + 2*j_L + k_L];
                        if (next_site<no_of_sites)
                        {
                            #pragma omp critical
                            {
                                if (spin[next_site] != -s)
                                {
                                    double field_local[dim_S];
                                    double spin_local[dim_S];
                                    double E_1 = Energy_old(next_site, spin_local, field_local);
                                    if (E_1>=0)
                                    {
                                        spin[next_site] = -s;
                                        nucleation_sites[i_3] = next_site;
                                        i_3++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            i_1 = i_2;
            i_2 = i_3;
        }
        
        return no_of_sites-i_3;
    }

    int unflipp_sites(double s, long int remaining_sites, long int *nucleation_sites, long int remaining_sites_new)
    {
        long int i;
        #pragma omp parallel for
        for(i=remaining_sites_new; i<remaining_sites; i++)
        {
            spin[nucleation_sites[no_of_sites-1-i]] = s;
        }
        m[0] = m_bkp[0];
        return 0;
    }

    long int find_extremes(double s, long int remaining_sites, long int *nucleation_sites)
    {
        // if (remaining_sites<=0) { return 0; } // remove
        h[0] = 0;
        long int i_1 = 0;
        while (spin[dim_S*i_1 + 0] == -s)
        {
            i_1++;
        }
        nucleation_sites[0] = i_1;
        double field_local[dim_S];
        double spin_local[dim_S];
        double E_1 = Energy_old(i_1, spin_local, field_local);
        double E_2;
        remaining_sites--;
        long int i_2 = i_1;
        long int no_of_nuclei = 1;
        while (remaining_sites > 0)
        {
            i_2++;
            while (spin[dim_S*i_2 + 0] == -s)
            {
                i_2++;
            }
            E_2 = Energy_old(i_2, spin_local, field_local);
            remaining_sites--;
            if (E_1 < E_2)
            {
                nucleation_sites[0] = i_2;
                E_1 = E_2;
                no_of_nuclei = 1;
            }
            else
            {
                if (E_1 == E_2)
                {
                    nucleation_sites[no_of_nuclei] = i_2;
                    no_of_nuclei++;
                }
            }
        }
        h[0] = s*E_1;

        return no_of_nuclei;
    }

    long int flip_all_unstable(double s, long int remaining_sites)
    {
        // if (remaining_sites<=0) { return 0; } // remove
        
        long int i = 0;
        long int no_of_sites_flipped = 0;
        // static int black_or_white = BLACK_WHITE;

        do
        {
            no_of_sites_flipped = 0;
            #pragma omp parallel for reduction(+:no_of_sites_flipped)
            for (i=0; i<no_of_sites; i++)
            {
                if (spin[i + 0] == s)
                {
                    double field_local[dim_S];
                    double spin_local[dim_S];
                    double E_old = Energy_old(i, spin_local, field_local);
                    if (E_old>=0)
                    {
                        spin[i] = -spin[i];
                        no_of_sites_flipped = no_of_sites_flipped + 1;
                    }
                }
            }
            remaining_sites -= no_of_sites_flipped;
        } while ( no_of_sites_flipped > 0 );

        return remaining_sites;
    }

    long int flip_unstables(long int* nucleation_sites, long int remaining_sites, long int no_of_nuclei)
    {
        int j_SS = 0, j_L, k_L;
        long int next_site = nucleation_sites[0];
        long int i_1=0, i_2=0;
        double s = spin[dim_S*nucleation_sites[i_2] + j_SS];
        for (i_2=0; i_2<no_of_nuclei; i_2++)
        {
            spin[dim_S*nucleation_sites[i_2] + j_SS] = -s;
        }
        i_2 = no_of_nuclei;
        
        do
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                for (k_L=0; k_L<2; k_L++)
                {
                    next_site = N_N_I[2*dim_L*nucleation_sites[i_1] + 2*j_L + k_L];
                    if ( next_site < no_of_sites )
                    {
                        if (spin[dim_S*next_site + j_SS] != -s)
                        {
                            double field_local[dim_S];
                            double spin_local[dim_S];
                            if (Energy_old(next_site, spin_local, field_local)>=0)
                            {
                                nucleation_sites[i_2] = next_site;
                                i_2++;
                                spin[dim_S*next_site + j_SS] = -spin[dim_S*next_site + j_SS];
                                
                                // remaining_sites = flip_unstable(next_site, remaining_sites);
                            }
                        }
                    }
                }
            }
            i_1++;
        }
        while(i_1 != i_2);

        return remaining_sites - i_2;
    }

    int zero_temp_spin_at_hc(int inc_dec, long int count_local)
    {
        start_time = get_time_if_parallel();
        int jj_S=0, j_S, j_L;
        //
            long int thermal_i_temp = thermal_i;
            long int average_j_temp = average_j;
            long int sampling_inter_temp = sampling_inter;
            int MC_algo_type_temp = MC_algo_type;
            int MC_update_type_temp = MC_update_type;
            double T_temp = T;
            double h_temp[dim_S];
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h_temp[j_S] = h[j_S];
            }

        thermal_i = 0;
        average_j = 0;
        sampling_inter = 0;
        MC_algo_type = -1; // Glauber
        MC_update_type = -1; // Checkerboard
        T = 0;

        printf("\nztne RFIM finding hc at T=%lf.. \n",  T);
        fflush(stdout);

        double delta_h = del_h;

        // h[0] = h_i_max;
        if (fabs(h_i_max) < fabs(h_i_min))
        {
            h[0] = (double)inc_dec*fabs(h_i_min);
        }
        else
        {
            h[0] = (double)inc_dec*fabs(h_i_max);
        }

        delta_h = del_h;
        order[0] = (double)inc_dec;
        h_order = 0;
        r_order = 0;
        initialize_spin_config();
        h_counter = count_local;
        // m_counter = (double)inc_dec;
        m[0] = (double)inc_dec;

        /* // set output_variable_name=1 here
        {
            output_m = 1;
            // output_E = 1;
            output_h = 1;
            output_append = 1;
        } */

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            if (inc_dec==1)
            {
                pos += sprintf(pos, "O(%d)_%dD_zt_hc(-)[%ld]_", dim_S, dim_L, h_counter+1);
            }
            else
            {
                pos += sprintf(pos, "O(%d)_%dD_zt_hc(+)[%ld]_", dim_S, dim_L, h_counter+1);
            }

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_BOND
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); 
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.2f,%.2f)", h_i_max, h_i_min);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", order[j_S]);
            }
            pos += sprintf(pos, "}.dat");
        }
        // column labels and parameters
        output_param_file(output_file_0);
        output_label(output_file_0, "", "Saved_Spin\t");
        fflush(stdout);
        long int remaining_sites = no_of_sites;
        output_data(output_file_0, "", "No\t");

        h[0] = 0.0;
        delta_h = del_h;
        long int no_of_nuclei;;
        no_of_nuclei = mark_unstables(order[0], remaining_sites, nucleation_sites);
        remaining_sites = flip_marked_parallel(order[0], remaining_sites, nucleation_sites, no_of_nuclei);
        
        m[0] = order[0]*(double)(2*remaining_sites-no_of_sites)/(double)no_of_sites;
        
        m_bkp[0] = m[0];
        output_data(output_file_0, "", "No\t");

        printf("Start : h=%lf ...\n", h[0]);
        
        double remaining_sites_new;
        fflush(stdout);
        while (remaining_sites > no_of_sites/2)
        {
            h[0] -= order[0]*delta_h;
            no_of_nuclei = mark_unstables(order[0], remaining_sites, nucleation_sites);
            remaining_sites_new = flip_marked_parallel(order[0], remaining_sites, nucleation_sites, no_of_nuclei);
            // remaining_sites_new = flip_all_unstable(order[0], remaining_sites);
            
            m[0] = order[0]*(double)(2*remaining_sites_new-no_of_sites)/(double)no_of_sites;
            
            if (remaining_sites_new < no_of_sites/2)
            {
                if(delta_h>=del_h_cutoff)
                {
                    h[0] += order[0]*delta_h;
                    delta_h /= 2.0;
                    
                    unflipp_sites(order[0], remaining_sites, nucleation_sites, remaining_sites_new);
                    printf("\rNow at: h = %lf(+%.2e), m = %lf(+%ld) |r| [t=%le s] ..  ", h[0], delta_h, m[0], remaining_sites, get_time_if_parallel()-start_time);
                    fflush(stdout);
                    continue;
                }
                else
                {
                    if (fabs(m_bkp[0])>fabs(m[0]))
                    {
                        remaining_sites = remaining_sites_new;
                        m_bkp[0] = m[0];
                        output_data(output_file_0, "", "No\t");
                        printf("\rNow at: h = %lf(+%.2e), m = %lf(+%ld) |b| [t=%le s] ..  ", h[0], delta_h, m[0], remaining_sites, get_time_if_parallel()-start_time);
                        fflush(stdout);
                    }
                    else
                    {
                        unflipp_sites(order[0], remaining_sites, nucleation_sites, remaining_sites_new);
                        m[0] = m_bkp[0];
                        h[0] += order[0]*delta_h;
                    }
                    
                    break;
                }
            }
            else 
            {
                
                if (remaining_sites - remaining_sites_new < (remaining_sites_new - no_of_sites/2)/2)
                {
                    delta_h *= 2.0;
                    if (delta_h > del_h)
                    {
                        delta_h = del_h;
                    }
                }
                remaining_sites = remaining_sites_new;
                m_bkp[0] = m[0];
                output_data(output_file_0, "", "No\t");
                printf("\rNow at: h = %lf(+%.2e), m = %lf(+%ld) |b| [t=%le s] ..  ", h[0], delta_h, m[0], remaining_sites, get_time_if_parallel()-start_time);
                fflush(stdout);
            }
            
        }
        printf("\nEnd : h=%lf .       \n", h[0]);
        #ifdef SAVE_SPIN_AFTER
            if (SAVE_SPIN_AFTER > 0)
            {
                h_counter++;
                if (h_counter%SAVE_SPIN_AFTER == 0)
                {
                    char append_string[128];
                    char *pos_append_string = append_string;
                    if (inc_dec==1)
                    {
                        pos_append_string += sprintf(pos_append_string, "_dec_%ld", h_counter/SAVE_SPIN_AFTER);
                    }
                    else
                    {
                        pos_append_string += sprintf(pos_append_string, "_inc_%ld", h_counter/SAVE_SPIN_AFTER);
                    }
                    
                    save_spin_config(append_string, "w", 2);

                    output_data(output_file_0, "", "Yes\t");
                }
            }
        #endif
        // reset_output_variable_name_0();
            
            thermal_i = thermal_i_temp;
            average_j = average_j_temp;
            sampling_inter = sampling_inter_temp;
            MC_algo_type = MC_algo_type_temp;
            MC_update_type = MC_update_type_temp;
            T = T_temp;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_temp[j_S];
            }

        printf("\n----------\nDone.\n");
        fflush(stdout);
        return 0;
    }

    int zero_temp_IM_hysteresis_with_changing_field(int inc_dec)
    {
        start_time = get_time_if_parallel();
        int jj_S=0, j_S, j_L;
        //
            long int thermal_i_temp = thermal_i;
            long int average_j_temp = average_j;
            long int sampling_inter_temp = sampling_inter;
            int MC_algo_type_temp = MC_algo_type;
            int MC_update_type_temp = MC_update_type;
            double T_temp = T;
            double h_temp[dim_S];
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h_temp[j_S] = h[j_S];
            }

        thermal_i = 0;
        average_j = 0;
        sampling_inter = 0;
        MC_algo_type = -1; // Glauber
        MC_update_type = -1; // Checkerboard
        T = 0;
        if (spin_bkp_reqd != 1)
        {
            spin_bkp_reqd = 1;
            allocate_memory();
        }

        printf("\nztne RFIM looping  at T=%lf.. \n",  T);
        fflush(stdout);

        double delta_h = del_h;

        // h[0] = h_i_max;
        if (fabs(h_i_max) < fabs(h_i_min))
        {
            h[0] = (double)inc_dec*fabs(h_i_min);
        }
        else
        {
            h[0] = (double)inc_dec*fabs(h_i_max);
        }

        delta_h = del_h;
        order[0] = (double)inc_dec;
        h_order = 0;
        r_order = 0;
        initialize_spin_config();
        h_counter = 0;
        m_counter = (double)inc_dec;

        m[0] = (double)inc_dec;
        // ensemble_m();
        // ensemble_E();

        // print statements:
        {
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({%lf-->%lf", order[0], -order[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", order[j_S]);
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        fflush(stdout);
        /* // set output_variable_name=1 here
        {
            output_m = 1;
            // output_E = 1;
            output_h = 1;
            output_append = 1;
        } */
        fflush(stdout);
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            if (inc_dec==1)
            {
                pos += sprintf(pos, "O(%d)_%dD_ztne_delh(-)_", dim_S, dim_L);
            }
            else
            {
                pos += sprintf(pos, "O(%d)_%dD_ztne_delh(+)_", dim_S, dim_L);
            }

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_BOND
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); 
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.2f,%.2f)", h_i_max, h_i_min);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", order[j_S]);
            }
            pos += sprintf(pos, "}.dat");
        }
        // column labels and parameters
        output_param_file(output_file_0);
        output_label(output_file_0, "", "Saved_Spin\t");
        fflush(stdout);
        // pFile_1 = fopen(output_file_0, "a");

        // nucleation_sites = (long int*)malloc(no_of_sites*sizeof(double));

        long int remaining_sites = no_of_sites;
        output_data(output_file_0, "", "No\t");

        h[0] = 0.0;
        delta_h = del_h;
        long int no_of_nuclei;;
        no_of_nuclei = mark_unstables(order[0], remaining_sites, nucleation_sites);
        remaining_sites = flip_marked_parallel(order[0], remaining_sites, nucleation_sites, no_of_nuclei);
        // remaining_sites = flip_all_unstable(order[0], remaining_sites);
        m[0] = order[0]*(double)(2*remaining_sites-no_of_sites)/(double)no_of_sites;
        // backing_up_spin();
        m_bkp[0] = m[0];
        output_data(output_file_0, "", "No\t");

        printf("----\n");
        printf("\nm = %lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("\norder = ({%lf-->%lf", order[0], -order[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", order[j_S]);
        }
        printf("}, %d, %d)\n", h_order, r_order);

        printf("Start : h=%lf ...\n", h[0]);
        double delta_m_local;
        double remaining_sites_new;
        fflush(stdout);
        while (remaining_sites)
        {
            h[0] -= order[0]*delta_h;
            no_of_nuclei = mark_unstables(order[0], remaining_sites, nucleation_sites);
            remaining_sites_new = flip_marked_parallel(order[0], remaining_sites, nucleation_sites, no_of_nuclei);
            // remaining_sites_new = flip_all_unstable(order[0], remaining_sites);
            
            m[0] = order[0]*(double)(2*remaining_sites_new-no_of_sites)/(double)no_of_sites;
            delta_m_local = fabs(m_bkp[0]-m[0]);
            if (delta_m_local>CUTOFF_M)
            {
                if(delta_h>=del_h_cutoff)
                {
                    h[0] += order[0]*delta_h;
                    delta_h /= 2.0;
                    // restoring_spin(h[0], delta_h, 0, delta_m_local, "h", 0);
                    unflipp_sites(order[0], remaining_sites, nucleation_sites, remaining_sites_new);
                    printf("\rNow at: h = %lf(+%.2e), m = %lf(+%.2e) |r| [t=%le s] ..  ", h[0], delta_h, m[0], delta_m_local, get_time_if_parallel()-start_time);
                    fflush(stdout);
                    continue;
                }
                else
                {
                    remaining_sites = remaining_sites_new;
                    m_bkp[0] = m[0];
                    // backing_up_spin();
                    output_data(output_file_0, "", "No\t");
                    printf("\rNow at: h = %lf(+%.2e), m = %lf(+%.2e) |b| [t=%le s] ..  ", h[0], delta_h, m[0], delta_m_local, get_time_if_parallel()-start_time);
                    fflush(stdout);
                }
            }
            else 
            {
                remaining_sites = remaining_sites_new;
                m_bkp[0] = m[0];
                // backing_up_spin();
                output_data(output_file_0, "", "No\t");
                if (delta_m_local<CUTOFF_M/3.0)
                {
                    delta_h *= 2.0;
                    if (delta_h > del_h)
                    {
                        delta_h = del_h;
                    }
                }
                printf("\rNow at: h = %lf(+%.2e), m = %lf(+%.2e) |b| [t=%le s] ..  ", h[0], delta_h, m[0], delta_m_local, get_time_if_parallel()-start_time);
                fflush(stdout);
            }
            #ifdef SAVE_SPIN_AFTER
                if (SAVE_SPIN_AFTER > 0)
                {
                    h_counter++;
                    if (h_counter%SAVE_SPIN_AFTER == 0)
                    {
                        char append_string[128];
                        char *pos_append_string = append_string;
                        if (inc_dec==1)
                        {
                            pos_append_string += sprintf(pos_append_string, "_dec_%ld", h_counter/SAVE_SPIN_AFTER);
                        }
                        else
                        {
                            pos_append_string += sprintf(pos_append_string, "_inc_%ld", h_counter/SAVE_SPIN_AFTER);
                        }
                        
                        save_spin_config(append_string, "w", 2);
                        // ensemble_all();
                        // pFile_output = pFile_1;
                        // output_data("output_file.dat", "", "Yes\t");
                        // pFile_output = NULL;
                        output_data(output_file_0, "", "Yes\t");
                    }
                }
                else
                {
                    if (order[0]*m_counter>order[0]*m[0] && remaining_sites>0) // ( h_counter % SAVE_SPIN_AFTER == 0 )
                    {
                        h_counter++;
                        m_counter -= order[0]*del_m_counter;
                        char append_string[128];
                        char *pos_append_string = append_string;
                        if (inc_dec==1)
                        {
                            pos_append_string += sprintf(pos_append_string, "_dec_%ld", h_counter);
                        }
                        else
                        {
                            pos_append_string += sprintf(pos_append_string, "_inc_%ld", h_counter);
                        }
                        
                        save_spin_config(append_string, "w", 2);
                        // ensemble_all();
                        // pFile_output = pFile_1;
                        // output_data("output_file.dat", "", "Yes\t");
                        // pFile_output = NULL;
                        output_data(output_file_0, "", "Yes\t");
                        
                    }
                }
            #endif
        }
        printf("\nEnd : h=%lf .       \n", h[0]);
        /*     
            // h[0] = h_i_min;
            if (fabs(h_i_max) < fabs(h_i_min))
            {
                h[0] = -fabs(h_i_min);
            }
            else
            {
                h[0] = -fabs(h_i_max);
            }

            order[0] = -1;
            m_counter = -1.0;
            remaining_sites = no_of_sites;
            printf("----\n");
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({%lf-->%lf", order[0], -order[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", order[j_S]);
            }
            printf("}, %d, %d)\n", h_order, r_order);
            fflush(stdout);
            
            pFile_1 = fopen(output_file_0, "a");
            fprintf(pFile_1, "===== Reversing direction of field sweep =====\n");
            fclose(pFile_1);
            remaining_sites = no_of_sites;
            // pFile_output = pFile_1;
            // output_data("output_file.dat", "", "No\t");
            // pFile_output = NULL;
            output_data(output_file_0, "", "No\t");

            h[0] = 0.0;
            delta_h = del_h;
            remaining_sites = flip_all_unstable(order[0], remaining_sites);
            m[0] = order[0]*(double)(2*remaining_sites-no_of_sites)/(double)no_of_sites;
            backing_up_spin();
            // pFile_output = pFile_1;
            // output_data("output_file.dat", "", "No\t");
            // pFile_output = NULL;
            output_data(output_file_0, "", "No\t");
            
            printf("Start : h=%lf ...\n", h[0]);
            while (remaining_sites)
            {
                h[0] -= order[0]*delta_h;
                double remaining_sites_new = flip_all_unstable(order[0], remaining_sites);
                
                m[0] = order[0]*(double)(2*remaining_sites-no_of_sites)/(double)no_of_sites;
                double delta_m_local = m_bkp[0]-m[0];
                if (fabs(delta_m_local)>CUTOFF_M)
                {
                    if(delta_h>=del_h_cutoff)
                    {
                        h[0] += order[0]*delta_h;
                        delta_h /= 2.0;
                        restoring_spin(h[0], delta_h, 0, delta_m_local, "h", 0);
                        continue;
                    }
                    else
                    {
                        remaining_sites = remaining_sites_new;
                        backing_up_spin();
                        // pFile_output = pFile_1;
                        // output_data("output_file.dat", "", "No\t");
                        // pFile_output = NULL;
                        output_data(output_file_0, "", "No\t");
                        printf("\rNow at: h = %lf, m = %lf .. ", h[0], m[0]);
                        fflush(stdout);
                    }
                    
                }
                else 
                {
                    remaining_sites = remaining_sites_new;
                    backing_up_spin();
                    // pFile_output = pFile_1;
                    // output_data("output_file.dat", "", "No\t");
                    // pFile_output = NULL;
                    output_data(output_file_0, "", "No\t");
                    printf("\rNow at: h = %lf, m = %lf .. ", h[0], m[0]);
                    fflush(stdout);
                    if (fabs(delta_m_local)<CUTOFF_M/3.0)
                    {
                        delta_h *= 2.0;
                        if (delta_h > del_h)
                        {
                            delta_h = del_h;
                        }
                    }
                }
                #ifdef SAVE_SPIN_AFTER
                    if (SAVE_SPIN_AFTER > 0)
                    {
                        h_counter++;
                        if (h_counter%SAVE_SPIN_AFTER == 0)
                        {
                            char append_string[128];
                            char *pos_append_string = append_string;
                            pos_append_string += sprintf(pos_append_string, "_inc_%ld", h_counter/SAVE_SPIN_AFTER);
                            save_spin_config(append_string, "w", 2);
                            // ensemble_all();
                            // pFile_output = pFile_1;
                            // output_data("output_file.dat", "", "Yes\t");
                            // pFile_output = NULL;
                            output_data(output_file_0, "", "Yes\t");
                        }
                    }
                    else
                    {
                        if (order[0]*m_counter>order[0]*m[0] && remaining_sites>0) // ( h_counter % SAVE_SPIN_AFTER == 0 )
                        {
                            h_counter++;
                            m_counter -= order[0]*del_m_counter;
                            char append_string[128];
                            char *pos_append_string = append_string;
                            pos_append_string += sprintf(pos_append_string, "_inc_%ld", h_counter);
                            save_spin_config(append_string, "w", 2);
                            // ensemble_all();
                            // pFile_output = pFile_1;
                            // output_data("output_file.dat", "", "Yes\t");
                            // pFile_output = NULL;
                            output_data(output_file_0, "", "Yes\t");
                        }
                    }
                #endif
                fflush(stdout);
            }
            printf("\rEnd : h=%lf .       \n", h[0]);
            // fclose(pFile_1);
         */    
        // reset_output_variable_name_0();
            
            thermal_i = thermal_i_temp;
            average_j = average_j_temp;
            sampling_inter = sampling_inter_temp;
            MC_algo_type = MC_algo_type_temp;
            MC_update_type = MC_update_type_temp;
            T = T_temp;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_temp[j_S];
            }

        printf("\n----------\nDone.\n");
        fflush(stdout);
        return 0;
    }

    int zero_temp_RFIM_hysteresis()
    {
        int jj_S=0, j_S, j_L;
        //
            long int thermal_i_temp = thermal_i;
            long int average_j_temp = average_j;
            long int sampling_inter_temp = sampling_inter;
            int MC_algo_type_temp = MC_algo_type;
            int MC_update_type_temp = MC_update_type;
            double T_temp = T;
            double h_temp[dim_S];
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h_temp[j_S] = h[j_S];
            }

        thermal_i = 0;
        average_j = 0;
        sampling_inter = 0;
        MC_algo_type = -1; // Glauber
        MC_update_type = -1; // Checkerboard
        T = 0;

        // h[0] = h_i_max;
        if (fabs(h_i_max) < fabs(h_i_min))
        {
            h[0] = fabs(h_i_min);
        }
        else
        {
            h[0] = fabs(h_i_max);
        }
        
        double delta_h = del_h;
        order[0] = 1;
        h_order = 0;
        r_order = 0;
        initialize_spin_config();
        h_counter = 0;
        m_counter = 1.0;

        printf("\nztne RFIM looping  at T=%lf.. \n",  T);

        ensemble_m();
        ensemble_E();

        // print statements:
        {
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({%lf-->%lf", order[0], -order[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", order[j_S]);
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        /* // set output_variable_name=1 here
        {
            output_m = 1;
            output_E = 1;
            output_h = 1;
            output_append = 1;
        } */

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hysteresis_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_BOND
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); 
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.2f,%.2f)", h_i_max, h_i_min);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", order[j_S]);
            }
            pos += sprintf(pos, "}.dat");
        }
        // column labels and parameters
        output_param_file(output_file_0);
        output_label(output_file_0, "", "Saved_Spin\t");

        pFile_1 = fopen(output_file_0, "a");

        long int no_of_nuclei;
        // nucleation_sites = (long int*)malloc(no_of_sites*sizeof(double));

        long int remaining_sites = no_of_sites;
        // ensemble_all();
        m[0] = order[0]*(double)(2*remaining_sites-no_of_sites)/(double)no_of_sites;
        pFile_output = pFile_1;
        output_data("output_file.dat", "", "No\t");
        pFile_output = NULL;

        printf("----\n");
        printf("\nm = %lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("\norder = ({%lf-->%lf", order[0], -order[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", order[j_S]);
        }
        printf("}, %d, %d)\n", h_order, r_order);

        printf("Start : h=%lf ...\n", h[0]);
        while (remaining_sites)
        {
            // no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);
            no_of_nuclei = mark_extremes(order[0], remaining_sites, nucleation_sites);

            // ensemble_all();
            pFile_output = pFile_1;
            output_data("output_file.dat", "", "No\t");
            pFile_output = NULL;
            
            // remaining_sites = flip_unstables(nucleation_sites, remaining_sites, no_of_nuclei);
            remaining_sites = flip_marked_parallel(order[0], remaining_sites, nucleation_sites, no_of_nuclei);

            printf("\rNow at: h=%lf, [%ld,%ld] ", h[0], remaining_sites, no_of_nuclei);
            fflush(stdout);
            // ensemble_all();
            m[0] = order[0]*(double)(2*remaining_sites-no_of_sites)/(double)no_of_sites;
            pFile_output = pFile_1;
            output_data("output_file.dat", "", "No\t");
            pFile_output = NULL;

            #ifdef SAVE_SPIN_AFTER
                if (SAVE_SPIN_AFTER > 0)
                {
                    h_counter++;
                    if (h_counter%SAVE_SPIN_AFTER == 0)
                    {
                        char append_string[128];
                        char *pos_append_string = append_string;
                        pos_append_string += sprintf(pos_append_string, "_dec_%ld", h_counter/SAVE_SPIN_AFTER);
                        save_spin_config(append_string, "w", 2);
                        // ensemble_all();
                        pFile_output = pFile_1;
                        output_data("output_file.dat", "", "Yes\t");
                        pFile_output = NULL;
                    }
                }
                else
                {
                    if (order[0]*m_counter>order[0]*m[0] && remaining_sites>0) // ( h_counter % SAVE_SPIN_AFTER == 0 )
                    {
                        h_counter++;
                        m_counter -= order[0]*del_m_counter;
                        char append_string[128];
                        char *pos_append_string = append_string;
                        pos_append_string += sprintf(pos_append_string, "_dec_%ld", h_counter);
                        save_spin_config(append_string, "w", 2);
                        // ensemble_all();
                        pFile_output = pFile_1;
                        output_data("output_file.dat", "", "Yes\t");
                        pFile_output = NULL;
                    }
                }
            #endif
        }
        printf("\rEnd : h=%lf .       \n", h[0]);
        
        // h[0] = h_i_min;
        if (fabs(h_i_max) < fabs(h_i_min))
        {
            h[0] = -fabs(h_i_min);
        }
        else
        {
            h[0] = -fabs(h_i_max);
        }

        order[0] = -1;
        m_counter = -1.0;
        remaining_sites = no_of_sites;
        

        fprintf(pFile_1, "===== Reversing direction of field sweep =====\n");
        // ensemble_all();
        m[0] = order[0]*(double)(2*remaining_sites-no_of_sites)/(double)no_of_sites;
        pFile_output = pFile_1;
        output_data("output_file.dat", "", "No\t");
        pFile_output = NULL;
        printf("----\n");
        printf("\nm = %lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("\norder = ({%lf-->%lf", order[0], -order[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", order[j_S]);
        }
        printf("}, %d, %d)\n", h_order, r_order);
        
        printf("Start : h=%lf ...\n", h[0]);
        while (remaining_sites)
        {
            // no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);
            no_of_nuclei = mark_extremes(order[0], remaining_sites, nucleation_sites);

            // ensemble_all();
            pFile_output = pFile_1;
            output_data("output_file.dat", "", "No\t");
            pFile_output = NULL;
            
            // remaining_sites = flip_unstables(nucleation_sites, remaining_sites, no_of_nuclei);
            remaining_sites = flip_marked_parallel(order[0], remaining_sites, nucleation_sites, no_of_nuclei);

            printf("\rNow at: h=%lf, [%ld,%ld] ", h[0], remaining_sites, no_of_nuclei);
            fflush(stdout);
            // ensemble_all();
            m[0] = order[0]*(double)(2*remaining_sites-no_of_sites)/(double)no_of_sites;
            pFile_output = pFile_1;
            output_data("output_file.dat", "", "No\t");
            pFile_output = NULL;
            #ifdef SAVE_SPIN_AFTER
                if (SAVE_SPIN_AFTER > 0)
                {
                    h_counter++;
                    if (h_counter%SAVE_SPIN_AFTER == 0)
                    {
                        char append_string[128];
                        char *pos_append_string = append_string;
                        pos_append_string += sprintf(pos_append_string, "_inc_%ld", h_counter/SAVE_SPIN_AFTER);
                        save_spin_config(append_string, "w", 2);
                        // ensemble_all();
                        pFile_output = pFile_1;
                        output_data("output_file.dat", "", "Yes\t");
                        pFile_output = NULL;
                    }
                }
                else
                {
                    if (order[0]*m_counter>order[0]*m[0] && remaining_sites>0) // ( h_counter % SAVE_SPIN_AFTER == 0 )
                    {
                        h_counter++;
                        m_counter -= order[0]*del_m_counter;
                        char append_string[128];
                        char *pos_append_string = append_string;
                        pos_append_string += sprintf(pos_append_string, "_inc_%ld", h_counter);
                        save_spin_config(append_string, "w", 2);
                        // ensemble_all();
                        pFile_output = pFile_1;
                        output_data("output_file.dat", "", "Yes\t");
                        pFile_output = NULL;
                    }
                }
            #endif
        }
        printf("\rEnd : h=%lf .       \n", h[0]);
        fclose(pFile_1);
        // reset_output_variable_name_0();
            
            thermal_i = thermal_i_temp;
            average_j = average_j_temp;
            sampling_inter = sampling_inter_temp;
            MC_algo_type = MC_algo_type_temp;
            MC_update_type = MC_update_type_temp;
            T = T_temp;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_temp[j_S];
            }

        printf("\n----------\nDone.\n");

        return 0;
    }

    int zero_temp_RFIM_ringdown(double delta_m)
    {
        int jj_S=0, j_S, j_L;
        //
            long int thermal_i_temp = thermal_i;
            long int average_j_temp = average_j;
            long int sampling_inter_temp = sampling_inter;
            int MC_algo_type_temp = MC_algo_type;
            int MC_update_type_temp = MC_update_type;
            double T_temp = T;
            double h_temp[dim_S];
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h_temp[j_S] = h[j_S];
            }

        thermal_i = 0;
        average_j = 0;
        sampling_inter = 0;
        MC_algo_type = -1; // Glauber
        MC_update_type = -1; // Checkerboard
        T = 0;

        // h[0] = h_i_max;
        if (fabs(h_i_max) < fabs(h_i_min))
        {
            h[0] = fabs(h_i_min);
        }
        else
        {
            h[0] = fabs(h_i_max);
        }
        
        double delta_h = del_h;
        order[0] = 1;
        h_order = 0;
        r_order = 0;
        initialize_spin_config();
        h_counter = 0;
        m_counter = 1.0;

        printf("\nztne RFIM looping  at T=%lf.. \n",  T);
        
        ensemble_m();
        ensemble_E();

        // print statements:
        {
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({%lf-->%lf", order[0], -order[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", order[j_S]);
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        /* // set output_variable_name=1 here
        {
            output_m = 1;
            output_E = 1;
            output_h = 1;
            output_append = 1;
        } */
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_ringdown_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_BOND
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.2f,%.2f)", h_i_max, h_i_min);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", order[j_S]);
            }
            pos += sprintf(pos, "}.dat");
        }
        // column labels and parameters
        output_param_file(output_file_0);
        output_label(output_file_0, "", "Saved_Spin\t");

        pFile_1 = fopen(output_file_0, "a");
        long int remaining_sites = 0;

        long int no_of_nuclei;
        // nucleation_sites = (long int*)malloc(no_of_sites*sizeof(double));

        float M_compare;
        for (M_compare=1.0; M_compare>0.0; M_compare -= delta_m)
        {
            // Put modified ztne RFIM hysteresis code here.
            
            order[0] = 1;
            remaining_sites = no_of_sites - remaining_sites;
            ensemble_all();
            pFile_output = pFile_1;
            output_data("output_file.dat", "", "No\t");
            pFile_output = NULL;
            
            double old_h=0.0, new_h=0.0;
            printf("----\n");
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({%lf-->%lf", order[0], -order[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", order[j_S]);
            }
            printf("}, %d, %d)\n", h_order, r_order);

            printf("Start : h=%lf ...\n", h[0]);

            no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);

            while ( old_h == new_h || M_compare > -m[0] )
            {
                // ensemble_all();
                pFile_output = pFile_1;
                output_data("output_file.dat", "", "No\t");
                pFile_output = NULL;
                
                remaining_sites = flip_unstables(nucleation_sites, remaining_sites, no_of_nuclei);

                printf("\rNow at: h=%lf ", h[0]);
                fflush(stdout);

                ensemble_all();
                pFile_output = pFile_1;
                output_data("output_file.dat", "", "No\t");
                pFile_output = NULL;

                #ifdef SAVE_SPIN_AFTER
                    if (SAVE_SPIN_AFTER > 0)
                    {
                        h_counter++;
                        if (h_counter%SAVE_SPIN_AFTER == 0)
                        {
                            char append_string[128];
                            char *pos_append_string = append_string;
                            pos_append_string += sprintf(pos_append_string, "_dec_%ld", h_counter/SAVE_SPIN_AFTER);
                            save_spin_config(append_string, "w", 2);
                            // ensemble_all();
                            pFile_output = pFile_1;
                            output_data("output_file.dat", "", "Yes\t");
                            pFile_output = NULL;
                        }
                    }
                    else
                    {
                        if (order[0]*m_counter>order[0]*m[0] && remaining_sites>0) // ( h_counter % SAVE_SPIN_AFTER == 0 )
                        {
                            h_counter++;
                            m_counter -= order[0]*del_m_counter;
                            char append_string[128];
                            char *pos_append_string = append_string;
                            pos_append_string += sprintf(pos_append_string, "_dec_%ld", h_counter);
                            save_spin_config(append_string, "w", 2);
                            // ensemble_all();
                            pFile_output = pFile_1;
                            output_data("output_file.dat", "", "Yes\t");
                            pFile_output = NULL;
                        }
                    }
                #endif

                old_h = h[0];

                no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);

                new_h = h[0];
            }
            h[0] = old_h;
            printf("\rEnd : h=%lf .       \n", h[0]);
            fprintf(pFile_1, "===== Reversing direction of field sweep =====\n");

            order[0] = -1;
            remaining_sites = no_of_sites - remaining_sites;
            printf("----\n");
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({%lf-->%lf", order[0], -order[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", order[j_S]);
            }
            printf("}, %d, %d)\n", h_order, r_order);

            ensemble_all();
            pFile_output = pFile_1;
            output_data("output_file.dat", "", "No\t");
            pFile_output = NULL;

            old_h = 0.0, new_h = 0.0;
            
            printf("Start : h=%lf ...\n", h[0]);
            no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);

            while (old_h == new_h || M_compare > m[0])
            {
                // ensemble_all();
                pFile_output = pFile_1;
                output_data("output_file.dat", "", "No\t");
                pFile_output = NULL;
                
                remaining_sites = flip_unstables(nucleation_sites, remaining_sites, no_of_nuclei);
                
                printf("\rNow at: h=%lf ", h[0]);
                fflush(stdout);

                ensemble_all();
                pFile_output = pFile_1;
                output_data("output_file.dat", "", "No\t");
                pFile_output = NULL;
                #ifdef SAVE_SPIN_AFTER
                    if (SAVE_SPIN_AFTER > 0)
                    {
                        h_counter++;
                        if (h_counter%SAVE_SPIN_AFTER == 0)
                        {
                            char append_string[128];
                            char *pos_append_string = append_string;
                            pos_append_string += sprintf(pos_append_string, "_inc_%ld", h_counter/SAVE_SPIN_AFTER);
                            save_spin_config(append_string, "w", 2);
                            // ensemble_all();
                            pFile_output = pFile_1;
                            output_data("output_file.dat", "", "Yes\t");
                            pFile_output = NULL;
                        }
                    }
                    else
                    {
                        if (order[0]*m_counter>order[0]*m[0] && remaining_sites>0) // ( h_counter % SAVE_SPIN_AFTER == 0 )
                        {
                            h_counter++;
                            m_counter -= order[0]*del_m_counter;
                            char append_string[128];
                            char *pos_append_string = append_string;
                            pos_append_string += sprintf(pos_append_string, "_inc_%ld", h_counter);
                            save_spin_config(append_string, "w", 2);
                            // ensemble_all();
                            pFile_output = pFile_1;
                            output_data("output_file.dat", "", "Yes\t");
                            pFile_output = NULL;
                        }
                    }
                #endif

                old_h = h[0];

                no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);

                new_h = h[0];
            }
            h[0] = old_h;
            printf("\rEnd : h=%lf .       \n", h[0]);
            fprintf(pFile_1, "===== Reversing direction of field sweep =====\n");
        }
        printf("\n----------\nDone.\n");
        fclose(pFile_1);
        // reset_output_variable_name_0();
        
            thermal_i = thermal_i_temp;
            average_j = average_j_temp;
            sampling_inter = sampling_inter_temp;
            MC_algo_type = MC_algo_type_temp;
            MC_update_type = MC_update_type_temp;
            T = T_temp;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_temp[j_S];
            }

        return 0;
    }

    int zero_temp_RFIM_return_point_memory()
    {
        int jj_S=0, j_S, j_L;
        //
            long int thermal_i_temp = thermal_i;
            long int average_j_temp = average_j;
            long int sampling_inter_temp = sampling_inter;
            int MC_algo_type_temp = MC_algo_type;
            int MC_update_type_temp = MC_update_type;
            double T_temp = T;
            double h_temp[dim_S];
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h_temp[j_S] = h[j_S];
            }

        thermal_i = 0;
        average_j = 0;
        sampling_inter = 0;
        MC_algo_type = -1; // Metropolis
        MC_update_type = -1; // Checkerboard
        T = 0;

        double delta_m[] = { 0.2, 0.1 , 0.05, 0.025 }; // change this to appropriate values 
        double m_start = 0.0;
        int depth_of_subloop = sizeof(delta_m) / sizeof(delta_m[0]);
        double *h_ext = (double *)malloc((depth_of_subloop+1)*sizeof(double));
        double *mag_rpm = (double *)malloc((depth_of_subloop+1)*sizeof(double));
        
        // h[0] = h_i_max;
        if (fabs(h_i_max) < fabs(h_i_min))
        {
            h[0] = fabs(h_i_min);
        }
        else
        {
            h[0] = fabs(h_i_max);
        }

        double delta_h = del_h;
        order[0] = 1;
        h_order = 0;
        r_order = 0;
        initialize_spin_config();
        h_counter = 0;
        m_counter = 1.0;

        printf("\nztne RFIM looping  at T=%lf.. \n",  T);
        
        ensemble_m();
        ensemble_E();
        
        // print statements:
        {
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({%lf-->%lf", order[0], -order[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", order[j_S]);
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        /* // set output_variable_name=1 here
        {
            output_m = 1;
            output_E = 1;
            output_h = 1;
            output_append = 1;
        } */

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_rpm_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_BOND
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.2f,%.2f)", h_i_max, h_i_min);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");
            #ifdef RANDOM_FIELD
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            #endif
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", order[j_S]);
            }
            pos += sprintf(pos, "}.dat");
        }
        // column labels and parameters
        output_param_file(output_file_0);
        output_label(output_file_0, "", "Saved_Spin\tRPM_error(rpm-m)\t");

        pFile_1 = fopen(output_file_0, "a");
        long int remaining_sites = 0;

        long int no_of_nuclei;
        // nucleation_sites = (long int*)malloc(no_of_sites*sizeof(double));

        float M_compare;
        int i;
        double old_h = 0.0, new_h = 0.0;

        order[0] = 1;
        remaining_sites = no_of_sites - remaining_sites;

        ensemble_all();
        pFile_output = pFile_1;
        output_data("output_file.dat", "", "No\t");
        pFile_output = NULL;

        no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);

        while (old_h == new_h || m_start < m[jj_S])
        {
            // ensemble_all();
            pFile_output = pFile_1;
            output_data("output_file.dat", "", "No\t");
            pFile_output = NULL;
            
            remaining_sites = flip_unstables(nucleation_sites, remaining_sites, no_of_nuclei);

            // printf("\rNow at: h=%lf ", h[0]);
            // fflush(stdout);

            ensemble_all();
            pFile_output = pFile_1;
            output_data("output_file.dat", "", "No\t");
            pFile_output = NULL;

            old_h = h[0];

            no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);

            new_h = h[0];
        }
        mag_rpm[0] = m[jj_S];

        h_ext[0] = old_h;

        printf("mag_rpm[0]=%lf, h_ext[0]=%lf \n", mag_rpm[0], h_ext[0]);

        for (i=0; i<depth_of_subloop; i++)
        {
            order[0] = - order[0];
            remaining_sites = no_of_sites - remaining_sites;
            fprintf(pFile_1, "===== Reversing direction of field sweep =====\n");

            char append_string[128];
            char *pos_append_string = append_string;
            h[0] = old_h;
            pos_append_string += sprintf(pos_append_string, "_frwd_loop_%d", i+1);
            save_spin_config(append_string, "w", 2);
            // ensemble_all();
            pFile_output = pFile_1;
            output_data("output_file.dat", "", "Yes\t");
            pFile_output = NULL;

            no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);

            while ( old_h == new_h || delta_m[i] > fabs( mag_rpm[i] - m[jj_S] ) )
            {
                // ensemble_all();
                pFile_output = pFile_1;
                output_data("output_file.dat", "", "No\t");
                pFile_output = NULL;
                
                remaining_sites = flip_unstables(nucleation_sites, remaining_sites, no_of_nuclei);

                // printf("\rNow at: h=%lf ", h[0]);
                // fflush(stdout);

                ensemble_all();
                pFile_output = pFile_1;
                output_data("output_file.dat", "", "No\t");
                pFile_output = NULL;

                old_h = h[0];

                no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);

                new_h = h[0];
            }
            mag_rpm[i+1] = m[jj_S];

            h_ext[i+1] = old_h;

            printf("mag_rpm[%d]=%lf, h_ext[%d]=%lf \n", i+1, mag_rpm[i+1], i+1, h_ext[i+1]);
        }
        fprintf(pFile_1,"===== Return Point Memory of subloops =====\n");
        for (i=depth_of_subloop-1; i>=0; i--)
        {
            order[0] = - order[0];
            remaining_sites = no_of_sites - remaining_sites;
            fprintf(pFile_1, "===== Reversing direction of field sweep =====\n");
            h[0] = old_h;
            ensemble_all();
            pFile_output = pFile_1;
            output_data("output_file.dat", "", "No\t");
            pFile_output = NULL;

            no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);

            while ( old_h == new_h || !( ( h_ext[i] >= old_h && h_ext[i] < new_h ) || ( h_ext[i] <= old_h && h_ext[i] > new_h ) ) )
            {
                // ensemble_all();
                pFile_output = pFile_1;
                output_data("output_file.dat", "", "No\t");
                pFile_output = NULL;
                

                remaining_sites = flip_unstables(nucleation_sites, remaining_sites, no_of_nuclei);

                // printf("\rNow at: h=%lf ", h[0]);
                // fflush(stdout);

                ensemble_all();
                pFile_output = pFile_1;
                output_data("output_file.dat", "", "No\t");
                pFile_output = NULL;

                old_h = h[0];

                no_of_nuclei = find_extremes(order[0], remaining_sites, nucleation_sites);

                new_h = h[0];
            }


            {
                char append_string[128];
                char *pos_append_string = append_string;
                // h[0] = old_h;
                pos_append_string += sprintf(pos_append_string, "_rtrn_loop_%d", i+1);
                save_spin_config(append_string, "w", 2);
                ensemble_all();
                char str_app[128];
                char *pos_app = str_app;
                pos_app += sprintf(pos_app, "Yes\t%.17e\t", mag_rpm[i]-m[jj_S]);
                pFile_output = pFile_1;
                output_data("output_file.dat", "", str_app);
                pFile_output = NULL;

                printf("m[0]=%lf, h_ext[%d]=%lf \n", m[jj_S], i, h_ext[i]);
            }
        }
        fclose(pFile_1);
        // reset_output_variable_name_0();
            
            thermal_i = thermal_i_temp;
            average_j = average_j_temp;
            sampling_inter = sampling_inter_temp;
            MC_algo_type = MC_algo_type_temp;
            MC_update_type = MC_update_type_temp;
            T = T_temp;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_temp[j_S];
            }
        
        free(h_ext);
        free(mag_rpm);
        return 0;
    }

//====================  RFXY helper functions         ====================//

    #ifdef enable_CUDA_CODE
        // __global__ void Energy_minimum_old_XY(long int sites, double* spin_local)
        __global__ void Energy_minimum_old_XY(long int sites)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            // long int stride = blockDim.x*gridDim.x;
            long int xyzi = index;
            
            if (index < sites)
            {
                int j_S, j_L, k_L;
                double Energy_min = 0.0;
                double field_local[dim_S];
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    field_local[j_S] = 0.0;
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        for (k_L=0; k_L<2; k_L++)
                        {
                            #ifdef RANDOM_BOND
                            field_local[j_S] = field_local[j_S] - (dev_J[j_L] + dev_J_random[2*dim_L*xyzi + 2*j_L + k_L]) * dev_spin[dim_S*dev_N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                            #else
                            field_local[j_S] = field_local[j_S] - dev_J[j_L] * dev_spin[dim_S*dev_N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                            #endif
                        }
                    }
                    #ifdef RANDOM_FIELD
                    field_local[j_S] = field_local[j_S] - (dev_h[j_S] + dev_h_random[dim_S*xyzi + j_S]);
                    #else
                    field_local[j_S] = field_local[j_S] - dev_h[j_S];
                    #endif
                    Energy_min = Energy_min + field_local[j_S] * field_local[j_S];
                }
                // if(Energy_min==0)
                // {
                //     for (j_S=0; j_S<dim_S; j_S++)
                //     {
                //         dev_spin_temp[dim_S*xyzi + j_S] = dev_spin[dim_S*xyzi + j_S];
                //     }
                // }
                // else // if(Energy_min>0)
                // {
                //     Energy_min = - sqrt(Energy_min);
                //     // Energy_min = - __dsqrt_rn(Energy_min);

                //     for (j_S=0; j_S<dim_S; j_S++)
                //     {
                //         dev_spin_temp[dim_S*xyzi + j_S] = field_local[j_S] / Energy_min;
                //     }
                // }
                {
                    Energy_min = -sqrt(Energy_min);
                    double energy_bool = (double)(Energy_min>=0);
                    Energy_min += energy_bool;
                    // Energy_min = ~(Energy_min); // unary bitflip
                    for (j_S=0; j_S<dim_S; j_S++)
                    {
                        dev_spin_temp[dim_S*xyzi + j_S] = (field_local[j_S] / Energy_min) + energy_bool * dev_spin[dim_S*xyzi + j_S];
                    }
                }
            }

            // return Energy_min;
        }

        __global__ void Energy_minimum_old_XY_checker(long int sites, int black_or_white)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            // long int stride = blockDim.x*gridDim.x;
            long int xyzi = dev_black_white_checkerboard[dev_no_of_black_white_sites[black_or_white]*black_or_white + index];
            
            if (index < sites)
            {
                int j_S, j_L, k_L;
                double Energy_min = 0.0;
                double field_local[dim_S];
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    field_local[j_S] = 0.0;
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        for (k_L=0; k_L<2; k_L++)
                        {
                            #ifdef RANDOM_BOND
                            field_local[j_S] = field_local[j_S] - (dev_J[j_L] + dev_J_random[2*dim_L*xyzi + 2*j_L + k_L]) * dev_spin[dim_S*dev_N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                            #else
                            field_local[j_S] = field_local[j_S] - dev_J[j_L] * dev_spin[dim_S*dev_N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                            #endif
                        }
                    }
                    #ifdef RANDOM_FIELD
                    field_local[j_S] = field_local[j_S] - (dev_h[j_S] + dev_h_random[dim_S*xyzi + j_S]);
                    #else
                    field_local[j_S] = field_local[j_S] - dev_h[j_S];
                    #endif
                    Energy_min = Energy_min + field_local[j_S] * field_local[j_S];
                }
                // if(Energy_min==0)
                // {
                //     for (j_S=0; j_S<dim_S; j_S++)
                //     {
                //         dev_spin_temp[dim_S*xyzi + j_S] = dev_spin[dim_S*xyzi + j_S];
                //     }
                // }
                // else // if(Energy_min>0)
                // {
                //     // Energy_min = - __dsqrt_rn(Energy_min);
                //     Energy_min = - sqrt(Energy_min);

                //     for (j_S=0; j_S<dim_S; j_S++)
                //     {
                //         dev_spin_temp[dim_S*xyzi + j_S] = field_local[j_S] / Energy_min;
                //     }
                // }
                {
                    Energy_min = -sqrt(Energy_min);
                    double energy_bool = (double)(Energy_min>=0);
                    Energy_min += energy_bool;
                    // Energy_min = ~(Energy_min); // unary bitflip
                    for (j_S=0; j_S<dim_S; j_S++)
                    {
                        dev_spin_temp[dim_S*xyzi + j_S] = (field_local[j_S] / Energy_min) + energy_bool * dev_spin[dim_S*xyzi + j_S];
                    }
                }
            }

            // return Energy_min;
        }
    #else
        double Energy_minimum_old_XY(long int xyzi, double* __restrict__ spin_local)
        {
            int j_S, j_L, k_L;
            double Energy_min = 0.0;
            double field_local[dim_S];
            
            for (j_S=0; j_S<dim_S; j_S++)
            {
                field_local[j_S] = 0.0;
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    for (k_L=0; k_L<2; k_L++)
                    {
                        #ifdef RANDOM_BOND
                        field_local[j_S] = field_local[j_S] - (J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                        #else
                        field_local[j_S] = field_local[j_S] - J[j_L] * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                        #endif
                    }
                }
                #ifdef RANDOM_FIELD
                field_local[j_S] = field_local[j_S] - (h[j_S] + h_random[dim_S*xyzi + j_S]);
                #else
                field_local[j_S] = field_local[j_S] - h[j_S];
                #endif
                Energy_min = Energy_min + field_local[j_S] * field_local[j_S];
            }
            if(Energy_min==0)
            {
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    spin_local[j_S] = spin[dim_S*xyzi + j_S];
                }
            }
            else
            {
                Energy_min = -sqrt(Energy_min);
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    spin_local[j_S] = field_local[j_S] / Energy_min;
                }
            }

            return Energy_min;
        }
        
        double Energy_minimum_old_XY_temp(long int xyzi)
        {
            int j_S, j_L, k_L;
            double Energy_min = 0.0;
            double field_local[dim_S];
            
            for (j_S=0; j_S<dim_S; j_S++)
            {
                field_local[j_S] = 0.0;
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    for (k_L=0; k_L<2; k_L++)
                    {
                        #ifdef RANDOM_BOND
                        field_local[j_S] = field_local[j_S] - (J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                        #else
                        field_local[j_S] = field_local[j_S] - J[j_L] * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                        #endif
                    }
                }
                #ifdef RANDOM_FIELD
                field_local[j_S] = field_local[j_S] - (h[j_S] + h_random[dim_S*xyzi + j_S]);
                #else
                field_local[j_S] = field_local[j_S] - h[j_S];
                #endif
                Energy_min = Energy_min + field_local[j_S] * field_local[j_S];
            }
            if(Energy_min==0)
            {
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    spin_temp[dim_S*xyzi + j_S] = spin[dim_S*xyzi + j_S];
                }
            }
            else
            {
                Energy_min = -sqrt(Energy_min);
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    spin_temp[dim_S*xyzi + j_S] = field_local[j_S] / Energy_min;
                }
            }

            return Energy_min;
        }
    #endif

    double Energy_minimum_new_XY_temp(long int xyzi)
    {
        int j_S, j_L, k_L;
        double Energy_min = 0.0;
        double field_local[dim_S];
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            field_local[j_S] = 0.0;
            for (j_L=0; j_L<dim_L; j_L++)
            {
                for (k_L=0; k_L<2; k_L++)
                {
                    #ifdef RANDOM_BOND
                    field_local[j_S] = field_local[j_S] - (J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                    #else
                    field_local[j_S] = field_local[j_S] - J[j_L] * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                    #endif
                }
            }
            #ifdef RANDOM_FIELD
            field_local[j_S] = field_local[j_S] - (h[j_S] + h_random[dim_S*xyzi + j_S]);
            #else
            field_local[j_S] = field_local[j_S] - h[j_S];
            #endif
            Energy_min = Energy_min + field_local[j_S] * field_local[j_S];
        }
        if(Energy_min==0)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin_temp[dim_S*xyzi + j_S] = spin[dim_S*xyzi + j_S];
            }
        }
        else
        {
            Energy_min = -sqrt(Energy_min);
            double spin_mod = 0;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin_temp[dim_S*xyzi + j_S] = field_local[j_S] / Energy_min;
                spin_temp[dim_S*xyzi + j_S] = spin_temp[dim_S*xyzi + j_S] + spin[dim_S*xyzi + j_S]/16;
                spin_mod = spin_mod + spin_temp[dim_S*xyzi + j_S] * spin_temp[dim_S*xyzi + j_S];
            }
            spin_mod = sqrt(spin_mod) ;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin_temp[j_S] = spin_temp[j_S] / spin_mod;
            }
        }

        return Energy_min;
    }

    double Energy_minimum_new_XY(long int xyzi, double* __restrict__ spin_local)
    {
        int j_S, j_L, k_L;
        double Energy_min = 0.0;
        double field_local[dim_S];
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            field_local[j_S] = 0.0;
            for (j_L=0; j_L<dim_L; j_L++)
            {
                for (k_L=0; k_L<2; k_L++)
                {
                    #ifdef RANDOM_BOND
                    field_local[j_S] = field_local[j_S] - (J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                    #else
                    field_local[j_S] = field_local[j_S] - J[j_L] * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                    #endif
                }
            }
            #ifdef RANDOM_FIELD
            field_local[j_S] = field_local[j_S] - (h[j_S] + h_random[dim_S*xyzi + j_S]);
            #else
            field_local[j_S] = field_local[j_S] - h[j_S];
            #endif
            Energy_min = Energy_min + field_local[j_S] * field_local[j_S];
        }
        if(Energy_min==0)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin_local[j_S] = spin[dim_S*xyzi + j_S];
            }
        }
        else
        {
            Energy_min = -sqrt(Energy_min);
            double spin_mod = 0;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin_local[j_S] = field_local[j_S] / Energy_min;
                spin_local[j_S] = 2*spin_local[j_S] + spin[dim_S*xyzi + j_S];
                spin_mod = spin_mod + spin_local[j_S] * spin_local[j_S];
            }
            spin_mod = sqrt(spin_mod) ;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin_local[j_S] = spin_local[j_S] / spin_mod;
            }
        }

        return Energy_min;
    }

    #ifdef enable_CUDA_CODE
        __global__ void update_to_minimum_checkerboard(long int sites, int* cutoff_bool, int black_or_white)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            // long int stride = blockDim.x*gridDim.x;
            long int xyzi = dev_black_white_checkerboard[dev_no_of_black_white_sites[black_or_white]*black_or_white + index];
            
            if (index < sites)
            {
                int j_S, j_L, k_L;
                double Energy_min = 0.0;
                double field_local[dim_S];
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    field_local[j_S] = 0.0;
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        for (k_L=0; k_L<2; k_L++)
                        {
                            #ifdef RANDOM_BOND
                            field_local[j_S] = field_local[j_S] - (dev_J[j_L] + dev_J_random[2*dim_L*xyzi + 2*j_L + k_L]) * dev_spin[dim_S*dev_N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                            #else
                            field_local[j_S] = field_local[j_S] - dev_J[j_L] * dev_spin[dim_S*dev_N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                            #endif
                        }
                    }
                    #ifdef RANDOM_FIELD
                    field_local[j_S] = field_local[j_S] - (dev_h[j_S] + dev_h_random[dim_S*xyzi + j_S]);
                    #else
                    field_local[j_S] = field_local[j_S] - dev_h[j_S];
                    #endif
                    Energy_min = Energy_min + field_local[j_S] * field_local[j_S];
                }
                // if(Energy_min==0)
                // {
                //     for (j_S=0; j_S<dim_S; j_S++)
                //     {
                //         dev_spin_temp[dim_S*xyzi + j_S] = dev_spin[dim_S*xyzi + j_S];
                //     }
                // }
                // else // if(Energy_min>0)
                // {
                //     // Energy_min = - __dsqrt_rn(Energy_min);
                //     Energy_min = - sqrt(Energy_min);

                //     for (j_S=0; j_S<dim_S; j_S++)
                //     {
                //         dev_spin_temp[dim_S*xyzi + j_S] = field_local[j_S] / Energy_min;
                //     }
                // }
                {
                    Energy_min = -sqrt(Energy_min);
                    double energy_bool = (double)(Energy_min>=0);
                    Energy_min += energy_bool;
                    // Energy_min = ~(Energy_min); // unary bitflip
                    for (j_S=0; j_S<dim_S; j_S++)
                    {
                        dev_spin_temp[dim_S*xyzi + j_S] = (field_local[j_S] / Energy_min) + energy_bool * dev_spin[dim_S*xyzi + j_S];
                    }
                }
                // CUTOFF CHECK
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    // double diff_spin = fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]);

                    // if ( diff_spin > dev_CUTOFF_SPIN[0] )
                    // if ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] )
                    // {
                    //     atomicOr(cutoff_bool, 1);
                    //     // cutoff_bool[0] = 1;
                    // }
                    int temp = cutoff_bool[0] || atomicOr(cutoff_bool, (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0]) );
                    // atomicAdd(&cutoff_bool[0], (double) (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] ));
                    // cutoff_bool[0] += (double) ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] );

                    dev_spin[dim_S*xyzi + j_S] = dev_spin_temp[dim_S*xyzi + j_S];
                }
            }

        }

        __global__ void update_to_minimum_checkerboard_cutoff_check(long int sites, int* cutoff_bool, int black_or_white)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            // long int stride = blockDim.x*gridDim.x;
            long int xyzi = dev_black_white_checkerboard[dev_no_of_black_white_sites[black_or_white]*black_or_white + index];
            
            if (index < sites)
            {
                int j_S, j_L, k_L;
                double Energy_min = 0.0;
                double field_local[dim_S];
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    field_local[j_S] = 0.0;
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        for (k_L=0; k_L<2; k_L++)
                        {
                            #ifdef RANDOM_BOND
                            field_local[j_S] = field_local[j_S] - (dev_J[j_L] + dev_J_random[2*dim_L*xyzi + 2*j_L + k_L]) * dev_spin[dim_S*dev_N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                            #else
                            field_local[j_S] = field_local[j_S] - dev_J[j_L] * dev_spin[dim_S*dev_N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                            #endif
                        }
                    }
                    #ifdef RANDOM_FIELD
                    field_local[j_S] = field_local[j_S] - (dev_h[j_S] + dev_h_random[dim_S*xyzi + j_S]);
                    #else
                    field_local[j_S] = field_local[j_S] - dev_h[j_S];
                    #endif
                    Energy_min = Energy_min + field_local[j_S] * field_local[j_S];
                }
                // if(Energy_min==0)
                // {
                //     for (j_S=0; j_S<dim_S; j_S++)
                //     {
                //         dev_spin_temp[dim_S*xyzi + j_S] = dev_spin[dim_S*xyzi + j_S];
                //     }
                // }
                // else // if(Energy_min>0)
                // {
                //     // Energy_min = - __dsqrt_rn(Energy_min);
                //     Energy_min = - sqrt(Energy_min);

                //     for (j_S=0; j_S<dim_S; j_S++)
                //     {
                //         dev_spin_temp[dim_S*xyzi + j_S] = field_local[j_S] / Energy_min;
                //     }
                // }
                {
                    Energy_min = -sqrt(Energy_min);
                    double energy_bool = (double)(Energy_min>=0);
                    Energy_min += energy_bool;
                    // Energy_min = ~(Energy_min); // unary bitflip
                    for (j_S=0; j_S<dim_S; j_S++)
                    {
                        dev_spin_temp[dim_S*xyzi + j_S] = (field_local[j_S] / Energy_min) + energy_bool * dev_spin[dim_S*xyzi + j_S];
                    }
                }
                // CUTOFF CHECK
                double delta_S = 0.0;
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    delta_S += (dev_spin_temp[xyzi*dim_S+j_S] - dev_spin_bkp[xyzi*dim_S+j_S])*(dev_spin_temp[xyzi*dim_S+j_S] - dev_spin_bkp[xyzi*dim_S+j_S]);
                    // double diff_spin = fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]);

                    // if ( diff_spin > dev_CUTOFF_SPIN[0] )
                    // if ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] )
                    // {
                    //     atomicOr(cutoff_bool, 1);
                    //     // cutoff_bool[0] = 1;
                    // }
                    int temp = cutoff_bool[0] || atomicOr(cutoff_bool, (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0]) );
                    // atomicAdd(&cutoff_bool[0], (double) (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] ));
                    // cutoff_bool[0] += (double) ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] );

                    dev_spin[xyzi*dim_S+j_S] = dev_spin_temp[xyzi*dim_S+j_S];
                }

                // if ( delta_S > dev_CUTOFF_S_SQ[0] )
                // {
                //     atomicOr(cutoff_bool+1, 1);
                //     // cutoff_bool[1] = 1;
                // }
                int temp = cutoff_bool[1] || atomicOr(cutoff_bool+1, (delta_S > dev_CUTOFF_S_SQ[0]) );
                // atomicAdd(&cutoff_bool[1], (double) ( delta_S > dev_CUTOFF_S_SQ[0] ));
                // cutoff_bool[1] += (double) ( delta_S > dev_CUTOFF_S_SQ[0] );
            }
        }

        __global__ void update_to_minimum_checkerboard_cutoff_continue(long int sites, int* cutoff_bool, int black_or_white)
        {
            long int index = threadIdx.x + blockIdx.x*blockDim.x;
            // long int stride = blockDim.x*gridDim.x;
            long int xyzi = dev_black_white_checkerboard[dev_no_of_black_white_sites[black_or_white]*black_or_white + index];
            
            if (index < sites)
            {
                int j_S, j_L, k_L;
                double Energy_min = 0.0;
                double field_local[dim_S];
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    field_local[j_S] = 0.0;
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        for (k_L=0; k_L<2; k_L++)
                        {
                            #ifdef RANDOM_BOND
                            field_local[j_S] = field_local[j_S] - (dev_J[j_L] + dev_J_random[2*dim_L*xyzi + 2*j_L + k_L]) * dev_spin[dim_S*dev_N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                            #else
                            field_local[j_S] = field_local[j_S] - dev_J[j_L] * dev_spin[dim_S*dev_N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                            #endif
                        }
                    }
                    #ifdef RANDOM_FIELD
                    field_local[j_S] = field_local[j_S] - (dev_h[j_S] + dev_h_random[dim_S*xyzi + j_S]);
                    #else
                    field_local[j_S] = field_local[j_S] - dev_h[j_S];
                    #endif
                    Energy_min = Energy_min + field_local[j_S] * field_local[j_S];
                }
                // if(Energy_min==0)
                // {
                //     for (j_S=0; j_S<dim_S; j_S++)
                //     {
                //         dev_spin_temp[dim_S*xyzi + j_S] = dev_spin[dim_S*xyzi + j_S];
                //     }
                // }
                // else // if(Energy_min>0)
                // {
                //     // Energy_min = - __dsqrt_rn(Energy_min);
                //     Energy_min = - sqrt(Energy_min);

                //     for (j_S=0; j_S<dim_S; j_S++)
                //     {
                //         dev_spin_temp[dim_S*xyzi + j_S] = field_local[j_S] / Energy_min;
                //     }
                // }
                {
                    Energy_min = -sqrt(Energy_min);
                    double energy_bool = (double)(Energy_min>=0);
                    Energy_min += energy_bool;
                    // Energy_min = ~(Energy_min); // unary bitflip
                    for (j_S=0; j_S<dim_S; j_S++)
                    {
                        dev_spin_temp[dim_S*xyzi + j_S] = (field_local[j_S] / Energy_min) + energy_bool * dev_spin[dim_S*xyzi + j_S];
                    }
                }
                // CUTOFF CHECK
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    
                    // double diff_spin = fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]);

                    // if ( diff_spin > dev_CUTOFF_SPIN[0] )
                    // if ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] )
                    // {
                    //     atomicOr(cutoff_bool, 1);
                    //     // cutoff_bool[0] = 1;
                    // }
                    int temp = cutoff_bool[0] || atomicOr(cutoff_bool, (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0]) );
                    // atomicAdd(&cutoff_bool[0], (double) (fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] ));
                    // cutoff_bool[0] += ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] );
                    
                    dev_spin[xyzi*dim_S+j_S] = dev_spin_temp[xyzi*dim_S+j_S];
                }
                
            }
        }
    #else
        double update_to_minimum_checkerboard(long int xyzi, double* __restrict__ spin_local)
        {
            int j_S;
            
            Energy_minimum_old_XY(xyzi, spin_local);

            double spin_diff_abs = 0.0;

            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin_diff_abs += fabs(spin[dim_S*xyzi + j_S] - spin_local[j_S]);
            }
            update_spin_single(xyzi, spin_local);
            return spin_diff_abs;
        }
    #endif
    
    #ifdef CUTOFF_BY_MAX
    #ifdef enable_CUDA_CODE
        int find_change()
        {
            static int print_first = 0;
            if (print_first == 0)
            {
                printf("\n Using find change max.. \n");
                print_first = !print_first;
            }
            int cutoff_local;
            int* dev_cutoff_local;
            // double* dev_spin_temp;
            cudaMalloc(&dev_cutoff_local, sizeof(int));
            // cudaMallocManaged(&dev_spin_temp, dim_S*no_of_sites*sizeof(double));
            long int site_i;
            static int black_or_white = BLACK_WHITE;

            // if (update_all_or_checker == 0)
            #ifdef UPDATE_ALL_NON_EQ
            {
                cutoff_local = 0;
                // cudaMemcpy(dev_cutoff_local, &cutoff_local, sizeof(int), cudaMemcpyHostToDevice);
                cudaMemset(dev_cutoff_local, 0, sizeof(int));
                
                Energy_minimum_old_XY<<< no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites);
                // Energy_minimum_old_XY<<< 1, no_of_sites >>>(no_of_sites);
                cudaDeviceSynchronize();

                update_spin_all<<< no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites, dev_cutoff_local);
                // update_spin_all<<< 1, no_of_sites >>>(no_of_sites, dev_cutoff_local);
                cudaDeviceSynchronize();

                cudaMemcpy(&cutoff_local, dev_cutoff_local, sizeof(int), cudaMemcpyDeviceToHost);
            }
            #endif
            // else
            #ifdef UPDATE_CHKR_NON_EQ
            {
                cutoff_local = 0;
                // cudaMemcpy(dev_cutoff_local, &cutoff_local, sizeof(double), cudaMemcpyHostToDevice);
                cudaMemset(dev_cutoff_local, 0, sizeof(int));
                update_to_minimum_checkerboard<<< no_of_black_white_sites[0]/gpu_threads + 1, gpu_threads >>>(no_of_black_white_sites[BLACK_WHITE], dev_cutoff_local, BLACK_WHITE);
                cudaDeviceSynchronize();
                
                update_to_minimum_checkerboard<<< no_of_black_white_sites[0]/gpu_threads + 1, gpu_threads >>>(no_of_black_white_sites[!BLACK_WHITE], dev_cutoff_local, !BLACK_WHITE);
                cudaDeviceSynchronize();

                cudaMemcpy(&cutoff_local, dev_cutoff_local, sizeof(int), cudaMemcpyDeviceToHost);
            }
            #endif
            cudaFree(dev_cutoff_local);
            // cudaFree(dev_spin_temp);
            return cutoff_local;
        }
    #else
        int find_change()
        {
            static int print_first = 0;
            if (print_first == 0)
            {
                printf("\n Using find change max.. \n");
                print_first = !print_first;
            }
            int cutoff_local;
            
            long int site_i;
            static int black_or_white = BLACK_WHITE;

            // if (update_all_or_checker == 0)
            #ifdef UPDATE_ALL_NON_EQ
            {
                cutoff_local = 0;

                #pragma omp parallel 
                {
                    #pragma omp for
                    for (site_i=0; site_i<no_of_sites; site_i++)
                    {
                        Energy_minimum_old_XY(site_i, &spin_temp[dim_S*site_i + 0]);
                    }
                    #pragma omp for reduction(||:cutoff_local)
                    for (site_i=0; site_i<no_of_sites*dim_S; site_i++)
                    {
                        // if ( fabs(spin[site_i] - spin_temp[site_i]) > CUTOFF_SPIN )
                        // {
                        //     cutoff_local = 1;
                        // }
                        // cutoff_local += ( fabs(spin[site_i] - spin_temp[site_i]) > CUTOFF_SPIN );
                        cutoff_local = cutoff_local || ( fabs(spin[site_i] - spin_temp[site_i]) > CUTOFF_SPIN );
                        
                        spin[site_i] = spin_temp[site_i];
                    }
                }
            }
            #endif
            // else
            #ifdef UPDATE_CHKR_NON_EQ
            {
                cutoff_local = 0;
                
                #pragma omp parallel 
                {
                    #pragma omp for reduction(|:cutoff_local)
                    for (site_i=0; site_i<no_of_black_white_sites[black_or_white]; site_i++)
                    {
                        long int site_index = black_white_checkerboard[no_of_black_white_sites[black_or_white]*black_or_white + site_i];
                        double spin_local[dim_S];
                        
                        // if ( update_to_minimum_checkerboard(site_index, spin_local) > CUTOFF_SPIN )
                        // {
                        //     cutoff_local = 1;
                        // }
                        // cutoff_local += ( update_to_minimum_checkerboard(site_index, spin_local) > CUTOFF_SPIN );
                        cutoff_local = cutoff_local | ( update_to_minimum_checkerboard(site_index, spin_local) > CUTOFF_SPIN );
                        
                    }

                    
                    #pragma omp for reduction(|:cutoff_local)
                    for (site_i=0; site_i<no_of_black_white_sites[!black_or_white]; site_i++)
                    {
                        long int site_index = black_white_checkerboard[no_of_black_white_sites[!black_or_white]*(!black_or_white) + site_i];
                        double spin_local[dim_S];
                        
                        // if ( update_to_minimum_checkerboard(site_index, spin_local) > CUTOFF_SPIN )
                        // {
                        //     cutoff_local = 1;
                        // }
                        // cutoff_local += ( update_to_minimum_checkerboard(site_index, spin_local) > CUTOFF_SPIN );
                        cutoff_local = cutoff_local | ( update_to_minimum_checkerboard(site_index, spin_local) > CUTOFF_SPIN );
                        
                    }            
                }
            }
            #endif
            // printf("\r%d ", cutoff_local);
            return cutoff_local;
        }
    #endif
    #endif

    #ifdef CUTOFF_BY_SUM
        int find_change()
        {
            static int print_first = 0;
            if (print_first == 0)
            {
                printf("\n Using find change sum.. \n");
                print_first = !print_first;
            }
            int cutoff_local;
            double cutoff_change;
            
            long int site_i;
            static int black_or_white = BLACK_WHITE;

            // if (update_all_or_checker == 0)
            #ifdef UPDATE_ALL_NON_EQ
            {
                cutoff_change = 0.0;

                #pragma omp parallel 
                {
                    #pragma omp for
                    for (site_i=0; site_i<no_of_sites; site_i++)
                    {
                        Energy_minimum_new_XY(site_i, &spin_temp[dim_S*site_i + 0]);
                    }
                    #pragma omp for reduction(+:cutoff_change)
                    for (site_i=0; site_i<no_of_sites*dim_S; site_i++)
                    {
                        cutoff_change += fabs(spin[site_i] - spin_temp[site_i]);
                        spin[site_i] = spin_temp[site_i];
                    }
                }
            }
            #endif
            // else
            #ifdef UPDATE_CHKR_NON_EQ
            {
                cutoff_change = 0.0;
                
                #pragma omp parallel 
                {
                    #pragma omp for reduction(+:cutoff_change)
                    for (site_i=0; site_i<no_of_black_white_sites[black_or_white]; site_i++)
                    {
                        long int site_index = black_white_checkerboard[no_of_black_white_sites[black_or_white]*black_or_white + site_i];
                        double spin_local[dim_S];

                        cutoff_change += update_to_minimum_checkerboard(site_index, spin_local);
                    }

                    #pragma omp for reduction(+:cutoff_change)
                    for (site_i=0; site_i<no_of_black_white_sites[!black_or_white]; site_i++)
                    {
                        long int site_index = black_white_checkerboard[no_of_black_white_sites[!black_or_white]*(!black_or_white) + site_i];
                        double spin_local[dim_S];

                        cutoff_change += update_to_minimum_checkerboard(site_index, spin_local);
                    }            
                }
            }
            #endif
            if (cutoff_change > CUTOFF_SPIN)
            {
                cutoff_local = 1;
            }
            else
            {
                cutoff_local = 0;
            }

            return cutoff_local;
        }
    #endif
    
    int save_to_file(double h_text, double delta_text, int jj_S, double delta_m, char text[], double* h_init, int sweep_or_loop)
    {
        int j_S;
        
        char str_prep[128], str_app[128];
        char *pos_prep = str_prep;
        char *pos_app = str_app;
        pos_prep += sprintf(pos_prep, "%.17e\t", h_text);
        pos_app += sprintf(pos_app, "%.17e\t", delta_m);
        pFile_output = pFile_1;
        output_data("output_file.dat", str_prep, str_app);
        pFile_output = NULL;
        /* fprintf(pFile_1, "%.12e\t", h_text);

        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", h[j_S]);
        }
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", m[j_S]);
        }
        
        #ifdef CHECK_AVALANCHE
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", delta_S_abs[j_S]);
        }
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", delta_S_squared[j_S]);
        }
        // fprintf(pFile_1, "%.12e\t", delta_S_max);
        #else
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", delta_M[j_S]);
        }
        // for(j_S=0; j_S<dim_S; j_S++)
        // {
        //     fprintf(pFile_1, "%.12e\t", delta_M[j_S]*delta_M[j_S]);
        // }
        // fprintf(pFile_1, "%.12e\t", delta_m);
        #endif
        fprintf(pFile_1, "%.12e\t", delta_m); // fprintf(pFile_1, "%.12e\t", delta_S_max);
        #ifdef EXPLORE_ENERGY_LANDSCAPE
        fprintf(pFile_1, "%.12e\t", E);
        fprintf(pFile_1, "%.12e\t", delta_E);
        #endif
        fprintf(pFile_1, "\n"); */
        
        #ifdef SAVE_SPIN_AFTER
            
            if ( h_counter % SAVE_SPIN_AFTER == 0 )
            {
                double temp_h[dim_S];
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    temp_h[j_S] = h[j_S];
                    if (sweep_or_loop == 1)
                    {
                        h[j_S] = 0.0;
                    }
                }
                h[jj_S] = h_init[0];

                char append_string[128];
                char *pos = append_string;
                if (sweep_or_loop == 1)
                {
                    pos += sprintf(pos, "_step_%d_%ld", repeat_loop, h_counter);
                }
                else
                {
                    pos += sprintf(pos, "_step_%d_%ld", repeat_sweep, h_counter);
                }
                if (dim_L==2 && dim_S>=2)
                {
                    save_spin_config(append_string, "a", 0);
                }
                else if (dim_S==1)
                {
                    save_spin_config(append_string, "a", 2);
                }
                    
                
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    h[j_S] = temp_h[j_S];
                }
            }
            h_counter++;
        #endif

        #ifdef PRINT_OUTPUT
        // if(reqd_to_print == 1)
        // {
            // printf(  "\n============================\n");
            printf(  "\r=1= %s = %.3e ", text, h_text );
            #ifndef CHECK_AVALANCHE
            printf(    ", d_m = %.3e ", delta_m );
            #else
            printf(    ", d_S = %.3e ", delta_m );
            #endif
            #ifdef EXPLORE_ENERGY_LANDSCAPE
            printf(    ", d_E = %.3e ", delta_E );
            #endif
            printf(    ", d_%s = %.3e ", text, order[jj_S]*delta_text );
            printf(    " -[ backup ]-  ");
            printf(    " Time = %.3e s | ", get_time_if_parallel() - start_time );
            fflush(stdout);
            // printf(  "\n============================\n");
        // }
        #endif

        return 0;
    }
    
    #ifdef enable_CUDA_CODE
        __global__ void convert_delta_S_max_to_cutoff(double* cutoff_bool)
        {
            
            if (dev_delta_spin_max_reduce[0] >= dev_CUTOFF_S_SQ[0])
            {
                cutoff_bool[1] = 1;
            }
            else
            {
                cutoff_bool[1] = 0;
            }
            return;
        }

        int check_avalanche()
        {
            cutoff_check[0] = 0;
            cutoff_check[1] = 0;

            int* dev_cutoff_check;
            cudaMalloc(&dev_cutoff_check, 2*sizeof(int));
            int j_S;
            long int site_i;
            static int black_or_white = BLACK_WHITE;
            do
            {
                cudaMemset(dev_cutoff_check, 0, 2*sizeof(int));

                // cutoff_check[0] = 0;
                // cutoff_check[1] = 0;

                // if (update_all_or_checker == 0)
                #ifdef UPDATE_ALL_NON_EQ
                {
                    // cutoff_check[0] = 0;
                    Energy_minimum_old_XY<<< no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites);
                    // Energy_minimum_old_XY<<< 1, no_of_sites >>>(no_of_sites);
                    cudaDeviceSynchronize();
                    
                    update_spin_all_cutoff_check<<< no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites, dev_cutoff_check);
                    // update_spin_all<<< 1, no_of_sites >>>(no_of_sites, dev_cutoff_local);
                    cudaDeviceSynchronize();

                    cudaMemcpy(cutoff_check, dev_cutoff_check, 2*sizeof(int), cudaMemcpyDeviceToHost);
                }
                #endif
                // else
                #ifdef UPDATE_CHKR_NON_EQ
                {
                    // cutoff_check[0] = 0;
                    update_to_minimum_checkerboard_cutoff_continue<<< no_of_black_white_sites[0]/gpu_threads + 1, gpu_threads >>>(no_of_black_white_sites[BLACK_WHITE], dev_cutoff_check, BLACK_WHITE);
                    cudaDeviceSynchronize();
                    
                    update_to_minimum_checkerboard_cutoff_continue<<< no_of_black_white_sites[1]/gpu_threads + 1, gpu_threads >>>(no_of_black_white_sites[!BLACK_WHITE], dev_cutoff_check, !BLACK_WHITE);
                    cudaDeviceSynchronize();

                    cudaMemcpy(cutoff_check, dev_cutoff_check, 2*sizeof(int), cudaMemcpyDeviceToHost);
                    
                    
                }
                #endif
            }
            while ( cutoff_check[0] > 0 && cutoff_check[1] == 0 );
            cudaFree(dev_cutoff_check);
            return 0;
        }

        int continue_avalanche()
        {
            cutoff_check[0] = 0;
            cutoff_check[1] = 0;

            int* dev_cutoff_check;
            cudaMalloc(&dev_cutoff_check, 2*sizeof(int));
            int j_S;
            long int site_i;
            static int black_or_white = BLACK_WHITE;
            do
            {
                cudaMemset(dev_cutoff_check, 0, 2*sizeof(int));

                // cutoff_check[0] = 0;
                // cutoff_check[1] = 0;

                // if (update_all_or_checker == 0)
                #ifdef UPDATE_ALL_NON_EQ
                {
                    // cutoff_check[0] = 0;
                    Energy_minimum_old_XY<<< no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites);
                    // Energy_minimum_old_XY<<< 1, no_of_sites >>>(no_of_sites);
                    cudaDeviceSynchronize();

                    update_spin_all_cutoff_continue<<< no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites, dev_cutoff_check);
                    // update_spin_all<<< 1, no_of_sites >>>(no_of_sites, dev_cutoff_local);
                    cudaDeviceSynchronize();

                    cudaMemcpy(cutoff_check, dev_cutoff_check, 2*sizeof(int), cudaMemcpyDeviceToHost);
                }
                #endif
                // else
                #ifdef UPDATE_CHKR_NON_EQ
                {
                    // cutoff_check[0] = 0;
                    update_to_minimum_checkerboard_cutoff_continue<<< no_of_black_white_sites[0]/gpu_threads + 1, gpu_threads >>>(no_of_black_white_sites[BLACK_WHITE], dev_cutoff_check, BLACK_WHITE);
                    cudaDeviceSynchronize();
                    
                    update_to_minimum_checkerboard_cutoff_continue<<< no_of_black_white_sites[1]/gpu_threads + 1, gpu_threads >>>(no_of_black_white_sites[!BLACK_WHITE], dev_cutoff_check, !BLACK_WHITE);
                    cudaDeviceSynchronize();

                    cudaMemcpy(cutoff_check, dev_cutoff_check, 2*sizeof(int), cudaMemcpyDeviceToHost);
                    
                    
                }
                #endif
            }
            while ( cutoff_check[0] > 0 );
            cudaFree(dev_cutoff_check);
            return 0;
        }
    #else
        int check_avalanche()
        {
            cutoff_check[0] = 0;
            cutoff_check[1] = 0;
            double cutoff_check_0 = 0;
            double cutoff_check_1 = 0;

            
            int j_S;
            long int site_i;
            static int black_or_white = BLACK_WHITE;
            do
            {
                cutoff_check_0 = 0;
                cutoff_check_1 = 0;

                // if (update_all_or_checker == 0)
                #ifdef UPDATE_ALL_NON_EQ
                {
                    // cutoff_check[0] = 0;

                    #pragma omp parallel 
                    {
                        #pragma omp for
                        for (site_i=0; site_i<no_of_sites; site_i++)
                        {
                            Energy_minimum_new_XY_temp(site_i);
                        }
                        
                        #pragma omp for private(j_S) reduction(+:cutoff_check_0,cutoff_check_1)
                        for (site_i=0; site_i<no_of_sites; site_i++)
                        {
                            double delta_S = 0.0;
                            for (j_S=0; j_S<dim_S; j_S++)
                            {
                                delta_S += (spin_temp[site_i*dim_S+j_S] - spin_bkp[site_i*dim_S+j_S])*(spin_temp[site_i*dim_S+j_S] - spin_bkp[site_i*dim_S+j_S]);
                                // if ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN )
                                // {
                                //     cutoff_check[0] = 1;
                                // }
                                // cutoff_check[0] += ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN );
                                cutoff_check_0 = cutoff_check_0 || ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN );
                                // cutoff_check[0] += (!cutoff_check[0]) * ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN );

                                spin[site_i*dim_S+j_S] = spin_temp[site_i*dim_S+j_S];
                            }
                            // if( delta_S > CUTOFF_S_SQ )
                            // {
                            //     cutoff_check[1] = 1;
                            // }
                            // cutoff_check[1] += ( delta_S > CUTOFF_S_SQ );
                            cutoff_check_1 = cutoff_check_1 || ( delta_S > CUTOFF_S_SQ );
                            // cutoff_check[1] += (!cutoff_check[1]) * ( delta_S > CUTOFF_S_SQ );
                        }
                    }
                    cutoff_check[0] = cutoff_check_0;
                    cutoff_check[1] = cutoff_check_1;
                }
                #endif
                // else
                #ifdef UPDATE_CHKR_NON_EQ
                {
                    #pragma omp parallel 
                    {
                        // cutoff_check[0] = 0;

                        #pragma omp for private(j_S) reduction(||:cutoff_check_0,cutoff_check_1)
                        for (site_i=0; site_i<no_of_black_white_sites[black_or_white]; site_i++)
                        {
                            long int site_index = black_white_checkerboard[no_of_black_white_sites[black_or_white]*black_or_white + site_i];
                            // double spin_local[dim_S];
                            Energy_minimum_old_XY_temp(site_index);

                            double delta_S = 0.0;
                            for (j_S=0; j_S<dim_S; j_S++)
                            {
                                delta_S += (spin_temp[site_index*dim_S+j_S] - spin_bkp[site_index*dim_S+j_S])*(spin_temp[site_index*dim_S+j_S] - spin_bkp[site_index*dim_S+j_S]);
                                // if ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN )
                                // {
                                //     cutoff_check[0] = 1;
                                // }
                                // cutoff_check[0] += ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                                cutoff_check_0 = cutoff_check_0 || ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                                // cutoff_check[0] += (!cutoff_check[0]) * ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );

                                spin[site_index*dim_S+j_S] = spin_temp[site_index*dim_S+j_S];
                            }
                            // if( delta_S > CUTOFF_S_SQ )
                            // {
                            //     cutoff_check[1] = 1;
                            // }
                            // cutoff_check[1] += ( delta_S > CUTOFF_S_SQ );
                            cutoff_check_1 = cutoff_check_1 || ( delta_S > CUTOFF_S_SQ );
                            // cutoff_check[1] += (!cutoff_check[1]) * ( delta_S > CUTOFF_S_SQ );
                        }

                        #pragma omp for private(j_S) reduction(||:cutoff_check_0,cutoff_check_1)
                        for (site_i=0; site_i<no_of_black_white_sites[!black_or_white]; site_i++)
                        {
                            long int site_index = black_white_checkerboard[no_of_black_white_sites[!black_or_white]*(!black_or_white) + site_i];
                            // double spin_local[dim_S];
                            Energy_minimum_old_XY_temp(site_index);

                            double delta_S = 0.0;
                            for (j_S=0; j_S<dim_S; j_S++)
                            {
                                delta_S += (spin_temp[site_index*dim_S+j_S] - spin_bkp[site_index*dim_S+j_S])*(spin_temp[site_index*dim_S+j_S] - spin_bkp[site_index*dim_S+j_S]);
                                // if ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN )
                                // {
                                //     cutoff_check[0] = 1;
                                // }
                                // cutoff_check[0] += ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                                cutoff_check_0 = cutoff_check_0 || ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                                // cutoff_check[0] += (!cutoff_check[0]) * ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );

                                spin[site_index*dim_S+j_S] = spin_temp[site_index*dim_S+j_S];
                            }
                            // if( delta_S > CUTOFF_S_SQ )
                            // {
                            //     cutoff_check[1] = 1;
                            // }
                            // cutoff_check[1] += ( delta_S > CUTOFF_S_SQ );
                            cutoff_check_1 = cutoff_check_1 || ( delta_S > CUTOFF_S_SQ );
                            // cutoff_check[1] += (!cutoff_check[1]) * ( delta_S > CUTOFF_S_SQ );
                        }         
                    }
                    cutoff_check[0] = cutoff_check_0;
                    cutoff_check[1] = cutoff_check_1;
                }
                #endif
            }
            while ( cutoff_check[0] > 0 && cutoff_check[1] == 0 );

            return 0;
        }

        int continue_avalanche()
        {
            cutoff_check[0] = 0;
            cutoff_check[1] = 0;
            double cutoff_check_0 = 0;
            double cutoff_check_1 = 0;


            
            int j_S;
            long int site_i;
            static int black_or_white = BLACK_WHITE;
            do
            {
                cutoff_check_0 = 0;
                // cutoff_check_1 = 0;

                // if (update_all_or_checker == 0)
                #ifdef UPDATE_ALL_NON_EQ
                {
                    // cutoff_check[0] = 0;

                    #pragma omp parallel 
                    {
                        #pragma omp for 
                        for (site_i=0; site_i<no_of_sites; site_i++)
                        {
                            Energy_minimum_new_XY_temp(site_i);
                        }
                        #pragma omp for private(j_S) reduction(||:cutoff_check_0)
                        for (site_i=0; site_i<no_of_sites; site_i++)
                        {
                            // double delta_S = 0.0;
                            for (j_S=0; j_S<dim_S; j_S++)
                            {
                                // if ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN )
                                // {
                                //     cutoff_check[0] = 1;
                                // }
                                // cutoff_check[0] += ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN );
                                cutoff_check_0 = cutoff_check_0 || ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN );
                                // cutoff_check[0] += (!cutoff_check[0]) * ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN );

                                spin[site_i*dim_S+j_S] = spin_temp[site_i*dim_S+j_S];
                            }
                            
                        }
                    }
                }
                #endif
                // else
                #ifdef UPDATE_CHKR_NON_EQ
                {
                    #pragma omp parallel 
                    {
                        // cutoff_check[0] = 0;

                        #pragma omp for private(j_S) reduction(||:cutoff_check_0)
                        for (site_i=0; site_i<no_of_black_white_sites[black_or_white]; site_i++)
                        {
                            long int site_index = black_white_checkerboard[no_of_black_white_sites[black_or_white]*black_or_white + site_i];
                            // double spin_local[dim_S];
                            Energy_minimum_old_XY_temp(site_index);

                            // double delta_S = 0.0;
                            for (j_S=0; j_S<dim_S; j_S++)
                            {
                                // if ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN )
                                // {
                                //     cutoff_check[0] = 1;
                                // }
                                // cutoff_check[0] += ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                                cutoff_check_0 = cutoff_check_0 || ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                                // cutoff_check[0] += (!cutoff_check[0]) * ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );

                                spin[site_index*dim_S+j_S] = spin_temp[site_index*dim_S+j_S];
                            }
                            
                        }

                        #pragma omp for private(j_S) reduction(||:cutoff_check_0)
                        for (site_i=0; site_i<no_of_black_white_sites[!black_or_white]; site_i++)
                        {
                            long int site_index = black_white_checkerboard[no_of_black_white_sites[!black_or_white]*(!black_or_white) + site_i];
                            // double spin_local[dim_S];
                            Energy_minimum_old_XY_temp(site_index);

                            // double delta_S = 0.0;
                            for (j_S=0; j_S<dim_S; j_S++)
                            {
                                // if ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN )
                                // {
                                //     cutoff_check[0] = 1;
                                // }
                                // cutoff_check[0] += ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                                cutoff_check_0 = cutoff_check_0 || ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                                // cutoff_check[0] += (!cutoff_check[0]) * ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );

                                spin[site_index*dim_S+j_S] = spin_temp[site_index*dim_S+j_S];
                            }
                            
                        }         
                    }
                    cutoff_check_0 = cutoff_check[0];
                }
                #endif
            }
            while ( cutoff_check[0] > 0 );

            return 0;
        }
    #endif

    #ifdef CHECK_AVALANCHE
        int const_delta_phi(double h_phi, double delta_phi, int jj_S, double h_start)
        {
            static long int counter = 1;
            static int sweep_or_loop = 1;

            int j_S;
            
            #ifdef OLD_FUNCTION
            if (h_phi == 0.0)
            {
                printf("\n=========================");
                printf("\n  phi = 0.0 (%.15e)  ", h_phi);
                printf("\n=========================\n");
                
                ensemble_m();
                ensemble_delta_S_squared_max();
                
                backing_up_spin();
                
                save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            ensemble_m();
            ensemble_delta_S_squared_max();

            // printf(  "\n============================\n");
            if(counter % 100 == 0)
            {
                save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);
                counter = 0;
            }
            else
            {
                save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);
            }
            // printf(  "\n============================\n");
            counter++;

            return 0;
        }
        
        int const_delta_h_axis(double h_jj_S, double delta_h, int jj_S, double h_start)
        {
            static long int counter = 1;
            static int sweep_or_loop = 2;

            int j_S;
            
            #ifdef OLD_FUNCTION
            if (h_jj_S == order[jj_S]*h_start)
            {
                printf("\n=========================");
                printf("\n  h_j = 0.0 (%.15e)  ", h_jj_S);
                printf("\n=========================\n");
                
                ensemble_m();
                ensemble_delta_S_squared_max();
                
                backing_up_spin();
                
                save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            ensemble_m();
            ensemble_delta_S_squared_max();

            // printf(  "\n============================\n");
            if(counter % 10 == 0)
            {
                save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_start, sweep_or_loop);
                counter = 0;
            }
            else
            {
                save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_start, sweep_or_loop);
            }
            // printf(  "\n============================\n");
            counter++;

            return 0;
        }
        
        int slope_subdivide_phi(double h_phi, double delta_phi, int jj_S, double h_start)
        {
            static int sweep_or_loop = 1;

            int j_S;
            
            #ifdef OLD_FUNCTION
            if (h_phi == 0.0)
            {
                printf("\n=========================");
                printf("\n  phi = 0.0 (%.15e)  ", h_phi);
                printf("\n=========================\n");
                
                ensemble_m();
                ensemble_delta_S_squared_max();
                
                backing_up_spin();
                
                save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);
                return 0;
            }
            #endif
            
            if (cutoff_check[1] == 0 && cutoff_check[0] == 0)
            {
                ensemble_m();
                ensemble_delta_S_squared_max();
                
                backing_up_spin();
                
                save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);
                
                return 0;
            }

            double h_phi_k, delta_phi_k;
            int cutoff_local;
            long int slope;
            if (cutoff_check[1] > 0)
            {
                find_delta_S_max();
                restoring_spin(h_phi, delta_phi, jj_S, delta_S_max, "phi", 1);
                // ensemble_m();
                slope = delta_S_max/CUTOFF_S_SQ + 1;
                delta_phi_k = delta_phi / (double) slope;
                if (delta_phi_k < del_phi_cutoff)
                {
                    slope = delta_phi/del_phi_cutoff + 1;
                    delta_phi_k = delta_phi / (double) slope;
                }
            }
            // else 
            // {
            //     // ensemble_E();
            //     backing_up_spin();

            //     save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);

            //     return 1;
            // }

            long int slope_i;
            for ( slope_i = 0; slope_i < slope-1; slope_i++ )
            {
                h_phi_k = h_phi - delta_phi + delta_phi_k * (double) (slope_i+1);
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*(h_phi_k));
                    h[1] = h_start * sin(2*pie*(h_phi_k));
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*(h_phi_k));
                    h[1] = h_start * cos(2*pie*(h_phi_k));
                }

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                if (delta_phi_k <= del_phi_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }


                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                slope_subdivide_phi(h_phi_k, delta_phi_k, jj_S, h_start);
            }
            {
                delta_phi_k = h_phi - h_phi_k;
                h_phi_k = h_phi;
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*(h_phi_k));
                    h[1] = h_start * sin(2*pie*(h_phi_k));
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*(h_phi_k));
                    h[1] = h_start * cos(2*pie*(h_phi_k));
                }

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                if (delta_phi_k <= del_phi_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }


                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                slope_subdivide_phi(h_phi_k, delta_phi_k, jj_S, h_start);
            }
            // printf("\n===\n");
            // printf(  "=2=");
            // printf("\n===\n");
            return 2;
        }

        int slope_subdivide_h_axis(double h_jj_S, double delta_h, int jj_S, double h_start)
        {
            static int sweep_or_loop = 2;

            int j_S;
            
            #ifdef OLD_FUNCTION
            if (h_jj_S == order[jj_S]*h_start)
            {
                printf("\n=========================");
                printf("\n  h_j = 0.0 (%.15e)  ", h_jj_S);
                printf("\n=========================\n");
                
                ensemble_m();
                ensemble_delta_S_squared_max();
                
                backing_up_spin();
                
                save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            if (cutoff_check[1] == 0 && cutoff_check[0] == 0)
            {
                ensemble_m();
                ensemble_delta_S_squared_max();
                // ensemble_E();
                backing_up_spin();
                
                save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_start, sweep_or_loop);
                
                return 0;
            }

            double h_jj_S_k, delta_h_k;
            int cutoff_local;
            long int slope;
            if (cutoff_check[1] > 0)
            {
                find_delta_S_max();
                restoring_spin(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", 1);
                // ensemble_m();
                slope = delta_S_max/CUTOFF_S_SQ + 1;
                delta_h_k = delta_h / (double) slope;
                if (delta_h_k < del_h_cutoff)
                {
                    slope = delta_h/del_h_cutoff + 1;
                    delta_h_k = delta_h / (double) slope;
                }
            }
            // else 
            // {
            //     // ensemble_E();
            //     backing_up_spin();

            //     save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start, sweep_or_loop);

            //     return 1;
            // }

            long int slope_i;
            for ( slope_i = 0; slope_i < slope-1; slope_i++ )
            {
                h_jj_S_k = h_jj_S + order[jj_S] * (delta_h - delta_h_k * (double) (slope_i+1));
                h[jj_S] = h_jj_S;

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif


                if (delta_h_k <= del_h_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }

                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif


                slope_subdivide_h_axis(h_jj_S_k, delta_h_k, jj_S, h_start);
            }
            {
                delta_h_k = h_jj_S - h_jj_S_k;
                h_jj_S_k = h_jj_S;
                h[jj_S] = h_jj_S;

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif

                if (delta_h_k <= del_h_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }
                

                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                slope_subdivide_h_axis(h_jj_S_k, delta_h_k, jj_S, h_start);
            }
            // printf("\n===\n");
            // printf(  "=2=");
            // printf("\n===\n");
            return 2;
        }
        
        int binary_subdivide_phi(double h_phi, double delta_phi, int jj_S, double h_start)
        {
            static int sweep_or_loop = 1;

            int j_S;
            
            #ifdef OLD_FUNCTION
            if (h_phi == 0.0)
            {
                printf("\n=========================");
                printf("\n  phi = 0.0 (%.15e)  ", h_phi);
                printf("\n=========================\n");
                
                ensemble_m();
                ensemble_delta_S_squared_max();
                
                backing_up_spin();
                
                save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);
                return 0;
            }
            #endif
            
            if (cutoff_check[1] == 0 && cutoff_check[0] == 0)
            {
                ensemble_m();
                ensemble_delta_S_squared_max();
                // ensemble_E();
                backing_up_spin();
                
                save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);
                
                return 0;
            }
            
            // if (cutoff_check[1] > 0 && cutoff_check[0] > 0)
            // if (cutoff_check[1] > 0 && cutoff_check[0] == 0)
            // if (cutoff_check[1] == 0 && cutoff_check[0] > 0)

            double h_phi_k, delta_phi_k;
            int cutoff_local;
            
            if (cutoff_check[1] > 0)
            {
                restoring_spin(h_phi, delta_phi, jj_S, delta_S_max, "phi", 1);
                // ensemble_m();
                delta_phi_k = delta_phi / 2.0;
            }
            // else 
            // {
            //     // ensemble_E();
            //     backing_up_spin();

            //     save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);

            //     return 1;
            // }

            
            // for ( h_phi_k = h_phi - order[jj_S] * delta_phi_k; h_phi_k * order[jj_S] <= h_phi * order[jj_S]; h_phi_k = h_phi_k + order[jj_S] * delta_phi_k )
            {
                h_phi_k = h_phi - order[jj_S] * delta_phi_k;
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*(h_phi_k));
                    h[1] = h_start * sin(2*pie*(h_phi_k));
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*(h_phi_k));
                    h[1] = h_start * cos(2*pie*(h_phi_k));
                }

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                if (delta_phi_k <= del_phi_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }


                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                binary_subdivide_phi(h_phi_k, delta_phi_k, jj_S, h_start);
            }
            
            {
                h_phi_k = h_phi;
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*(h_phi_k));
                    h[1] = h_start * sin(2*pie*(h_phi_k));
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*(h_phi_k));
                    h[1] = h_start * cos(2*pie*(h_phi_k));
                }

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif

                if (delta_phi_k <= del_phi_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }


                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                binary_subdivide_phi(h_phi_k, delta_phi_k, jj_S, h_start);
            }
            // printf("\n===\n");
            // printf(  "=2=");
            // printf("\n===\n");
            return 2;
            
        }
        
        int binary_subdivide_h_axis(double h_jj_S, double delta_h, int jj_S, double h_start)
        {
            static int sweep_or_loop = 2;

            int j_S;
            
            #ifdef OLD_FUNCTION
            if (h_jj_S == order[jj_S]*h_start)
            {
                printf("\n=========================");
                printf("\n  h_j = 0.0 (%.15e)  ", h_jj_S);
                printf("\n=========================\n");
                
                ensemble_m();
                ensemble_delta_S_squared_max();
                
                backing_up_spin();
                
                save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            if (cutoff_check[1] == 0 && cutoff_check[0] == 0)
            {
                ensemble_m();
                ensemble_delta_S_squared_max();
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_start, sweep_or_loop);
                
                return 0;
            }

            double h_jj_S_k, delta_h_k;
            int cutoff_local;
            
            if (cutoff_check[1] > 0)
            {
                restoring_spin(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", 1);
                // ensemble_m();
                delta_h_k = delta_h / 2.0;
            }
            // else 
            // {
            //     // ensemble_E();
            //     backing_up_spin();

            //     save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start, sweep_or_loop);

            //     return 1;
            // }

            
            // for ( h_jj_S_k = h_jj_S - order[jj_S] * delta_h_k; h_jj_S_k * order[jj_S] <= h_jj_S * order[jj_S]; h_jj_S_k = h_jj_S_k + order[jj_S] * delta_h_k )
            {
                h_jj_S_k = h_jj_S + order[jj_S] * delta_h_k;
                h[jj_S] = h_jj_S;

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                

                if (delta_h_k <= del_h_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }

                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                binary_subdivide_h_axis(h_jj_S_k, delta_h_k, jj_S, h_start);
            }
            
            {
                h_jj_S_k = h_jj_S;
                h[jj_S] = h_jj_S;
                
                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif


                if (delta_h_k <= del_h_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }

                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                binary_subdivide_h_axis(h_jj_S_k, delta_h_k, jj_S, h_start);
            }
            // printf("\n===\n");
            // printf(  "=2=");
            // printf("\n===\n");
            return 2;
            
        }

        int dynamic_binary_subdivide_phi(double *h_phi, double *delta_phi, int jj_S, double h_start)
        {
            static long int counter = 1;
            static int last_phi_restored = 0;
            static int sweep_or_loop = 1;

            // double h_phi_k, delta_phi_k;
            // h_phi_k = *h_phi;
            // delta_phi_k = *delta_phi;
            int j_S;

            #ifdef OLD_FUNCTION
            if (h_phi[0] == 0.0)
            {
                printf("\n=========================");
                printf("\n  phi = 0.0 (%.15e)  ", h_phi[0]);
                printf("\n=========================\n");
                
                ensemble_m();
                ensemble_delta_S_squared_max();
                
                backing_up_spin();
                
                save_to_file(h_phi[0], delta_phi[0], jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            if (cutoff_check[1] == 0 && cutoff_check[0] == 0)
            {
                ensemble_m();
                ensemble_delta_S_squared_max();
                // ensemble_E();
                backing_up_spin();
                
                save_to_file(h_phi[0], delta_phi[0], jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);


                if (delta_phi[0] >= del_phi)
                {
                    delta_phi[0] = del_phi;
                    return 0;
                }
                else
                {
                    if (delta_S_max < CUTOFF_S_SQ/4 && last_phi_restored == 0)
                    {
                        delta_phi[0] = 2*delta_phi[0];
                        return 0;
                    }
                    else
                    {
                        last_phi_restored = 0;
                        return 0;
                    }
                }
            }
            
            if (cutoff_check[1] > 0)
            {
                restoring_spin(h_phi[0], delta_phi[0], jj_S, delta_S_max, "phi", 1);
                // ensemble_m();
                h_phi[0] = h_phi[0] - delta_phi[0] * order[jj_S];
                delta_phi[0] = delta_phi[0] / 2.0;
                last_phi_restored = 1;
                return 0;
            }

            return 6;
        }

        int dynamic_binary_subdivide_h_axis(double *h_jj_S, double *delta_h, int jj_S, double h_start)
        {
            static long int counter = 1;
            static int last_h_restored = 0;
            static int sweep_or_loop = 2;

            // double h_phi_k, delta_phi_k;
            // h_phi_k = *h_phi;
            // delta_phi_k = *delta_phi;
            int j_S;
            
            #ifdef OLD_FUNCTION
            if (h_jj_S[0] == order[jj_S]*h_start)
            {
                printf("\n=========================");
                printf("\n  h_j = 0.0 (%.15e)  ", h_jj_S[0]);
                printf("\n=========================\n");
                
                ensemble_m();
                ensemble_delta_S_squared_max();
                
                backing_up_spin();
                
                save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_S_max, "h_j", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            if (cutoff_check[1] == 0 && cutoff_check[0] == 0)
            {
                ensemble_m();
                ensemble_delta_S_squared_max();
                // ensemble_E();
                backing_up_spin();
                
                save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_S_max, "h_j", &h_start, sweep_or_loop);


                if (delta_h[0] >= del_h)
                {
                    delta_h[0] = del_h;
                    return 0;
                }
                else
                {
                    if (delta_S_max < CUTOFF_S_SQ/4 && last_h_restored == 0)
                    {
                        delta_h[0] = 2*delta_h[0];
                        return 0;
                    }
                    else
                    {
                        last_h_restored = 0;
                        return 0;
                    }
                }
            }
            
            if (cutoff_check[1] > 0)
            {
                restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_S_max, "h_j", 1);
                // ensemble_m();
                h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                delta_h[0] = delta_h[0] / 2.0;
                last_h_restored = 1;
                return 0;
            }

            return 0;
        }

        int dynamic_binary_slope_divide_phi(double *h_phi, double *delta_phi, int jj_S, double h_start)
        {
            static int last_phi_restored = 0;
            static const double reqd_ratio = 1.1;
            static const double reqd_ratio_sq = 1.1*1.1;
            static long int counter = 1;
            static int sweep_or_loop = 1;
            
            // double h_phi_k, delta_phi_k;
            // h_phi_k = *h_phi;
            // delta_phi_k = *delta_phi;
            int j_S;
            
            #ifdef OLD_FUNCTION
            if (h_phi[0] == 0.0)
            {
                printf("\n=========================");
                printf("\n  phi = 0.0 (%.15e)  ", h_phi[0]);
                printf("\n=========================\n");
                
                ensemble_m();
                ensemble_delta_S_squared_max();
                
                backing_up_spin();
                
                save_to_file(h_phi[0], delta_phi[0], jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            if (cutoff_check[1] == 0 && cutoff_check[0] == 0)
            {
                ensemble_m();
                ensemble_delta_S_squared_max();
                // ensemble_E();
                backing_up_spin();
                
                save_to_file(h_phi[0], delta_phi[0], jj_S, delta_S_max, "phi", &h_start, sweep_or_loop);


                if (delta_phi[0] >= del_phi)
                {
                    if (delta_S_max > CUTOFF_S_SQ/reqd_ratio_sq && last_phi_restored == 0)
                    {
                        delta_phi[0] = delta_phi[0] * (sqrt(CUTOFF_S_SQ/delta_S_max)) / reqd_ratio;
                        // if (delta_h[0] >= del_h*h_start)
                        if (delta_phi[0] >= del_phi)
                        {
                            // delta_h[0] = del_h*h_start;
                            delta_phi[0] = del_phi;
                        }

                        return 0;
                    }
                    else
                    {
                        delta_phi[0] = del_phi;
                        last_phi_restored = 0;
                        return 0;
                    }

                    return 0;
                }
                else
                {
                    if (delta_phi[0] <= del_phi_cutoff)
                    {
                        if (delta_S_max < CUTOFF_S_SQ/reqd_ratio_sq && last_phi_restored == 0)
                        {
                            delta_phi[0] = delta_phi[0] * (sqrt(CUTOFF_S_SQ/delta_S_max)) / reqd_ratio;
                            if (delta_phi[0] >= del_phi)
                            {
                                delta_phi[0] = del_phi;
                            }

                            return 0;
                        }
                        else
                        {
                            last_phi_restored = 0;
                            return 0;
                        }
                    }
                    else // if( (delta_phi[0] > del_phi_cutoff) && (delta_phi[0] < del_phi) )
                    {
                        if (delta_S_max < CUTOFF_S_SQ/4 && last_phi_restored == 0)
                        {
                            delta_phi[0] = delta_phi[0] * 2;
                            if (delta_phi[0] >= del_phi)
                            {
                                delta_phi[0] = del_phi;
                            }
                            return 0;
                        }
                        else // if (delta_S_max*delta_S_max >= CUTOFF_S_SQ/4)
                        {
                            if ( last_phi_restored == 0 )
                            {
                                delta_phi[0] = delta_phi[0] * (sqrt(CUTOFF_S_SQ/delta_S_max)) / reqd_ratio;
                                if (delta_phi[0] >= del_phi)
                                {
                                    delta_phi[0] = del_phi;
                                }
                                return 0;
                            }
                            else
                            {
                                last_phi_restored = 0;
                                return 0;
                            }
                        }
                    }
                }
            }
            
            if (cutoff_check[1] > 0)
            {
                restoring_spin(h_phi[0], delta_phi[0], jj_S, delta_S_max, "phi", 1);
                // ensemble_m();
                h_phi[0] = h_phi[0] - delta_phi[0] * order[jj_S];
                delta_phi[0] = delta_phi[0] / 2.0;
                last_phi_restored = 1;
                return 0;
            }

            return 6;
        }

        int dynamic_binary_slope_divide_h_axis(double *h_jj_S, double *delta_h, int jj_S, double h_start)
        {
            static int last_h_restored = 0;
            static const double reqd_ratio = 1.1;
            static const double reqd_ratio_sq = 1.1*1.1;
            static long int counter = 1;
            static int sweep_or_loop = 2;
            
            // double h_phi_k, delta_phi_k;
            // h_phi_k = *h_phi;
            // delta_phi_k = *delta_phi;
            int j_S;
            
            #ifdef OLD_FUNCTION
            if (h_jj_S[0] == order[jj_S]*h_start)
            {
                printf("\n=========================");
                printf("\n  h_j = 0.0 (%.15e)  ", h_jj_S[0]);
                printf("\n=========================\n");
                
                ensemble_m();
                ensemble_delta_S_squared_max();
                
                backing_up_spin();
                
                save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_S_max, "h_j", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            if (cutoff_check[1] == 0 && cutoff_check[0] == 0)
            {
                ensemble_m();
                ensemble_delta_S_squared_max();
                // ensemble_E();
                backing_up_spin();
                
                save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_S_max, "h_j", &h_start, sweep_or_loop);


                if (delta_h[0] >= del_h)
                {
                    if (delta_S_max > CUTOFF_S_SQ/reqd_ratio_sq && last_h_restored == 0)
                    {
                        delta_h[0] = delta_h[0] * (sqrt(CUTOFF_S_SQ/delta_S_max)) / reqd_ratio;
                        // if (delta_h[0] >= del_h*h_start)
                        if (delta_h[0] >= del_h)
                        {
                            // delta_h[0] = del_h*h_start;
                            delta_h[0] = del_h;
                        }
                        
                        return 0;
                    }
                    else
                    {
                        delta_h[0] = del_h;
                        last_h_restored = 0;
                        return 0;
                    }
                    
                    return 0;
                }
                else
                {
                    if (delta_h[0] <= del_h_cutoff)
                    {
                        if (delta_S_max < CUTOFF_S_SQ/reqd_ratio_sq && last_h_restored == 0)
                        {
                            delta_h[0] = delta_h[0] * (sqrt(CUTOFF_S_SQ/delta_S_max)) / reqd_ratio;
                            // if (delta_h[0] >= del_h*h_start)
                            if (delta_h[0] >= del_h)
                            {
                                // delta_h[0] = del_h*h_start;
                                delta_h[0] = del_h;
                            }
                            
                            return 0;
                        }
                        else
                        {
                            last_h_restored = 0;
                            return 0;
                        }
                    }
                    else // if( (delta_h[0] > del_h_cutoff) && (delta_h[0] < del_h) )
                    {
                        if (delta_S_max < CUTOFF_S_SQ/4 && last_h_restored == 0)
                        {
                            delta_h[0] = delta_h[0] * 2;
                            // if (delta_h[0] >= del_h*h_start)
                            if (delta_h[0] >= del_h)
                            {
                                // delta_h[0] = del_h*h_start;
                                delta_h[0] = del_h;
                            }
                            return 0;
                        }
                        else // if (delta_S_max*delta_S_max >= CUTOFF_S_SQ/4)
                        {
                            if ( last_h_restored == 0 )
                            {
                                delta_h[0] = delta_h[0] * (sqrt(CUTOFF_S_SQ/delta_S_max)) / reqd_ratio;
                                // if (delta_h[0] >= del_h*h_start)
                                if (delta_h[0] >= del_h)
                                {
                                    // delta_h[0] = del_h*h_start;
                                    delta_h[0] = del_h;
                                }
                                return 0;
                            }
                            else
                            {
                                last_h_restored = 0;
                                return 0;
                            }
                        }
                    }
                }
            }
            
            if (cutoff_check[1] > 0)
            {
                restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_S_max, "h_j", 1);
                // ensemble_m();
                h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                delta_h[0] = delta_h[0] / 2.0;
                last_h_restored = 1;
                return 0;
            }

            return 6;
        }
    #else
        double calculate_magnetization_change()
        {
            double delta_m = 0.0;
            int j_S;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                // old_m[j_S] = m[j_S];
                delta_M[j_S] = m[j_S];
            }
            
            ensemble_m();
            
            for (j_S=0; j_S<dim_S; j_S++)
            {
                // new_m[j_S] = m[j_S];
                delta_M[j_S] = m[j_S] - delta_M[j_S];
                delta_m += ( delta_M[j_S] ) * ( delta_M[j_S] );
            }
            delta_m = sqrt( delta_m ) ;

            return delta_m;
        }

        int calculate_Energy_change()
        {
            // double delta_E = 0.0;
            
            delta_E = E;
            
            ensemble_E();
            
            delta_E = E - delta_E;
            
            return 0;
        }

        int const_delta_phi(double h_phi, double delta_phi, int jj_S, double h_start)
        {
            static long int counter = 1;
            static int sweep_or_loop = 1;

            int j_S;
            
            // double old_m[dim_S], new_m[dim_S];
            double delta_m = calculate_magnetization_change();
            #ifdef EXPLORE_ENERGY_LANDSCAPE
            calculate_Energy_change();
            #endif
            #ifdef OLD_FUNCTION
            if (h_phi == 0.0)
            {
                printf("\n=========================");
                printf("\n  phi = 0.0 (%.15e)  ", h_phi);
                printf("\n=========================\n");

                // backing_up_spin();

                save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            // printf(  "\n============================\n");
            if(counter % 100 == 0)
            {
                save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                counter = 0;
            }
            else
            {
                save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", &h_start, sweep_or_loop);
            }
            // printf(  "\n============================\n");
            counter++;

            return 0;
        }
        
        int const_delta_h_axis(double h_jj_S, double delta_h, int jj_S, double h_start)
        {
            static long int counter = 1;
            static int sweep_or_loop = 2;

            int j_S;
            
            // double old_m[dim_S], new_m[dim_S];
            double delta_m = calculate_magnetization_change();
            #ifdef EXPLORE_ENERGY_LANDSCAPE
            calculate_Energy_change();
            #endif
            #ifdef OLD_FUNCTION
            if (h_jj_S == order[jj_S]*h_start)
            {
                printf("\n=========================");
                printf("\n  h_j = 0.0 (%.15e)  ", h_jj_S);
                printf("\n=========================\n");

                // backing_up_spin();

                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                return 0;
            }
            #endif
            
            // printf(  "\n============================\n");
            if(counter % 10 == 0)
            {
                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                counter = 0;
            }
            else
            {
                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
            }
            // printf(  "\n============================\n");
            counter++;

            return 0;
        }

        int slope_subdivide_phi(double h_phi, double delta_phi, int jj_S, double h_start)
        {
            static int sweep_or_loop = 1;

            int j_S;
            
            // double old_m[dim_S], new_m[dim_S];
            double delta_m = calculate_magnetization_change();
            
            #ifdef OLD_FUNCTION
            if (h_phi == 0.0)
            {
                printf("\n=========================");
                printf("\n  phi = 0.0 (%.15e)  ", h_phi);
                printf("\n=========================\n");
                
                backing_up_spin();

                save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            if (delta_phi <= del_phi_cutoff)
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", &h_start, sweep_or_loop);

                return 0;
            }

            double h_phi_k, delta_phi_k;
            int cutoff_local;
            long int slope;
            if (delta_m > CUTOFF_M)
            {
                restoring_spin(h_phi, delta_phi, jj_S, delta_m, "phi", 1);
                // ensemble_m();
                slope = delta_m/CUTOFF_M + 1;
                delta_phi_k = delta_phi / (double) slope;
                if (delta_phi_k < del_phi_cutoff)
                {
                    slope = delta_phi/del_phi_cutoff + 1;
                    delta_phi_k = delta_phi / (double) slope;
                }
            }
            else 
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", &h_start, sweep_or_loop);

                return 1;
            }

            long int slope_i;
            for ( slope_i = 0; slope_i < slope-1; slope_i++ )
            {
                h_phi_k = h_phi - delta_phi + delta_phi_k * (double) (slope_i+1);
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*(h_phi_k));
                    h[1] = h_start * sin(2*pie*(h_phi_k));
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*(h_phi_k));
                    h[1] = h_start * cos(2*pie*(h_phi_k));
                }

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                continue_avalanche();
                // cutoff_local = 0;
                // do
                // {
                //     // double cutoff_local_last = cutoff_local;
                //     cutoff_local = find_change();
                // }
                // while (cutoff_local > 0); // 10^-14


                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                slope_subdivide_phi(h_phi_k, delta_phi_k, jj_S, h_start);
            }
            {
                delta_phi_k = h_phi - h_phi_k;
                h_phi_k = h_phi;
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*(h_phi_k));
                    h[1] = h_start * sin(2*pie*(h_phi_k));
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*(h_phi_k));
                    h[1] = h_start * cos(2*pie*(h_phi_k));
                }

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                continue_avalanche();
                // cutoff_local = 0;
                // do
                // {
                //     // double cutoff_local_last = cutoff_local;
                //     cutoff_local = find_change();
                // }
                // while (cutoff_local > 0); // 10^-14


                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                slope_subdivide_phi(h_phi_k, delta_phi_k, jj_S, h_start);
            }
            // printf("\n===\n");
            // printf(  "=2=");
            // printf("\n===\n");
            return 2;
        }

        int slope_subdivide_h_axis(double h_jj_S, double delta_h, int jj_S, double h_start)
        {
            static int sweep_or_loop = 2;

            int j_S;
            
            // double old_m[dim_S], new_m[dim_S];
            double delta_m = calculate_magnetization_change();
            
            #ifdef OLD_FUNCTION
            if (h_jj_S == order[jj_S]*h_start)
            {
                printf("\n=========================");
                printf("\n  h_j = 0.0 (%.15e)  ", h_jj_S);
                printf("\n=========================\n");

                // backing_up_spin();

                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                return 0;
            }
            #endif
            
            // if (delta_h <= del_h_cutoff*h_start)
            if (delta_h <= del_h_cutoff)
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start, sweep_or_loop);

                return 0;
            }

            double h_jj_S_k, delta_h_k;
            int cutoff_local;
            long int slope;
            if (delta_m > CUTOFF_M)
            {
                restoring_spin(h_jj_S, delta_h, jj_S, delta_m, "h_j", 1);
                // ensemble_m();
                slope = delta_m/CUTOFF_M + 1;
                delta_h_k = delta_h / (double) slope;
                // if (delta_h_k < del_h_cutoff*h_start)
                if (delta_h_k < del_h_cutoff)
                {
                    slope = delta_h/del_h_cutoff + 1;
                    delta_h_k = delta_h / (double) slope;
                }
            }
            else 
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start, sweep_or_loop);

                return 1;
            }

            long int slope_i;
            for ( slope_i = 0; slope_i < slope-1; slope_i++ )
            {
                h_jj_S_k = h_jj_S + order[jj_S] * (delta_h - delta_h_k * (double) (slope_i+1));
                h[jj_S] = h_jj_S;

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                continue_avalanche();
                
                // cutoff_local = 0;
                // do
                // {
                //     // double cutoff_local_last = cutoff_local;
                //     cutoff_local = find_change();
                // }
                // while (cutoff_local > 0); // 10^-14


                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif


                slope_subdivide_h_axis(h_jj_S_k, delta_h_k, jj_S, h_start);
            }
            {
                delta_h_k = h_jj_S - h_jj_S_k;
                h_jj_S_k = h_jj_S;
                h[jj_S] = h_jj_S;

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                continue_avalanche();
                
                // cutoff_local = 0;
                // do
                // {
                //     // double cutoff_local_last = cutoff_local;
                //     cutoff_local = find_change();
                // }
                // while (cutoff_local > 0); // 10^-14


                slope_subdivide_h_axis(h_jj_S_k, delta_h_k, jj_S, h_start);
            }
            // printf("\n===\n");
            // printf(  "=2=");
            // printf("\n===\n");
            return 2;
        }
        
        int binary_subdivide_phi(double h_phi, double delta_phi, int jj_S, double h_start)
        {
            static int sweep_or_loop = 1;

            int j_S;
        
            // double old_m[dim_S], new_m[dim_S];
            double delta_m = calculate_magnetization_change();
            

            #ifdef OLD_FUNCTION
            if (h_phi == 0.0)
            {
                printf("\n=========================");
                printf("\n  phi = 0.0 (%.15e)  ", h_phi);
                printf("\n=========================\n");
                
                backing_up_spin();

                save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            if (delta_phi <= del_phi_cutoff)
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                
                return 0;
            }

            double h_phi_k, delta_phi_k;
            int cutoff_local;
            
            if (delta_m > CUTOFF_M)
            {
                restoring_spin(h_phi, delta_phi, jj_S, delta_m, "phi", 1);
                // ensemble_m();
                delta_phi_k = delta_phi / 2.0;
            }
            else 
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", &h_start, sweep_or_loop);

                return 1;
            }

            
            // for ( h_phi_k = h_phi - order[jj_S] * delta_phi_k; h_phi_k * order[jj_S] <= h_phi * order[jj_S]; h_phi_k = h_phi_k + order[jj_S] * delta_phi_k )
            {
                h_phi_k = h_phi - order[jj_S] * delta_phi_k;
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*(h_phi_k));
                    h[1] = h_start * sin(2*pie*(h_phi_k));
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*(h_phi_k));
                    h[1] = h_start * cos(2*pie*(h_phi_k));
                }

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                continue_avalanche();
                
                // cutoff_local = 0;
                // do
                // {
                //     // double cutoff_local_last = cutoff_local;
                //     cutoff_local = find_change();
                // }
                // while (cutoff_local > 0); // 10^-14
    

                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                binary_subdivide_phi(h_phi_k, delta_phi_k, jj_S, h_start);
            }
            
            {
                h_phi_k = h_phi;
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*(h_phi_k));
                    h[1] = h_start * sin(2*pie*(h_phi_k));
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*(h_phi_k));
                    h[1] = h_start * cos(2*pie*(h_phi_k));
                }

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                continue_avalanche();
                
                // cutoff_local = 0;
                // do
                // {
                //     // double cutoff_local_last = cutoff_local;
                //     cutoff_local = find_change();
                // }
                // while (cutoff_local > 0); // 10^-14


                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                binary_subdivide_phi(h_phi_k, delta_phi_k, jj_S, h_start);
            }
            // printf("\n===\n");
            // printf(  "=2=");
            // printf("\n===\n");
            return 2;
            
        }
        
        int binary_subdivide_h_axis(double h_jj_S, double delta_h, int jj_S, double h_start)
        {
            static int sweep_or_loop = 2;

            int j_S;
            
            // double old_m[dim_S], new_m[dim_S];
            double delta_m = calculate_magnetization_change();
            
            #ifdef OLD_FUNCTION
            if (h_jj_S == order[jj_S]*h_start)
            {
                printf("\n=========================");
                printf("\n  h_j = 0.0 (%.15e)  ", h_jj_S);
                printf("\n=========================\n");

                // backing_up_spin();

                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            // if (delta_h <= del_h_cutoff*h_start)
            if (delta_h <= del_h_cutoff)
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                
                return 0;
            }

            double h_jj_S_k, delta_h_k;
            int cutoff_local;
            
            if (delta_m > CUTOFF_M)
            {
                restoring_spin(h_jj_S, delta_h, jj_S, delta_m, "h_j", 1);
                // ensemble_m();
                delta_h_k = delta_h / 2.0;
            }
            else 
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start, sweep_or_loop);

                return 1;
            }

            
            // for ( h_jj_S_k = h_jj_S - order[jj_S] * delta_h_k; h_jj_S_k * order[jj_S] <= h_jj_S * order[jj_S]; h_jj_S_k = h_jj_S_k + order[jj_S] * delta_h_k )
            {
                h_jj_S_k = h_jj_S + order[jj_S] * delta_h_k;
                h[jj_S] = h_jj_S;

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                continue_avalanche();
                
                // cutoff_local = 0;
                // do
                // {
                //     // double cutoff_local_last = cutoff_local;
                //     cutoff_local = find_change();
                // }
                // while (cutoff_local > 0); // 10^-14

                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                binary_subdivide_h_axis(h_jj_S_k, delta_h_k, jj_S, h_start);
            }
            
            {
                h_jj_S_k = h_jj_S;
                h[jj_S] = h_jj_S;
                
                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                continue_avalanche();
                
                // cutoff_local = 0;
                // do
                // {
                //     // double cutoff_local_last = cutoff_local;
                //     cutoff_local = find_change();
                // }
                // while (cutoff_local > 0); // 10^-14


                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                binary_subdivide_h_axis(h_jj_S_k, delta_h_k, jj_S, h_start);
            }
            // printf("\n===\n");
            // printf(  "=2=");
            // printf("\n===\n");
            return 2;
            
        }

        int dynamic_binary_subdivide_phi(double *h_phi, double *delta_phi, int jj_S, double h_start)
        {
            static long int counter = 1;
            static int last_phi_restored = 0;
            static int sweep_or_loop = 1;

            // double h_phi_k, delta_phi_k;
            // h_phi_k = *h_phi;
            // delta_phi_k = *delta_phi;
            int j_S;
            
            // double old_m[dim_S], new_m[dim_S];
            double delta_m = calculate_magnetization_change();
            

            #ifdef OLD_FUNCTION
            if (h_phi[0] == 0.0)
            {
                printf("\n=========================");
                printf("\n  phi = 0.0 (%.15e)  ", h_phi[0]);
                printf("\n=========================\n");
                
                backing_up_spin();

                save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            double ratio_delta_m = CUTOFF_M/del_phi_cutoff;
            if (delta_m > del_phi_cutoff)
            {
                ratio_delta_m = CUTOFF_M/delta_m;
            }

            if (delta_phi[0] <= del_phi_cutoff)
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                
                if (ratio_delta_m > 2 && last_phi_restored == 0)
                {
                    delta_phi[0] = delta_phi[0] * 2;
                    if (delta_phi[0] >= del_phi)
                    {
                        delta_phi[0] = del_phi;
                    }
                }
                else
                {
                    last_phi_restored = 0;
                }
                
                return 0;
            }
            else
            {
                if (delta_phi[0] < del_phi)
                {
                    if (ratio_delta_m > 2 && last_phi_restored == 0)
                    {
                        backing_up_spin();

                        save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", &h_start, sweep_or_loop);

                        delta_phi[0] = delta_phi[0] * 2;
                        if (delta_phi[0] >= del_phi)
                        {
                            delta_phi[0] = del_phi;
                        }
                        return 1;
                    }
                    else
                    {
                        if (ratio_delta_m <= 1)
                        {
                            restoring_spin(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
                            last_phi_restored = 1;
                            h_phi[0] = h_phi[0] - delta_phi[0] * order[jj_S];
                            
                            delta_phi[0] = delta_phi[0] / 2;
                            return 2;
                        }
                        else
                        {
                            backing_up_spin();
                            last_phi_restored = 0;
                            save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                            return 3;
                        }
                    }
                }
                else
                {
                    if (ratio_delta_m <= 1)
                    {
                        restoring_spin(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
                        last_phi_restored = 1;
                        h_phi[0] = h_phi[0] - delta_phi[0] * order[jj_S];
                        
                        delta_phi[0] = delta_phi[0] / 2;
                        return 4;
                    }
                    else
                    {
                        backing_up_spin();
                        last_phi_restored = 0;
                        save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                        return 5;
                    }
                }
            }
            return 6;
        }

        int dynamic_binary_subdivide_h_axis(double *h_jj_S, double *delta_h, int jj_S, double h_start)
        {
            static long int counter = 1;
            static int last_h_restored = 0;
            static int sweep_or_loop = 2;

            // double h_phi_k, delta_phi_k;
            // h_phi_k = *h_phi;
            // delta_phi_k = *delta_phi;
            int j_S;
            
            // double old_m[dim_S], new_m[dim_S];
            double delta_m = calculate_magnetization_change();
            
            #ifdef OLD_FUNCTION
            if (h_jj_S[0] == order[jj_S]*h_start)
            {
                printf("\n=========================");
                printf("\n  h_j = 0.0 (%.15e)  ", h_jj_S[0]);
                printf("\n=========================\n");

                // backing_up_spin();

                save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            double ratio_delta_m = CUTOFF_M/del_h_cutoff;
            if (delta_m > del_h_cutoff)
            {
                ratio_delta_m = CUTOFF_M/delta_m;
            }

            // if (delta_h[0] <= del_h_cutoff*h_start)
            if (delta_h[0] <= del_h_cutoff)
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                
                if (ratio_delta_m > 2 && last_h_restored == 0)
                {
                    delta_h[0] = delta_h[0] * 2;
                    // if (delta_h[0] >= del_h*h_start)
                    if (delta_h[0] >= del_h)
                    {
                        // delta_h[0] = del_h*h_start;
                        delta_h[0] = del_h;
                    }
                }
                else
                {
                    last_h_restored = 0;
                }
                
                return 0;
            }
            else
            {
                // if (delta_h[0] < del_h*h_start)
                if (delta_h[0] < del_h)
                {
                    if (ratio_delta_m > 2 && last_h_restored == 0)
                    {
                        backing_up_spin();

                        save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", &h_start, sweep_or_loop);

                        delta_h[0] = delta_h[0] * 2;
                        // if (delta_h[0] >= del_h*h_start)
                        if (delta_h[0] >= del_h)
                        {
                            // delta_h[0] = del_h*h_start;
                            delta_h[0] = del_h;
                        }
                        return 1;
                    }
                    else
                    {
                        if (ratio_delta_m <= 1)
                        {
                            restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", 1);
                            last_h_restored = 1;
                            h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                            
                            delta_h[0] = delta_h[0] / 2;
                            return 2;
                        }
                        else
                        {
                            backing_up_spin();
                            last_h_restored = 0;
                            save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                            return 3;
                        }
                    }
                }
                else
                {
                    if (ratio_delta_m <= 1)
                    {
                        restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", 1);
                        last_h_restored = 1;
                        h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                        
                        delta_h[0] = delta_h[0] / 2;
                        return 4;
                    }
                    else
                    {
                        backing_up_spin();
                        last_h_restored = 0;
                        save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                        return 5;
                    }
                }
            }
            return 6;
        }

        int dynamic_binary_slope_divide_phi(double *h_phi, double *delta_phi, int jj_S, double h_start)
        {
            static int last_phi_restored = 0;
            static const double reqd_ratio = 1.1;
            static long int counter = 1;
            static int sweep_or_loop = 1;

            // double h_phi_k, delta_phi_k;
            // h_phi_k = *h_phi;
            // delta_phi_k = *delta_phi;
            int j_S;
            
            // double old_m[dim_S], new_m[dim_S];
            double delta_m = calculate_magnetization_change();
            

            #ifdef OLD_FUNCTION
            if (h_phi[0] == 0.0)
            {
                printf("\n=========================");
                printf("\n  phi = 0.0 (%.15e)  ", h_phi[0]);
                printf("\n=========================\n");
                
                backing_up_spin();

                save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            double ratio_delta_m = CUTOFF_M/del_phi_cutoff;
            if (delta_m > del_phi_cutoff)
            {
                ratio_delta_m = CUTOFF_M/delta_m;
            }

            if (delta_phi[0] <= del_phi_cutoff)
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                
                if (ratio_delta_m > 2 && last_phi_restored == 0)
                {
                    delta_phi[0] = delta_phi[0] * 2;
                    if (delta_phi[0] >= del_phi)
                    {
                        delta_phi[0] = del_phi;
                    }
                }
                else
                {
                    if (ratio_delta_m > reqd_ratio && last_phi_restored == 0)
                    {
                        delta_phi[0] = delta_phi[0] * ratio_delta_m / reqd_ratio;
                        if (delta_phi[0] >= del_phi)
                        {
                            delta_phi[0] = del_phi;
                        }
                    }
                }
                if (last_phi_restored == 1)
                {
                    last_phi_restored = 0;
                }

                return 0;
            }
            else
            {
                if (delta_phi[0] < del_phi)
                {
                    if (ratio_delta_m > 2 && last_phi_restored == 0)
                    {
                        backing_up_spin();

                        save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", &h_start, sweep_or_loop);

                        delta_phi[0] = delta_phi[0] * 2;
                        if (delta_phi[0] >= del_phi)
                        {
                            delta_phi[0] = del_phi;
                        }
                        return 1;
                    }
                    else
                    {
                        if (ratio_delta_m <= 1)
                        {
                            restoring_spin(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
                            last_phi_restored = 1;
                            h_phi[0] = h_phi[0] - delta_phi[0] * order[jj_S];
                            
                            delta_phi[0] = delta_phi[0] / 2;
                            return 2;
                        }
                        else
                        {
                            backing_up_spin();

                            save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                            if (last_phi_restored == 0)
                            {
                                delta_phi[0] = delta_phi[0] * ratio_delta_m / reqd_ratio;
                                if (delta_phi[0] >= del_phi)
                                {
                                    delta_phi[0] = del_phi;
                                }
                            }
                            else
                            {
                                last_phi_restored = 0;
                            }
                            
                            return 3;
                        }
                    }
                }
                else
                {
                    if (ratio_delta_m <= 1)
                    {
                        restoring_spin(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
                        last_phi_restored = 1;
                        h_phi[0] = h_phi[0] - delta_phi[0] * order[jj_S];
                        
                        delta_phi[0] = delta_phi[0] / 2;
                        return 4;
                    }
                    else
                    {
                        backing_up_spin();

                        save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", &h_start, sweep_or_loop);
                        last_phi_restored = 0;
                        if (ratio_delta_m < reqd_ratio)
                        {
                            delta_phi[0] = delta_phi[0] * ratio_delta_m / reqd_ratio;
                        }
                        
                        return 5;
                    }
                }
            }
            return 6;
        }

        int dynamic_binary_slope_divide_h_axis(double *h_jj_S, double *delta_h, int jj_S, double h_start)
        {
            static int last_h_restored = 0;
            static const double reqd_ratio = 1.1;
            static long int counter = 1;
            static int sweep_or_loop = 2;

            // double h_phi_k, delta_phi_k;
            // h_phi_k = *h_phi;
            // delta_phi_k = *delta_phi;
            int j_S;
            
            // double old_m[dim_S], new_m[dim_S];
            double delta_m = calculate_magnetization_change();
            
            #ifdef OLD_FUNCTION
            if (h_jj_S[0] == order[jj_S]*h_start)
            {
                printf("\n=========================");
                printf("\n  h_j = 0.0 (%.15e)  ", h_jj_S[0]);
                printf("\n=========================\n");

                // backing_up_spin();

                save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                return 0;
            }
            #endif

            double ratio_delta_m = CUTOFF_M/del_h_cutoff;
            if (delta_m > del_h_cutoff)
            {
                ratio_delta_m = CUTOFF_M/delta_m;
            }

            // if (delta_h[0] <= del_h_cutoff*h_start)
            if (delta_h[0] <= del_h_cutoff)
            {
                // ensemble_E();
                backing_up_spin();

                save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                
                if (ratio_delta_m > 2 && last_h_restored == 0)
                {
                    delta_h[0] = delta_h[0] * 2;
                    // if (delta_h[0] >= del_h*h_start)
                    if (delta_h[0] >= del_h)
                    {
                        // delta_h[0] = del_h*h_start;
                        delta_h[0] = del_h;
                    }
                }
                else
                {
                    if (ratio_delta_m > reqd_ratio && last_h_restored == 0)
                    {
                        delta_h[0] = delta_h[0] * ratio_delta_m / reqd_ratio;
                        // if (delta_h[0] >= del_h*h_start)
                        if (delta_h[0] >= del_h)
                        {
                            // delta_h[0] = del_h*h_start;
                            delta_h[0] = del_h;
                        }
                    }
                }
                if (last_h_restored == 1)
                {
                    last_h_restored = 0;
                }

                return 0;
            }
            else
            {
                // if (delta_h[0] < del_h*h_start)
                if (delta_h[0] < del_h)
                {
                    if (ratio_delta_m > 2 && last_h_restored == 0)
                    {
                        backing_up_spin();

                        save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", &h_start, sweep_or_loop);

                        delta_h[0] = delta_h[0] * 2;
                        // if (delta_h[0] >= del_h*h_start)
                        if (delta_h[0] >= del_h)
                        {
                            // delta_h[0] = del_h*h_start;
                            delta_h[0] = del_h;
                        }
                        return 1;
                    }
                    else
                    {
                        if (ratio_delta_m <= 1)
                        {
                            restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", 1);
                            last_h_restored = 1;
                            h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                            
                            delta_h[0] = delta_h[0] / 2;
                            return 2;
                        }
                        else
                        {
                            backing_up_spin();

                            save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                            if (last_h_restored == 0)
                            {
                                delta_h[0] = delta_h[0] * ratio_delta_m / reqd_ratio;
                                // if (delta_h[0] >= del_h*h_start)
                                if (delta_h[0] >= del_h)
                                {
                                    // delta_h[0] = del_h*h_start;
                                    delta_h[0] = del_h;
                                }
                            }
                            else
                            {
                                last_h_restored = 0;
                            }
                            
                            return 3;
                        }
                    }
                }
                else
                {
                    if (ratio_delta_m <= 1)
                    {
                        restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", 1);
                        last_h_restored = 1;
                        h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                        
                        delta_h[0] = delta_h[0] / 2;
                        return 4;
                    }
                    else
                    {
                        backing_up_spin();

                        save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_j", &h_start, sweep_or_loop);
                        last_h_restored = 0;
                        if (ratio_delta_m < reqd_ratio)
                        {
                            delta_h[0] = delta_h[0] * ratio_delta_m / reqd_ratio;
                        }
                        
                        return 5;
                    }
                }
            }
            return 6;
        }
    #endif
    
    int check_spin_direction(int jj_S)
    {
        long int i;
        int Spin_Order = 1;
        #pragma omp parallel for reduction(&&:Spin_Order)
        for (i=0; i<no_of_sites; i++)
        {
            Spin_Order = Spin_Order && (spin[i*dim_S+jj_S] * order[jj_S] > Spin_Saturation_Limit);
        }
        return Spin_Order;
    }

//====================  RFXY ZTNE                     ====================//

    int zero_temp_RFXY_hysteresis_axis_checkerboard_old(int jj_S, double order_start, double* h_sweep_abs)
    {
        restore_checkpoint(/* startif */-1, TYPE_DOUBLE, dim_S, h, 0); // reset static variable
        start_time = get_time_if_parallel();
        // CUTOFF_S_SQ = 4.0*del_h*del_h;
        // del_h_cutoff = del_h/no_of_sites;
        CUTOFF_M = 1.0/(double)no_of_sites;
        CUTOFF_S_SQ = 1.0;
        #ifdef _OPENMP
        #ifdef BUNDLE
            omp_set_num_threads(BUNDLE);
        #else
            omp_set_num_threads(num_of_threads);
        #endif
        #endif
        /* #ifdef _OPENMP
            if (num_of_threads<=16)
            {
                omp_set_num_threads(num_of_threads);
            }
            else 
            {
                if (num_of_threads<=20)
                {
                    omp_set_num_threads(16);
                }
                else
                {
                    omp_set_num_threads(num_of_threads-4);
                }
            }
        #endif */
        
        #ifdef enable_CUDA_CODE
            #ifdef CUDA_with_managed
            cudaMemcpy(dev_CUTOFF_SPIN, &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_CUTOFF_S_SQ, &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_spin, spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_J, J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpy(dev_J_random, J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpy(dev_h_random, h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_N_N_I, N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_no_of_black_white_sites, no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_black_white_checkerboard, black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #else
            cudaMemcpyToSymbol("dev_CUTOFF_SPIN", &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_CUTOFF_S_SQ", &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_spin", spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_J", J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpyToSymbol("dev_J_random", J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpyToSymbol("dev_h_random", h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_N_N_I", N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_no_of_black_white_sites", no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_black_white_checkerboard", black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #endif
        #endif

        double T_temp = T;
        T = 0;
        ax_ro = 0;
        or_ho_ra = 0;
        int cutoff_local = 0;
        int j_S, j_L;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[jj_S] = order_start;
            }
            else
            {
                order[j_S] = 0;
            }
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = h_sweep_abs[j_S];
        }
        // double h_start = order[jj_S]*(h_max+h_i_max);
        double h_start, h_end;
        double delta_h;
        double sigma_h_trnsvrs = 0.0;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S != jj_S)
            {
                sigma_h_trnsvrs += sigma_h[j_S] * sigma_h[j_S];
            }
        }
        sigma_h_trnsvrs = sqrt(sigma_h_trnsvrs);
        if (sigma_h_trnsvrs > 0.0)
        {
            h_start = -order[jj_S]*(sigma_h_trnsvrs);
            // delta_h = sigma_h_trnsvrs*del_h;
            h_order = 0;
            r_order = 0;
            // initialize_spin_config();
        }
        else
        {
            if (sigma_h[jj_S] == 0.0)
            {
                h_start = -order[jj_S]*(h_max);
                // delta_h = h_max*del_h;
            }
            else
            {
                // h[!jj_S] = sigma_h[jj_S]*del_h_cutoff;
                // h[!jj_S] = 0.10*del_h_cutoff;
                // h_start = order[jj_S]*(h_max);
                if (h_i_max >= -h_i_min)
                {
                    h_start = -order[jj_S]*(h_max + h_i_max);
                }
                else
                {
                    h_start = -order[jj_S]*(h_max - h_i_min);
                }
                // delta_h = (h_i_max + h_max)*del_h;
            }
            
            h_order = 0;
            r_order = 1;
            // initialize_spin_config();
        }
        initialize_spin_config();
        delta_h = del_h;
        h_end = -h_start;
        

        printf("\nztne RFXY looping along h[%d] at T=%lf.. \n", jj_S,  T);

        ensemble_m();
        ensemble_E();
        
        // set output_variable_name=1 here
        {
            output_prepend = 1;
            output_h = 1;
            output_m = 1;
            #ifdef CHECK_AVALANCHE
            output_delta_S_abs = 1;
            output_delta_S_squared = 1;
            #else
            output_delta_M = 1;
            #endif

            #ifdef EXPLORE_ENERGY_LANDSCAPE
            output_E = 1;
            output_delta_E = 1;
            #endif
            output_append = 1;
        }

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hys_axis_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%lf", T);
            /* pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); */
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%lf_%lf)", h_start, delta_h);
                }
                else
                {
                    pos += sprintf(pos, "%lf", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%.3f", order[j_S]);
            // }
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_h%d_r%d", h_order, r_order);
            #ifdef enable_CUDA_CODE
            pos += sprintf(pos, "_cuda");
            #endif
            pos += sprintf(pos, ".dat");

        }
        
        // column labels and parameters
        output_param_file(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        #ifdef UPDATE_ALL_NON_EQ
        {
            printf("\nUpdating all sites simultaneously.. \n");
            
            fprintf(pFile_1, "==================================================================================\n");
            fprintf(pFile_1, "Updating all sites simultaneously.. \n");
            fprintf(pFile_1, "==================================================================================\n");
        }
        #endif
        // else
        #ifdef UPDATE_CHKR_NON_EQ
        {
            printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

            fprintf(pFile_1, "==================================================================================\n");
            fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
            fprintf(pFile_1, "==================================================================================\n");
        }
        #endif
        fclose(pFile_1);
        #ifdef CHECK_AVALANCHE
        output_label(output_file_0, "h[%d]\t", "max(|delta{S}|)\t");
        #else
        output_label(output_file_0, "h[%d]\t", "|delta{m[%d]}|\t");
        #endif

        long int site_i;
        int black_or_white = BLACK_WHITE;
        double h_jj_S;


        // print statements:
        {
            // fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            // fprintf(pFile_1, "\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                    // fprintf(pFile_1, ",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                    // fprintf(pFile_1, "%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                    // fprintf(pFile_1, "%.3f", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
            // fprintf(pFile_1, "}, %d, %d)\n", h_order, r_order);
            // fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
        }
        h_jj_S = h_start;

        // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
        //     delta_h = del_h_cutoff*fabs(h_start);
        // #endif
        pFile_1 = fopen(output_file_0, "a");
        // print rate
        {
            #ifdef CONST_RATE
                printf("\n ====== CONSTANT RATE ====== \n");
            #endif

            #ifdef DIVIDE_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE ====== \n");
            #endif
            
            #ifdef BINARY_DIVISION
                printf("\n ====== BINARY DIVISION to LOCATE AVALANCHE ====== \n");
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                printf("\n ====== SPEEDUP/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
        }
        
        printf("\n=========================");
        printf("\n  h[%d] = h_start (%.15e)  ", jj_S, h_jj_S);
        printf("\n=========================\n");
        {
            h[jj_S] = h_jj_S;
            ensemble_m();
            backing_up_spin();
            // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE) 
            //     backing_up_spin();
            // #endif

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #else
                cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #endif
            #endif

            
            #ifdef CHECK_AVALANCHE
                continue_avalanche();
            #else
                cutoff_local = 0;
                do
                {
                    // double cutoff_local_last = cutoff_local;
                    cutoff_local = find_change();
                }
                while (cutoff_local > 0); // 10^-14
            #endif
            
            
            #ifdef CHECK_AVALANCHE
            ensemble_delta_S_squared_max();
            ensemble_m();
            backing_up_spin();
            save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_sweep_abs[jj_S], 2);
            #else
            double delta_m = calculate_magnetization_change();
            backing_up_spin();
            save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_sweep_abs[jj_S], 2);
            #endif


        }

        // for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
        while (order[jj_S] * h_jj_S < order[jj_S] * h_end)
        {
            // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
            // if (h_phi * order[jj_S] + delta_phi >= 1.0 && h_phi * order[jj_S] < 1.0)
            if (h_jj_S * order[jj_S] + delta_h >= h_end * order[jj_S] )
            {
                // delta_phi = 1.0 - h_phi * order[jj_S];
                delta_h = (h_end - h_jj_S) * order[jj_S];
                // h_phi = 1.0;
                h_jj_S = h_end;
            }
            else
            {
                h_jj_S = h_jj_S + order[jj_S] * delta_h;
            }
            // #else
            //     h_jj_S = h_jj_S - order[jj_S] * delta_h;
            // #endif
            h[jj_S] = h_jj_S;
            
            // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE) 
            //     backing_up_spin();
            // #endif

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #else
                cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #endif
            #endif
            
            
            #ifdef CHECK_AVALANCHE
                #ifdef CONST_RATE
                continue_avalanche();
                #else
                if (delta_h <= del_h_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }
                #endif
            #else
                cutoff_local = 0;
                do
                {
                    // double cutoff_local_last = cutoff_local;
                    cutoff_local = find_change();
                }
                while (cutoff_local > 0); // 10^-14
            #endif
            
            #ifdef enable_CUDA_CODE
            // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
            #endif
            
            
            #ifdef CONST_RATE
                const_delta_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif

            #ifdef DIVIDE_BY_SLOPE
                slope_subdivide_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
            #ifdef BINARY_DIVISION
                binary_subdivide_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                dynamic_binary_subdivide_h_axis(&h_jj_S, &delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                dynamic_binary_slope_divide_h_axis(&h_jj_S, &delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
        }
        fclose(pFile_1);
        printf("\n=========================");
        printf("\n  |h[%d]| >= |h_end| (%.15e)  ", jj_S, h_jj_S);
        printf("\n=========================\n");

        // ----------------------------------------------//
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[j_S] = -order_start;
            }
            else
            {
                order[j_S] = 0;
            }
        }
        // h_start = order[jj_S]*(h_max+h_i_max);
        if (sigma_h_trnsvrs > 0.0)
        {
            h_start = -order[jj_S]*(sigma_h_trnsvrs);
            // initialize_spin_config();
        }
        else
        {
            if (sigma_h[jj_S] == 0.0)
            {
                h_start = -order[jj_S]*(h_max);
            }
            else
            {
                // h_start = order[jj_S]*(h_max);
                if (h_i_max >= -h_i_min)
                {
                    h_start = -order[jj_S]*(h_max + h_i_max);
                }
                else
                {
                    h_start = -order[jj_S]*(h_max - h_i_min);
                }
            }
            
        }
        delta_h = del_h;
        h_end = -h_start;
        // h_order = 0;
        // r_order = 1;
        // initialize_spin_config();
        
        ensemble_m();
        ensemble_E();

        pFile_1 = fopen(output_file_0, "a");
        // print statements:
        {
            // fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            // fprintf(pFile_1, "\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                    // fprintf(pFile_1, ",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                    // fprintf(pFile_1, "%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                    // fprintf(pFile_1, "%.3f", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
            // fprintf(pFile_1, "}, %d, %d)\n", h_order, r_order);
            // fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
        }
        h_jj_S = h_start;

        // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
        //     delta_h = del_h_cutoff*fabs(h_start);
        // #endif
        printf("\n=========================");
        printf("\n  h[%d] = h_start (%.15e)  ", jj_S, h_jj_S);
        printf("\n=========================\n");
        {
            h[jj_S] = h_jj_S;
            ensemble_m();
            backing_up_spin();
            // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE) 
            //     backing_up_spin();
            // #endif

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #else
                cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #endif
            #endif

            
            #ifdef CHECK_AVALANCHE
                continue_avalanche();
            #else
                cutoff_local = 0;
                do
                {
                    // double cutoff_local_last = cutoff_local;
                    cutoff_local = find_change();
                }
                while (cutoff_local > 0); // 10^-14
            #endif
            
            
            #ifdef CHECK_AVALANCHE
            ensemble_delta_S_squared_max();
            ensemble_m();
            backing_up_spin();
            save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_sweep_abs[jj_S], 2);
            #else
            double delta_m = calculate_magnetization_change();
            backing_up_spin();
            save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_sweep_abs[jj_S], 2);
            #endif

        }

        // for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
        while (order[jj_S] * h_jj_S < order[jj_S] * h_end)
        {
            // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
            if (h_jj_S * order[jj_S] + delta_h >= h_end * order[jj_S])
            {
                // delta_phi = 1.0 - h_phi * order[jj_S];
                delta_h = (h_end - h_jj_S) * order[jj_S];
                // h_phi = 1.0;
                h_jj_S = h_end;
            }
            else
            {
                h_jj_S = h_jj_S + order[jj_S] * delta_h;
            }
            // #else
            //     h_jj_S = h_jj_S - order[jj_S] * delta_h;
            // #endif
            h[jj_S] = h_jj_S;
            
            // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE) 
            //     backing_up_spin();
            // #endif

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #else
                cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #endif
            #endif

            
            #ifdef CHECK_AVALANCHE
                #ifdef CONST_RATE
                continue_avalanche();
                #else
                if (delta_h <= del_h_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }
                #endif
            #else
                cutoff_local = 0;
                do
                {
                    // double cutoff_local_last = cutoff_local;
                    cutoff_local = find_change();
                }
                while (cutoff_local > 0); // 10^-14
            #endif

            #ifdef enable_CUDA_CODE
            // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
            #endif


            #ifdef CONST_RATE
                const_delta_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif

            #ifdef DIVIDE_BY_SLOPE
                slope_subdivide_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
            #ifdef BINARY_DIVISION
                binary_subdivide_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                dynamic_binary_subdivide_h_axis(&h_jj_S, &delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                dynamic_binary_slope_divide_h_axis(&h_jj_S, &delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
        }
        fclose(pFile_1);
        printf("\n=========================");
        printf("\n  |h[%d]| >= |h_end| (%.15e)  ", jj_S, h_jj_S);
        printf("\n=========================\n");

        T=T_temp;
        // if (update_all_or_checker == 0)
        // {
        //     free(spin_temp);
        // }
        return 0;
    }

    int zero_temp_RFXY_hysteresis_axis_checkerboard(int jj_S, double order_start, double* h_start, double h_abs_jj_S, int finite_sweep, char output_file_name[])
    {
        restore_checkpoint(/* startif */-1, TYPE_DOUBLE, dim_S, h, 0); // reset static variable
        start_time = get_time_if_parallel();
        // CUTOFF_S_SQ = 4.0*pie*pie*del_phi*del_phi;

        // del_phi_cutoff = del_phi/no_of_sites;
        Spin_Saturation_Limit = sqrt(3.0)/2.0;
        CUTOFF_M = 1.0/no_of_sites;
        CUTOFF_S_SQ = 1.0;
        #ifdef _OPENMP
        #ifdef BUNDLE
            omp_set_num_threads(BUNDLE);
        #else
            omp_set_num_threads(num_of_threads);
        #endif
        #endif
        /* #ifdef _OPENMP
            if (num_of_threads<=16)
            {
                omp_set_num_threads(num_of_threads);
            }
            else 
            {
                if (num_of_threads<=20)
                {
                    omp_set_num_threads(16);
                }
                else
                {
                    omp_set_num_threads(num_of_threads-4);
                }
            }
        #endif */

        #ifdef enable_CUDA_CODE
            #ifdef CUDA_with_managed
            cudaMemcpy(dev_CUTOFF_SPIN, &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_CUTOFF_S_SQ, &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_spin, spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_J, J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpy(dev_J_random, J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpy(dev_h_random, h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_N_N_I, N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_no_of_black_white_sites, no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_black_white_checkerboard, black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #else
            cudaMemcpyToSymbol("dev_CUTOFF_S_SQ", &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_spin", spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_J", J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpyToSymbol("dev_J_random", J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpyToSymbol("dev_h_random", h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_N_N_I", N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_no_of_black_white_sites", no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_black_white_checkerboard", black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #endif
        #endif

        double T_temp = T;
        T = 0;
        double delta_h = del_h;

        #ifdef UPDATE_ALL_NON_EQ // if (update_all_or_checker == 0)
        {
            // spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        }
        #endif

        int cutoff_local = 0;
        int j_S, j_L;
        double *m_sweep = (double*)malloc(dim_S*hysteresis_repeat*sizeof(double));
        double *h_sweep = (double*)malloc(hysteresis_repeat*sizeof(double));
        
        for (j_S=0; j_S<dim_S*hysteresis_repeat; j_S++)
        {
            m_sweep[j_S] = 2;
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = order_start;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = h_start[j_S];
        }

        double h_jj_S = h_start[jj_S];
        printf("\nztne RFXY h axial sweep along h[%d] at T=%lf.. \n", jj_S, T);

        // long int site_i;
        // int black_or_white = BLACK_WHITE;

        repeat_sweep = 1; //int repeat_sweep = 1;
        int repeat_cond = 1;
        int restore_chkpt = 1;
        int resume_chkpt = 0;
        int is_complete = 0;
        
        
        int Spin_Order = 0;

        // print rate
        {
            #ifdef CONST_RATE
                printf("\n ====== CONSTANT RATE ====== \n");
            #endif

            #ifdef DIVIDE_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE ====== \n");
            #endif
            
            #ifdef BINARY_DIVISION
                printf("\n ====== BINARY DIVISION to LOCATE AVALANCHE ====== \n");
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                printf("\n ====== SPEEDUP/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
        }

        while (repeat_cond)
        {
            pFile_1 = fopen(output_file_name, "a");
            
            delta_h = del_h;

            #if defined (RESTORE_CHKPT_VALUE) && defined (CHECKPOINT_TIME)
                if (restore_chkpt == RESTORE_CHKPT_VALUE)
                {
                    for (j_S=0; j_S<dim_S; j_S++)
                    {
                        h[j_S] = h_start[j_S];
                    }

                    resume_chkpt = restore_checkpoint(1, TYPE_DOUBLE, dim_S, h, 0);
                    restore_checkpoint(0, TYPE_INT, 1, &is_complete, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, 1, &h_jj_S, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, 1, &delta_h, 0); // new
                    restore_checkpoint(0, TYPE_DOUBLE, 1, &order[jj_S], 0);
                    restore_checkpoint(0, TYPE_INT, 1, &repeat_sweep, 0);
                    restore_checkpoint(0, TYPE_INT, 1, &repeat_cond, 0);
                    restore_checkpoint(0, TYPE_LONGINT, 1, &h_counter, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_sweep, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, hysteresis_repeat, h_sweep, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, dim_S, m, 0);
                    restore_checkpoint(0, TYPE_VOID, dim_S, h, 1);
                    // ensemble_m();

                    restore_chkpt = !restore_chkpt;
                    if (is_complete == 1)
                    {
                        printf("\n---- Already Completed ----\n");

                        fclose(pFile_1);
                        free(m_sweep);
                        free(h_sweep);
                        T = T_temp;
                        return 1;
                    }
                }
            #endif
            
            printf("\n=========================");
            printf("\n [START] h[%d] = %.15e  ", jj_S, h_jj_S);
            printf("\n=========================\n");

            if ( resume_chkpt == 0 )
            {
                h[jj_S] = h_jj_S;
            
                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                ensemble_m();
                #ifdef EXPLORE_ENERGY_LANDSCAPE
                ensemble_E();
                #endif
                backing_up_spin();
            
                #ifdef CHECK_AVALANCHE
                    continue_avalanche();
                #else
                    continue_avalanche();
                    // cutoff_local = 0;
                    // do
                    // {
                    //     // double cutoff_local_last = cutoff_local;
                    //     cutoff_local = find_change();
                    // }
                    // while (cutoff_local > 0); // 10^-14
                #endif

                #ifdef EXPLORE_ENERGY_LANDSCAPE
                calculate_Energy_change();
                #endif

                #ifdef CHECK_AVALANCHE
                    ensemble_delta_S_squared_max();
                    ensemble_m();
                    backing_up_spin();
                    save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_start[jj_S], 2);
                #else
                    double delta_m = calculate_magnetization_change();
                    
                    backing_up_spin();
                    save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start[jj_S], 2);
                #endif
            }
            else
            {
                backing_up_spin();
                resume_chkpt = 0;
            }


            // while (h_phi * order[jj_S] <= 1.0)
            while (Spin_Order == 0)
            {
                #ifdef CHECKPOINT_TIME
                    if (get_time_if_parallel()-start_time > CHECKPOINT_TIME)
                    {
                        for (j_S=0; j_S<dim_S; j_S++)
                        {
                            h[j_S] = h_start[j_S];
                        }

                        checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &h_jj_S, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_h, 0); // new
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &order[jj_S], 0);
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_sweep, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
                        checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_sweep, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, hysteresis_repeat, h_sweep, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
                        checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);

                        fprintf(pFile_1, "===== Checkpointed here. =====\n");
                        fclose(pFile_1);
                        free(m_sweep);
                        free(h_sweep);
                        T = T_temp;
                        return -1;
                    }
                #endif

                h_jj_S = h_jj_S + order[jj_S] * delta_h;
                h[jj_S] = h_jj_S;
                
                
                // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE)
                //     backing_up_spin();
                // #endif

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                
                #ifdef CHECK_AVALANCHE
                    #ifdef CONST_RATE
                    continue_avalanche();
                    #else
                    if (delta_h <= del_h_cutoff)
                    {
                        continue_avalanche();
                    }
                    else
                    {
                        check_avalanche();
                    }
                    #endif
                #else
                    continue_avalanche();
                    // cutoff_local = 0;
                    // do
                    // {
                    //     // double cutoff_local_last = cutoff_local;
                    //     cutoff_local = find_change();
                    // }
                    // while (cutoff_local > 0); // 10^-14
                #endif
                
                #ifdef enable_CUDA_CODE
                    // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                    
                #ifdef CONST_RATE
                    const_delta_h_axis(h_jj_S, delta_h, jj_S, h_start[jj_S]);
                #endif

                #ifdef DIVIDE_BY_SLOPE
                    slope_subdivide_h_axis(h_jj_S, delta_h, jj_S, h_start[jj_S]);
                #endif
                
                #ifdef BINARY_DIVISION
                    binary_subdivide_h_axis(h_jj_S, delta_h, jj_S, h_start[jj_S]);
                #endif

                #ifdef DYNAMIC_BINARY_DIVISION
                    dynamic_binary_subdivide_h_axis(&h_jj_S, &delta_h, jj_S, h_start[jj_S]);
                #endif
                
                #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                    dynamic_binary_slope_divide_h_axis(&h_jj_S, &delta_h, jj_S, h_start[jj_S]);
                #endif
                
                
                // Start editing here
                // Start editing here
                // Start editing here
                // Start editing here
                // Start editing here

                if (finite_sweep == 0)
                {
                    Spin_Order = check_spin_direction(jj_S);
                }
                else
                {
                    if (finite_sweep == 1)
                    {
                        if (m[jj_S]*order[jj_S] > 0.99)
                        {
                            Spin_Order = 1;
                        }
                    }
                    else
                    {
                        if (fabs(h[jj_S]) >= h_abs_jj_S)
                        {
                            Spin_Order = 1;
                            h_jj_S = order[jj_S]*h_abs_jj_S;
                        }
                    }
                }
            }
            
            printf("\n=========================");
            printf("\n [END] h[%d] = %.15e  ", jj_S, h_jj_S);
            printf("\n=========================\n");
            Spin_Order = 0;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_start[j_S];
            }

            int i;
            for(i=0; i<repeat_sweep-1; i++)
            {
                repeat_cond = 0;
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    if (fabs(m_sweep[j_S+i*dim_S] - m[j_S]) > CUTOFF_SPIN )
                    {
                        repeat_cond = 1;
                    }
                }
                if (repeat_cond == 0)
                {
                    if (fabs(h_sweep[jj_S+i] - h_jj_S) < del_h)
                    {
                        break;
                    }
                    else
                    {
                        repeat_cond = 1;
                    }
                }
            }
            for(j_S=0; j_S<dim_S; j_S++)
            {
                m_sweep[j_S + (repeat_sweep-1)*dim_S] = m[j_S];
            }
            h_sweep[jj_S + (repeat_sweep-1)] = h_jj_S;

            if (repeat_cond == 0)
            {
                fprintf(pFile_1, "sweep %d - sweep %d\n", repeat_sweep, i+1);
                printf("\nsweep %d - sweep %d\n", repeat_sweep, i+1);
            }
            else
            {
                fprintf(pFile_1, "sweep %d\n", repeat_sweep);
                printf("\nsweep %d\n", repeat_sweep);
            }
            fclose(pFile_1);
            char append_string[128];
            char *pos = append_string;
            pos += sprintf(pos, "_sweep_%d", repeat_sweep);
            save_spin_config(append_string, "a", 1);
            repeat_sweep++;
            order[jj_S] = -order[jj_S];
            h_counter = 0;
            if (repeat_sweep-1 == hysteresis_repeat)
            {
                break;
            }
        }
        is_complete = 1;
        delta_h = del_h;
        // h_phi = 0.0;
        // if (update_all_or_checker == 0)
        // {
        //     free(spin_temp);
        // }
        #ifdef CHECKPOINT_TIME
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_start[j_S];
            }
            checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
            checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &h_jj_S, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_h, 0); // new
            checkpoint_backup(0, TYPE_DOUBLE, 1, &order[jj_S], 0);
            checkpoint_backup(0, TYPE_INT, 1, &repeat_sweep, 0);
            checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
            checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_sweep, 0);
            checkpoint_backup(0, TYPE_DOUBLE, hysteresis_repeat, h_sweep, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
            checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);
        #endif
        T = T_temp;
        free(m_sweep);
        free(h_sweep);
        return 0;
    }

    int zero_temp_RFXY_hysteresis_rotate_checkerboard_old(int jj_S, double order_start, double h_start, char output_file_name[])
    {
        restore_checkpoint(/* startif */-1, TYPE_DOUBLE, dim_S, h, 0); // reset static variable
        start_time = get_time_if_parallel();
        // CUTOFF_S_SQ = 4.0*pie*pie*del_phi*del_phi;

        // del_phi_cutoff = del_phi/no_of_sites;
        CUTOFF_M = 1.0/no_of_sites;
        CUTOFF_S_SQ = 1.0;
        #ifdef _OPENMP
        #ifdef BUNDLE
            omp_set_num_threads(BUNDLE);
        #else
            omp_set_num_threads(num_of_threads);
        #endif
        #endif
        /* #ifdef _OPENMP
            if (num_of_threads<=16)
            {
                omp_set_num_threads(num_of_threads);
            }
            else 
            {
                if (num_of_threads<=20)
                {
                    omp_set_num_threads(16);
                }
                else
                {
                    omp_set_num_threads(num_of_threads-4);
                }
            }
        #endif */

        #ifdef enable_CUDA_CODE
            #ifdef CUDA_with_managed
            cudaMemcpy(dev_CUTOFF_SPIN, &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_CUTOFF_S_SQ, &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_spin, spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_J, J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpy(dev_J_random, J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpy(dev_h_random, h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_N_N_I, N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_no_of_black_white_sites, no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_black_white_checkerboard, black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #else
            cudaMemcpyToSymbol("dev_CUTOFF_S_SQ", &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_spin", spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_J", J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpyToSymbol("dev_J_random", J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpyToSymbol("dev_h_random", h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_N_N_I", N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_no_of_black_white_sites", no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_black_white_checkerboard", black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #endif
        #endif
        double T_temp = T;
        T = 0;
        double delta_phi = del_phi;

        #ifdef UPDATE_ALL_NON_EQ // if (update_all_or_checker == 0)
        {
            // spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        }
        #endif

        int cutoff_local = 0;
        int j_S, j_L;
        double *m_loop = (double*)malloc(dim_S*hysteresis_repeat*sizeof(double));
        
        for (j_S=0; j_S<dim_S*hysteresis_repeat; j_S++)
        {
            m_loop[j_S] = 2;
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = order_start;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = 0;
        }
        
        double h_phi = 0;
        printf("\nztne RFXY h rotating with |h|=%lf at T=%lf.. \n", h_start, T);

        // long int site_i;
        // int black_or_white = BLACK_WHITE;

        repeat_loop = 1;
        int repeat_cond = 1;
        int restore_chkpt = 1;
        int is_complete = 0;
        
        // print rate
        {
            #ifdef CONST_RATE
                printf("\n ====== CONSTANT RATE ====== \n");
            #endif

            #ifdef DIVIDE_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE ====== \n");
            #endif
            
            #ifdef BINARY_DIVISION
                printf("\n ====== BINARY DIVISION to LOCATE AVALANCHE ====== \n");
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                printf("\n ====== SPEEDUP/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
        }
        
        while (repeat_cond)
        {
            pFile_1 = fopen(output_file_name, "a");
            h_phi = 0.0;
            // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
                delta_phi = del_phi;
            // #endif
            // for (h_phi = 0.0; h_phi * order[jj_S] <= 1.0; h_phi = h_phi + order[jj_S] * delta_phi)
            while (h_phi * order[jj_S] <= 1.0)
            {
                #ifdef CHECKPOINT_TIME
                    if (get_time_if_parallel()-start_time > CHECKPOINT_TIME)
                    {
                        if (jj_S == 0)
                        {
                            h[0] = h_start;
                            h[1] = 0.0;
                        }
                        else
                        {
                            h[0] = 0.0;
                            h[1] = h_start;
                        }
                        checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &h_phi, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_loop, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
                        checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
                        checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);

                        fprintf(pFile_1, "===== Checkpointed here. =====\n");
                        fclose(pFile_1);
                        free(m_loop);
                        T = T_temp;
                        return -1;
                    }

                    if (restore_chkpt == RESTORE_CHKPT_VALUE)
                    {
                        if (jj_S == 0)
                        {
                            h[0] = h_start;
                            h[1] = 0.0;
                        }
                        else
                        {
                            h[0] = 0.0;
                            h[1] = h_start;
                        }
                        restore_checkpoint(1, TYPE_DOUBLE, dim_S, h, 0);
                        restore_checkpoint(0, TYPE_INT, 1, &is_complete, 0);
                        restore_checkpoint(0, TYPE_DOUBLE, 1, &h_phi, 0);
                        restore_checkpoint(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
                        restore_checkpoint(0, TYPE_INT, 1, &repeat_loop, 0);
                        restore_checkpoint(0, TYPE_INT, 1, &repeat_cond, 0);
                        restore_checkpoint(0, TYPE_LONGINT, 1, &h_counter, 0);
                        restore_checkpoint(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
                        restore_checkpoint(0, TYPE_DOUBLE, dim_S, m, 0);
                        restore_checkpoint(0, TYPE_VOID, dim_S, h, 1);
                        // ensemble_m();

                        restore_chkpt = !restore_chkpt;
                        if (is_complete == 1)
                        {
                            printf("\n---- Already Completed ----\n");

                            fclose(pFile_1);
                            free(m_loop);
                            T = T_temp;
                            return 1;
                        }
                    }
                    
                #endif

                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*h_phi);
                    h[1] = h_start * sin(2*pie*h_phi);
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*h_phi);
                    h[1] = h_start * cos(2*pie*h_phi);
                }
                
                // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE)
                //     backing_up_spin();
                // #endif

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                
                #ifdef CHECK_AVALANCHE
                    #ifndef CONST_RATE
                    if (h_phi == 0.0)
                    {
                        continue_avalanche();
                    }
                    else
                    {
                        if (delta_phi <= del_phi_cutoff)
                        {
                            continue_avalanche();
                        }
                        else
                        {
                            check_avalanche();
                        }
                    }
                    #else
                    continue_avalanche();
                    #endif
                #else
                    cutoff_local = 0;
                    do
                    {
                        // double cutoff_local_last = cutoff_local;
                        cutoff_local = find_change();
                    }
                    while (cutoff_local > 0); // 10^-14
                #endif
                
                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                    
                #ifdef CONST_RATE
                    const_delta_phi(h_phi, delta_phi, jj_S, h_start);
                #endif

                #ifdef DIVIDE_BY_SLOPE
                    slope_subdivide_phi(h_phi, delta_phi, jj_S, h_start);
                #endif
                
                #ifdef BINARY_DIVISION
                    binary_subdivide_phi(h_phi, delta_phi, jj_S, h_start);
                #endif

                #ifdef DYNAMIC_BINARY_DIVISION
                    dynamic_binary_subdivide_phi(&h_phi, &delta_phi, jj_S, h_start);
                #endif
                
                #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                    dynamic_binary_slope_divide_phi(&h_phi, &delta_phi, jj_S, h_start);
                #endif


                // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
                    if (h_phi * order[jj_S] + delta_phi >= 1.0 && h_phi * order[jj_S] < 1.0)
                    {
                        delta_phi = 1.0 - h_phi * order[jj_S];
                        h_phi = 1.0;
                    }
                    else
                    {
                        h_phi = h_phi + order[jj_S] * delta_phi;
                    }
                // #else
                //     h_phi = h_phi + order[jj_S] * delta_phi;
                // #endif
            }
            printf("\n=========================");
            printf("\n  phi > 1.0 (%.15e)  ", h_phi);
            printf("\n=========================\n");
            if (jj_S == 0)
            {
                h[0] = h_start;
                h[1] = 0.0;
            }
            else
            {
                h[0] = 0.0;
                h[1] = h_start;
            }

            int i;
            for(i=0; i<repeat_loop-1; i++)
            {
                repeat_cond = 0;
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    if (fabs(m_loop[j_S+i*dim_S] - m[j_S]) > CUTOFF_SPIN )
                    {
                        repeat_cond = 1;
                    }
                }
                if (repeat_cond == 0)
                {
                    break;
                }
            }
            for(j_S=0; j_S<dim_S; j_S++)
            {
                m_loop[j_S + (repeat_loop-1)*dim_S] = m[j_S];
            }
            if (repeat_cond == 0)
            {
                fprintf(pFile_1, "loop %d - loop %d\n", repeat_loop, i+1);
                printf("\nloop %d - loop %d\n", repeat_loop, i+1);
            }
            else
            {
                fprintf(pFile_1, "loop %d\n", repeat_loop);
                printf("\nloop %d\n", repeat_loop);
            }
            fclose(pFile_1);
            char append_string[128];
            char *pos = append_string;
            pos += sprintf(pos, "_loop_%d", repeat_loop);
            save_spin_config(append_string, "a", 1);
            if (repeat_loop == hysteresis_repeat)
            {
                break;
            }
            repeat_loop++;
            // h_counter = 0;
        }
        is_complete = 1;
        h_phi = 0.0;
        // if (update_all_or_checker == 0)
        // {
        //     free(spin_temp);
        // }
        #ifdef CHECKPOINT_TIME
            if (jj_S == 0)
            {
                h[0] = h_start;
                h[1] = 0.0;
            }
            else
            {
                h[0] = 0.0;
                h[1] = h_start;
            }
            checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
            checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &h_phi, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
            checkpoint_backup(0, TYPE_INT, 1, &repeat_loop, 0);
            checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
            checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
            checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);
        #endif
        T = T_temp;
        free(m_loop);
        return 0;
    }

    int zero_temp_RFXY_hysteresis_rotate_checkerboard(int jj_S, double order_start, double h_start, char output_file_name[])
    {
        restore_checkpoint(/* startif */-1, TYPE_DOUBLE, dim_S, h, 0); // reset static variable
        start_time = get_time_if_parallel();
        // CUTOFF_S_SQ = 4.0*pie*pie*del_phi*del_phi;

        // del_phi_cutoff = del_phi/no_of_sites;
        
        CUTOFF_M = 1.0/no_of_sites;
        CUTOFF_S_SQ = 1.0;
        #ifdef _OPENMP
        #ifdef BUNDLE
            omp_set_num_threads(BUNDLE);
        #else
            omp_set_num_threads(num_of_threads);
        #endif
        #endif
        /* #ifdef _OPENMP
            if (num_of_threads<=16)
            {
                omp_set_num_threads(num_of_threads);
            }
            else 
            {
                if (num_of_threads<=20)
                {
                    omp_set_num_threads(16);
                }
                else
                {
                    omp_set_num_threads(num_of_threads-4);
                }
            }
        #endif */

        #ifdef enable_CUDA_CODE
            #ifdef CUDA_with_managed
            cudaMemcpy(dev_CUTOFF_SPIN, &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_CUTOFF_S_SQ, &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_spin, spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_J, J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpy(dev_J_random, J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpy(dev_h_random, h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_N_N_I, N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_no_of_black_white_sites, no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_black_white_checkerboard, black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #else
            cudaMemcpyToSymbol("dev_CUTOFF_S_SQ", &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_spin", spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_J", J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpyToSymbol("dev_J_random", J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpyToSymbol("dev_h_random", h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_N_N_I", N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_no_of_black_white_sites", no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_black_white_checkerboard", black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #endif
        #endif

        double T_temp = T;
        T = 0;
        double delta_phi = del_phi;

        
        #ifdef UPDATE_ALL_NON_EQ // if (update_all_or_checker == 0)
        {
            // spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        }
        #endif


        int cutoff_local = 0;
        int j_S, j_L;
        double *m_loop = (double*)malloc(dim_S*hysteresis_repeat*sizeof(double));
        
        for (j_S=0; j_S<dim_S*hysteresis_repeat; j_S++)
        {
            m_loop[j_S] = 2;
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = order_start;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = 0;
        }

        double h_phi = 0;
        printf("\nztne RFXY h rotating with |h|=%lf at T=%lf.. \n", h_start, T);

        // long int site_i;
        // int black_or_white = BLACK_WHITE;

        repeat_loop = 1;
        int repeat_cond = 1;
        int restore_chkpt = 1;
        int resume_chkpt = 0;
        int is_complete = 0;
        
        int Spin_Order = 0;

        // print rate
        {
            #ifdef CONST_RATE
                printf("\n ====== CONSTANT RATE ====== \n");
            #endif

            #ifdef DIVIDE_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE ====== \n");
            #endif
            
            #ifdef BINARY_DIVISION
                printf("\n ====== BINARY DIVISION to LOCATE AVALANCHE ====== \n");
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                printf("\n ====== SPEEDUP/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
        }

        while (repeat_cond)
        {
            pFile_1 = fopen(output_file_name, "a");
            h_phi = 0.0;
            delta_phi = del_phi;

            #if defined (RESTORE_CHKPT_VALUE) && defined (CHECKPOINT_TIME)
                if (restore_chkpt == RESTORE_CHKPT_VALUE)
                {
                    if (jj_S == 0)
                    {
                        h[0] = h_start;
                        h[1] = 0.0;
                    }
                    else
                    {
                        h[0] = 0.0;
                        h[1] = h_start;
                    }

                    resume_chkpt = restore_checkpoint(1, TYPE_DOUBLE, dim_S, h, 0);
                    restore_checkpoint(0, TYPE_INT, 1, &is_complete, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, 1, &h_phi, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
                    restore_checkpoint(0, TYPE_INT, 1, &repeat_loop, 0);
                    restore_checkpoint(0, TYPE_INT, 1, &repeat_cond, 0);
                    restore_checkpoint(0, TYPE_LONGINT, 1, &h_counter, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, dim_S, m, 0);
                    restore_checkpoint(0, TYPE_VOID, dim_S, h, 1);
                    // ensemble_m();

                    restore_chkpt = !restore_chkpt;
                    if (is_complete == 1)
                    {
                        printf("\n---- Already Completed ----\n");

                        fclose(pFile_1);
                        free(m_loop);
                        T = T_temp;
                        return 1;
                    }
                }
            #endif
            
            printf("\n=========================");
            printf("\n [START] phi(h[%d]) = %.15e  ", jj_S, h_phi);
            printf("\n=========================\n");

            if ( resume_chkpt == 0 )
            {
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*h_phi);
                    h[1] = h_start * sin(2*pie*h_phi);
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*h_phi);
                    h[1] = h_start * cos(2*pie*h_phi);
                }

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                ensemble_m();
                #ifdef EXPLORE_ENERGY_LANDSCAPE
                ensemble_E();
                #endif
                backing_up_spin();
            
                #ifdef CHECK_AVALANCHE
                    continue_avalanche();
                #else
                    continue_avalanche();
                    // cutoff_local = 0;
                    // do
                    // {
                    //     // double cutoff_local_last = cutoff_local;
                    //     cutoff_local = find_change();
                    // }
                    // while (cutoff_local > 0); // 10^-14
                #endif

                #ifdef EXPLORE_ENERGY_LANDSCAPE
                calculate_Energy_change();
                #endif

                #ifdef CHECK_AVALANCHE
                    ensemble_delta_S_squared_max();
                    ensemble_m();
                    backing_up_spin();
                    save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", &h_start, 1);
                #else
                    double delta_m = calculate_magnetization_change();
                    backing_up_spin();
                    save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", &h_start, 1);
                #endif
            }
            else
            {
                backing_up_spin();
                resume_chkpt = 0;
            }


            // while (h_phi * order[jj_S] <= 1.0)
            while (h_phi * order[jj_S] < 1.0)
            {
                #ifdef CHECKPOINT_TIME
                    if (get_time_if_parallel()-start_time > CHECKPOINT_TIME)
                    {
                        if (jj_S == 0)
                        {
                            h[0] = h_start;
                            h[1] = 0.0;
                        }
                        else
                        {
                            h[0] = 0.0;
                            h[1] = h_start;
                        }

                        checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &h_phi, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_loop, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
                        checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
                        checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);

                        fprintf(pFile_1, "===== Checkpointed here. =====\n");
                        fclose(pFile_1);
                        free(m_loop);
                        T = T_temp;
                        return -1;
                    }
                #endif


                if (h_phi * order[jj_S] + delta_phi >= 1.0 && h_phi * order[jj_S] < 1.0)
                {
                    delta_phi = 1.0 - h_phi * order[jj_S];
                    h_phi = 1.0;
                }
                else
                {
                    h_phi = h_phi + order[jj_S] * delta_phi;
                }

                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*h_phi);
                    h[1] = h_start * sin(2*pie*h_phi);
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*h_phi);
                    h[1] = h_start * cos(2*pie*h_phi);
                }
                
                
                // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE)
                //     backing_up_spin();
                // #endif

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                
                #ifdef CHECK_AVALANCHE
                    #ifdef CONST_RATE
                    continue_avalanche();
                    #else
                    if (delta_phi <= del_phi_cutoff)
                    {
                        continue_avalanche();
                    }
                    else
                    {
                        check_avalanche();
                    }
                    #endif
                #else
                    continue_avalanche();
                    // cutoff_local = 0;
                    // do
                    // {
                    //     // double cutoff_local_last = cutoff_local;
                    //     cutoff_local = find_change();
                    // }
                    // while (cutoff_local > 0); // 10^-14
                #endif
                
                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                    
                #ifdef CONST_RATE
                    const_delta_phi(h_phi, delta_phi, jj_S, h_start);
                #endif

                #ifdef DIVIDE_BY_SLOPE
                    slope_subdivide_phi(h_phi, delta_phi, jj_S, h_start);
                #endif
                
                #ifdef BINARY_DIVISION
                    binary_subdivide_phi(h_phi, delta_phi, jj_S, h_start);
                #endif

                #ifdef DYNAMIC_BINARY_DIVISION
                    dynamic_binary_subdivide_phi(&h_phi, &delta_phi, jj_S, h_start);
                #endif
                
                #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                    dynamic_binary_slope_divide_phi(&h_phi, &delta_phi, jj_S, h_start);
                #endif
                

                // Start editing here
                // Start editing here
                // Start editing here
                // Start editing here
                // Start editing here
                // Spin_Order = check_spin_direction(jj_S);
            }
            printf("\n=========================");
            printf("\n [END] phi(h[%d]) = %.15e  ", jj_S, h_phi);
            printf("\n=========================\n");
            if (jj_S == 0)
            {
                h[0] = h_start;
                h[1] = 0.0;
            }
            else
            {
                h[0] = 0.0;
                h[1] = h_start;
            }

            int i;
            for(i=0; i<repeat_loop-1; i++)
            {
                repeat_cond = 0;
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    if (fabs(m_loop[j_S+i*dim_S] - m[j_S]) > CUTOFF_SPIN )
                    {
                        repeat_cond = 1;
                    }
                }
                if (repeat_cond == 0)
                {
                    break;
                }
            }
            for(j_S=0; j_S<dim_S; j_S++)
            {
                m_loop[j_S + (repeat_loop-1)*dim_S] = m[j_S];
            }

            if (repeat_cond == 0)
            {
                fprintf(pFile_1, "loop %d - loop %d\n", repeat_loop, i+1);
                printf("\nloop %d - loop %d\n", repeat_loop, i+1);
            }
            else
            {
                fprintf(pFile_1, "loop %d\n", repeat_loop);
                printf("\nloop %d\n", repeat_loop);
            }
            fclose(pFile_1);
            char append_string[128];
            char *pos = append_string;
            pos += sprintf(pos, "_loop_%d", repeat_loop);
            save_spin_config(append_string, "a", 1);
            repeat_loop++;
            h_counter = 0;
            if (repeat_loop-1 == hysteresis_repeat)
            {
                break;
            }
        }
        is_complete = 1;
        h_phi = 0.0;
        delta_phi = del_phi;
        // h_phi = 0.0;
        // if (update_all_or_checker == 0)
        // {
        //     free(spin_temp);
        // }
        #ifdef CHECKPOINT_TIME
            if (jj_S == 0)
            {
                h[0] = h_start;
                h[1] = 0.0;
            }
            else
            {
                h[0] = 0.0;
                h[1] = h_start;
            }
            checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
            checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &h_phi, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
            checkpoint_backup(0, TYPE_INT, 1, &repeat_loop, 0);
            checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
            checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
            checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);
        #endif
        T = T_temp;
        free(m_loop);
        return 0;
    }

//====================  RFXY FTNE                     ====================//

    int finite_temp_RFXY_hysteresis_axis_checkerboard_old(int jj_S, double order_start, double* h_sweep_abs)
    {
        restore_checkpoint(/* startif */-1, TYPE_DOUBLE, dim_S, h, 0); // reset static variable
        start_time = get_time_if_parallel();
        // CUTOFF_S_SQ = 4.0*del_h*del_h;
        // del_h_cutoff = del_h/no_of_sites;
        CUTOFF_M = 1.0/(double)no_of_sites;
        CUTOFF_S_SQ = 1.0;
        #ifdef _OPENMP
        #ifdef BUNDLE
            omp_set_num_threads(BUNDLE);
        #else
            omp_set_num_threads(num_of_threads);
        #endif
        #endif
        /* #ifdef _OPENMP
            if (num_of_threads<=16)
            {
                omp_set_num_threads(num_of_threads);
            }
            else 
            {
                if (num_of_threads<=20)
                {
                    omp_set_num_threads(16);
                }
                else
                {
                    omp_set_num_threads(num_of_threads-4);
                }
            }
        #endif */
        
        #ifdef enable_CUDA_CODE
            #ifdef CUDA_with_managed
            cudaMemcpy(dev_CUTOFF_SPIN, &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_CUTOFF_S_SQ, &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_spin, spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_J, J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpy(dev_J_random, J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpy(dev_h_random, h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_N_N_I, N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_no_of_black_white_sites, no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_black_white_checkerboard, black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #else
            cudaMemcpyToSymbol("dev_CUTOFF_SPIN", &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_CUTOFF_S_SQ", &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_spin", spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_J", J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpyToSymbol("dev_J_random", J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpyToSymbol("dev_h_random", h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_N_N_I", N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_no_of_black_white_sites", no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_black_white_checkerboard", black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #endif
        #endif

        // T != 0;
        ax_ro = 0;
        or_ho_ra = 0;
        int cutoff_local = 0;
        int j_S, j_L;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[jj_S] = order_start;
            }
            else
            {
                order[j_S] = 0;
            }
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = h_sweep_abs[j_S];
        }
        // double h_start = order[jj_S]*(h_max+h_i_max);
        double h_start, h_end;
        double delta_h;
        double sigma_h_trnsvrs = 0.0;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S != jj_S)
            {
                sigma_h_trnsvrs += sigma_h[j_S] * sigma_h[j_S];
            }
        }
        sigma_h_trnsvrs = sqrt(sigma_h_trnsvrs);
        if (sigma_h_trnsvrs > 0.0)
        {
            h_start = -order[jj_S]*(sigma_h_trnsvrs);
            // delta_h = sigma_h_trnsvrs*del_h;
            h_order = 0;
            r_order = 0;
            // initialize_spin_config();
        }
        else
        {
            if (sigma_h[jj_S] == 0.0)
            {
                h_start = -order[jj_S]*(h_max);
                // delta_h = h_max*del_h;
            }
            else
            {
                // h[!jj_S] = sigma_h[jj_S]*del_h_cutoff;
                // h[!jj_S] = 0.10*del_h_cutoff;
                // h_start = order[jj_S]*(h_max);
                if (h_i_max >= -h_i_min)
                {
                    h_start = -order[jj_S]*(h_max + h_i_max);
                }
                else
                {
                    h_start = -order[jj_S]*(h_max - h_i_min);
                }
                // delta_h = (h_i_max + h_max)*del_h;
            }
            
            h_order = 0;
            r_order = 1;
            // initialize_spin_config();
        }
        initialize_spin_config();
        delta_h = del_h;
        h_end = -h_start;
        

        printf("\nftne RFXY looping along h[%d] at T=%lf.. \n", jj_S,  T);

        ensemble_m();
        ensemble_E();

        // set output_variable_name=1 here
        {
            output_prepend = 1;
            output_h = 1;
            output_m = 1;
            #ifdef CHECK_AVALANCHE
            output_delta_S_abs = 1;
            output_delta_S_squared = 1;
            #else
            output_delta_M = 1;
            #endif

            #ifdef EXPLORE_ENERGY_LANDSCAPE
            output_E = 1;
            output_delta_E = 1;
            #endif
            output_append = 1;
        }
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hys_axis_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%lf", T);
            /* pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); */
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%lf_%lf)", h_start, delta_h);
                }
                else
                {
                    pos += sprintf(pos, "%lf", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%.3f", order[j_S]);
            // }
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_h%d_r%d", h_order, r_order);
            #ifdef enable_CUDA_CODE
            pos += sprintf(pos, "_cuda");
            #endif
            pos += sprintf(pos, ".dat");

        }
        
        // column labels and parameters
        output_param_file(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        #ifdef UPDATE_ALL_NON_EQ
        {
            printf("\nUpdating all sites simultaneously.. \n");
            
            fprintf(pFile_1, "==================================================================================\n");
            fprintf(pFile_1, "Updating all sites simultaneously.. \n");
            fprintf(pFile_1, "==================================================================================\n");
        }
        #endif
        // else
        #ifdef UPDATE_CHKR_NON_EQ
        {
            printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

            fprintf(pFile_1, "==================================================================================\n");
            fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
            fprintf(pFile_1, "==================================================================================\n");
        }
        #endif
        fclose(pFile_1);
        #ifdef CHECK_AVALANCHE
        output_label(output_file_0, "h[%d]\t", "max(|delta{S}|)\t");
        #else
        output_label(output_file_0, "h[%d]\t", "|delta{m[%d]}|\t");
        #endif


        #ifdef UPDATE_ALL_NON_EQ // if (update_all_or_checker == 0)
        {
            // spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        }
        #endif

        long int site_i;
        int black_or_white = BLACK_WHITE;
        double h_jj_S;


        pFile_1 = fopen(output_file_0, "a");
        // print statements:
        {
            // fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            // fprintf(pFile_1, "\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                    // fprintf(pFile_1, ",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                    // fprintf(pFile_1, "%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%.3f", order[j_S]);
                    // fprintf(pFile_1, "%.3f", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
            // fprintf(pFile_1, "}, %d, %d)\n", h_order, r_order);
            // fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
        }
        h_jj_S = h_start;

        // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
        //     delta_h = del_h_cutoff*fabs(h_start);
        // #endif

        // print rate
        {
            #ifdef CONST_RATE
                printf("\n ====== CONSTANT RATE ====== \n");
            #endif

            #ifdef DIVIDE_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE ====== \n");
            #endif
            
            #ifdef BINARY_DIVISION
                printf("\n ====== BINARY DIVISION to LOCATE AVALANCHE ====== \n");
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                printf("\n ====== SPEEDUP/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
        }

        printf("\n=========================");
        printf("\n  h[%d] = h_start (%.15e)  ", jj_S, h_jj_S);
        printf("\n=========================\n");
        {
            h[jj_S] = h_jj_S;
            ensemble_m();
            backing_up_spin();
            // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE) 
            //     backing_up_spin();
            // #endif

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #else
                cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #endif
            #endif

            
            #ifdef CHECK_AVALANCHE
                continue_avalanche();
            #else
                cutoff_local = 0;
                do
                {
                    // double cutoff_local_last = cutoff_local;
                    cutoff_local = find_change();
                }
                while (cutoff_local > 0); // 10^-14
            #endif
            
            
            #ifdef CHECK_AVALANCHE
                ensemble_delta_S_squared_max();
                ensemble_m();
                backing_up_spin();
                save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_sweep_abs[jj_S], 2);
            #else
                double delta_m = calculate_magnetization_change();
                backing_up_spin();
                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_sweep_abs[jj_S], 2);
            #endif


        }
        // for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
        while (order[jj_S] * h_jj_S < order[jj_S] * h_end)
        {
            // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
            // if (h_phi * order[jj_S] + delta_phi >= 1.0 && h_phi * order[jj_S] < 1.0)
            if (h_jj_S * order[jj_S] + delta_h >= h_end * order[jj_S] )
            {
                // delta_phi = 1.0 - h_phi * order[jj_S];
                delta_h = (h_end - h_jj_S) * order[jj_S];
                // h_phi = 1.0;
                h_jj_S = h_end;
            }
            else
            {
                h_jj_S = h_jj_S + order[jj_S] * delta_h;
            }
            // #else
            //     h_jj_S = h_jj_S - order[jj_S] * delta_h;
            // #endif
            h[jj_S] = h_jj_S;
            
            // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE) 
            //     backing_up_spin();
            // #endif

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #else
                cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #endif
            #endif
            
            
            #ifdef CHECK_AVALANCHE
                #ifdef CONST_RATE
                continue_avalanche();
                Monte_Carlo_Sweep(1, /* MC_algo_type */0, MC_update_type);
                continue_avalanche();
                #else
                if (delta_h <= del_h_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }
                #endif
            #else
                cutoff_local = 0;
                do
                {
                    // double cutoff_local_last = cutoff_local;
                    cutoff_local = find_change();
                }
                while (cutoff_local > 0); // 10^-14
            #endif
            
            #ifdef enable_CUDA_CODE
            // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
            #endif
            
            
            #ifdef CONST_RATE
                const_delta_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif

            #ifdef DIVIDE_BY_SLOPE
                slope_subdivide_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
            #ifdef BINARY_DIVISION
                binary_subdivide_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                dynamic_binary_subdivide_h_axis(&h_jj_S, &delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                dynamic_binary_slope_divide_h_axis(&h_jj_S, &delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
        }
        fclose(pFile_1);
        printf("\n=========================");
        printf("\n  |h[%d]| >= |h_end| (%.15e)  ", jj_S, h_jj_S);
        printf("\n=========================\n");

        // ----------------------------------------------//
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[j_S] = -order_start;
            }
            else
            {
                order[j_S] = 0;
            }
        }
        // h_start = order[jj_S]*(h_max+h_i_max);
        if (sigma_h_trnsvrs > 0.0)
        {
            h_start = -order[jj_S]*(sigma_h_trnsvrs);
            // initialize_spin_config();
        }
        else
        {
            if (sigma_h[jj_S] == 0.0)
            {
                h_start = -order[jj_S]*(h_max);
            }
            else
            {
                // h_start = order[jj_S]*(h_max);
                if (h_i_max >= -h_i_min)
                {
                    h_start = -order[jj_S]*(h_max + h_i_max);
                }
                else
                {
                    h_start = -order[jj_S]*(h_max - h_i_min);
                }
            }
            
        }
        delta_h = del_h;
        h_end = -h_start;
        // h_order = 0;
        // r_order = 1;
        // initialize_spin_config();
        
        ensemble_m();
        ensemble_E();

        pFile_1 = fopen(output_file_0, "a");
        // print statements:
        {
            // fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            // fprintf(pFile_1, "\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                    // fprintf(pFile_1, ",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                    // fprintf(pFile_1, "%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                    // fprintf(pFile_1, "%.3f", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
            // fprintf(pFile_1, "}, %d, %d)\n", h_order, r_order);
            // fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
        }
        h_jj_S = h_start;

        // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
        //     delta_h = del_h_cutoff*fabs(h_start);
        // #endif
        printf("\n=========================");
        printf("\n  h[%d] = h_start (%.15e)  ", jj_S, h_jj_S);
        printf("\n=========================\n");
        {
            h[jj_S] = h_jj_S;
            ensemble_m();
            backing_up_spin();
            // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE) 
            //     backing_up_spin();
            // #endif

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #else
                cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #endif
            #endif

            
            #ifdef CHECK_AVALANCHE
                continue_avalanche();
            #else
                cutoff_local = 0;
                do
                {
                    // double cutoff_local_last = cutoff_local;
                    cutoff_local = find_change();
                }
                while (cutoff_local > 0); // 10^-14
            #endif
            
            
            #ifdef CHECK_AVALANCHE
                ensemble_delta_S_squared_max();
                ensemble_m();
                backing_up_spin();
                save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_sweep_abs[jj_S], 2);
            #else
                double delta_m = calculate_magnetization_change();
                backing_up_spin();
                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_sweep_abs[jj_S], 2);
            #endif

        }
        // for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
        while (order[jj_S] * h_jj_S < order[jj_S] * h_end)
        {
            // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
            if (h_jj_S * order[jj_S] + delta_h >= h_end * order[jj_S])
            {
                // delta_phi = 1.0 - h_phi * order[jj_S];
                delta_h = (h_end - h_jj_S) * order[jj_S];
                // h_phi = 1.0;
                h_jj_S = h_end;
            }
            else
            {
                h_jj_S = h_jj_S + order[jj_S] * delta_h;
            }
            // #else
            //     h_jj_S = h_jj_S - order[jj_S] * delta_h;
            // #endif
            h[jj_S] = h_jj_S;
            
            // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE) 
            //     backing_up_spin();
            // #endif

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #else
                cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #endif
            #endif

            
            #ifdef CHECK_AVALANCHE
                #ifdef CONST_RATE
                continue_avalanche();
                #else
                if (delta_h <= del_h_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }
                #endif
            #else
                cutoff_local = 0;
                do
                {
                    // double cutoff_local_last = cutoff_local;
                    cutoff_local = find_change();
                }
                while (cutoff_local > 0); // 10^-14
            #endif

            #ifdef enable_CUDA_CODE
            // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
            #endif


            #ifdef CONST_RATE
                const_delta_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif

            #ifdef DIVIDE_BY_SLOPE
                slope_subdivide_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
            #ifdef BINARY_DIVISION
                binary_subdivide_h_axis(h_jj_S, delta_h, jj_S, order[jj_S]*h_start);
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                dynamic_binary_subdivide_h_axis(&h_jj_S, &delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                dynamic_binary_slope_divide_h_axis(&h_jj_S, &delta_h, jj_S, order[jj_S]*h_start);
            #endif
            
        }
        fclose(pFile_1);
        printf("\n=========================");
        printf("\n  |h[%d]| >= |h_end| (%.15e)  ", jj_S, h_jj_S);
        printf("\n=========================\n");

        // if (update_all_or_checker == 0)
        // {
        //     free(spin_temp);
        // }
        return 0;
    }

    int finite_temp_RFXY_hysteresis_axis_checkerboard(int jj_S, double order_start, double* h_start, double h_abs_jj_S, int finite_sweep, char output_file_name[])
    {
        restore_checkpoint(/* startif */-1, TYPE_DOUBLE, dim_S, h, 0); // reset static variable
        start_time = get_time_if_parallel();
        // CUTOFF_S_SQ = 4.0*pie*pie*del_phi*del_phi;

        // del_phi_cutoff = del_phi/no_of_sites;
        Spin_Saturation_Limit = sqrt(3.0)/2.0;
        CUTOFF_M = 1.0/no_of_sites;
        CUTOFF_S_SQ = 1.0;
        #ifdef _OPENMP
        #ifdef BUNDLE
            omp_set_num_threads(BUNDLE);
        #else
            omp_set_num_threads(num_of_threads);
        #endif
        #endif
        /* #ifdef _OPENMP
            if (num_of_threads<=16)
            {
                omp_set_num_threads(num_of_threads);
            }
            else 
            {
                if (num_of_threads<=20)
                {
                    omp_set_num_threads(16);
                }
                else
                {
                    omp_set_num_threads(num_of_threads-4);
                }
            }
        #endif */

        #ifdef enable_CUDA_CODE
            #ifdef CUDA_with_managed
            cudaMemcpy(dev_CUTOFF_SPIN, &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_CUTOFF_S_SQ, &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_spin, spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_J, J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpy(dev_J_random, J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpy(dev_h_random, h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_N_N_I, N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_no_of_black_white_sites, no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_black_white_checkerboard, black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #else
            cudaMemcpyToSymbol("dev_CUTOFF_S_SQ", &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_spin", spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_J", J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpyToSymbol("dev_J_random", J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpyToSymbol("dev_h_random", h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_N_N_I", N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_no_of_black_white_sites", no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_black_white_checkerboard", black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #endif
        #endif

        // T != 0;
        double delta_h = del_h;

        #ifdef UPDATE_ALL_NON_EQ // if (update_all_or_checker == 0)
        {
            // spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        }
        #endif

        int cutoff_local = 0;
        int j_S, j_L;
        double *m_sweep = (double*)malloc(dim_S*hysteresis_repeat*sizeof(double));
        double *h_sweep = (double*)malloc(hysteresis_repeat*sizeof(double));
        
        for (j_S=0; j_S<dim_S*hysteresis_repeat; j_S++)
        {
            m_sweep[j_S] = 2;
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = order_start;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = h_start[j_S];
        }

        double h_jj_S = h_start[jj_S];
        printf("\nftne RFXY h axial sweep along h[%d] at T=%lf.. \n", jj_S, T);

        // long int site_i;
        // int black_or_white = BLACK_WHITE;

        repeat_sweep = 1; //int repeat_sweep = 1;
        int repeat_cond = 1;
        int restore_chkpt = 1;
        int resume_chkpt = 0;
        int is_complete = 0;
        
        
        int Spin_Order = 0;

        // print rate
        {
            #ifdef CONST_RATE
                printf("\n ====== CONSTANT RATE ====== \n");
            #endif

            #ifdef DIVIDE_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE ====== \n");
            #endif
            
            #ifdef BINARY_DIVISION
                printf("\n ====== BINARY DIVISION to LOCATE AVALANCHE ====== \n");
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                printf("\n ====== SPEEDUP/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
        }

        while (repeat_cond)
        {
            pFile_1 = fopen(output_file_name, "a");
            
            delta_h = del_h;

            #if defined (RESTORE_CHKPT_VALUE) && defined (CHECKPOINT_TIME)
                if (restore_chkpt == RESTORE_CHKPT_VALUE)
                {
                    for (j_S=0; j_S<dim_S; j_S++)
                    {
                        h[j_S] = h_start[j_S];
                    }

                    resume_chkpt = restore_checkpoint(1, TYPE_DOUBLE, dim_S, h, 0);
                    restore_checkpoint(0, TYPE_INT, 1, &is_complete, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, 1, &h_jj_S, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, 1, &delta_h, 0); // new
                    restore_checkpoint(0, TYPE_DOUBLE, 1, &order[jj_S], 0);
                    restore_checkpoint(0, TYPE_INT, 1, &repeat_sweep, 0);
                    restore_checkpoint(0, TYPE_INT, 1, &repeat_cond, 0);
                    restore_checkpoint(0, TYPE_LONGINT, 1, &h_counter, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_sweep, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, hysteresis_repeat, h_sweep, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, dim_S, m, 0);
                    restore_checkpoint(0, TYPE_VOID, dim_S, h, 1);
                    // ensemble_m();

                    restore_chkpt = !restore_chkpt;
                    if (is_complete == 1)
                    {
                        printf("\n---- Already Completed ----\n");

                        fclose(pFile_1);
                        free(m_sweep);
                        free(h_sweep);
                        return 1;
                    }
                }
            #endif

            printf("\n=========================");
            printf("\n [START] h[%d] = %.15e  ", jj_S, h_jj_S);
            printf("\n=========================\n");

            if ( resume_chkpt == 0 )
            {
                h[jj_S] = h_jj_S;
            
                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                ensemble_m();
                #ifdef EXPLORE_ENERGY_LANDSCAPE
                ensemble_E();
                #endif
                backing_up_spin();
            
                #ifdef CHECK_AVALANCHE
                    continue_avalanche();
                #else
                    continue_avalanche();
                    // cutoff_local = 0;
                    // do
                    // {
                    //     // double cutoff_local_last = cutoff_local;
                    //     cutoff_local = find_change();
                    // }
                    // while (cutoff_local > 0); // 10^-14
                #endif

                #ifdef EXPLORE_ENERGY_LANDSCAPE
                calculate_Energy_change();
                #endif

                #ifdef CHECK_AVALANCHE
                    ensemble_delta_S_squared_max();
                    ensemble_m();
                    backing_up_spin();
                    save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_j", &h_start[jj_S], 2);
                #else
                    double delta_m = calculate_magnetization_change();
                    backing_up_spin();
                    save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_j", &h_start[jj_S], 2);
                #endif
            }
            else
            {
                backing_up_spin();
                resume_chkpt = 0;
            }

            // while (h_phi * order[jj_S] <= 1.0)
            while (Spin_Order == 0)
            {
                #ifdef CHECKPOINT_TIME
                    if (get_time_if_parallel()-start_time > CHECKPOINT_TIME)
                    {
                        for (j_S=0; j_S<dim_S; j_S++)
                        {
                            h[j_S] = h_start[j_S];
                        }

                        checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &h_jj_S, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_h, 0); // new
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &order[jj_S], 0);
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_sweep, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
                        checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_sweep, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, hysteresis_repeat, h_sweep, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
                        checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);

                        fprintf(pFile_1, "===== Checkpointed here. =====\n");
                        fclose(pFile_1);
                        free(m_sweep);
                        free(h_sweep);
                        return -1;
                    }
                #endif

                h_jj_S = h_jj_S + order[jj_S] * delta_h;
                h[jj_S] = h_jj_S;
                
                
                // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE)
                //     backing_up_spin();
                // #endif

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                
                #ifdef CHECK_AVALANCHE
                    #ifdef CONST_RATE
                    continue_avalanche();
                    Monte_Carlo_Sweep(1, /* MC_algo_type */0, MC_update_type);
                    continue_avalanche();
                    #else
                    if (delta_h <= del_h_cutoff)
                    {
                        continue_avalanche();
                    }
                    else
                    {
                        check_avalanche();
                    }
                    #endif
                #else
                    continue_avalanche();
                    Monte_Carlo_Sweep(1, /* MC_algo_type */0, MC_update_type);
                    continue_avalanche();
                    // cutoff_local = 0;
                    // do
                    // {
                    //     // double cutoff_local_last = cutoff_local;
                    //     cutoff_local = find_change();
                    // }
                    // while (cutoff_local > 0); // 10^-14
                #endif
                
                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                    
                #ifdef CONST_RATE
                    const_delta_h_axis(h_jj_S, delta_h, jj_S, h_start[jj_S]);
                #endif

                #ifdef DIVIDE_BY_SLOPE
                    slope_subdivide_h_axis(h_jj_S, delta_h, jj_S, h_start[jj_S]);
                #endif
                
                #ifdef BINARY_DIVISION
                    binary_subdivide_h_axis(h_jj_S, delta_h, jj_S, h_start[jj_S]);
                #endif

                #ifdef DYNAMIC_BINARY_DIVISION
                    dynamic_binary_subdivide_h_axis(&h_jj_S, &delta_h, jj_S, h_start[jj_S]);
                #endif
                
                #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                    dynamic_binary_slope_divide_h_axis(&h_jj_S, &delta_h, jj_S, h_start[jj_S]);
                #endif
                
                
                // Start editing here
                // Start editing here
                // Start editing here
                // Start editing here
                // Start editing here

                if (finite_sweep == 0)
                {
                    Spin_Order = check_spin_direction(jj_S);
                }
                else
                {
                    if (finite_sweep == 1)
                    {
                        if (m[jj_S]*order[jj_S] > 0.99)
                        {
                            Spin_Order = 1;
                        }
                    }
                    else
                    {
                        if (fabs(h[jj_S]) >= h_abs_jj_S)
                        {
                            Spin_Order = 1;
                            h_jj_S = order[jj_S]*h_abs_jj_S;
                        }
                    }
                }
            }
            
            printf("\n=========================");
            printf("\n [END] h[%d] = %.15e  ", jj_S, h_jj_S);
            printf("\n=========================\n");
            Spin_Order = 0;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_start[j_S];
            }

            int i;
            for(i=0; i<repeat_sweep-1; i++)
            {
                repeat_cond = 0;
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    if (fabs(m_sweep[j_S+i*dim_S] - m[j_S]) > CUTOFF_SPIN )
                    {
                        repeat_cond = 1;
                    }
                }
                if (repeat_cond == 0)
                {
                    if (fabs(h_sweep[jj_S+i] - h_jj_S) < del_h)
                    {
                        break;
                    }
                    else
                    {
                        repeat_cond = 1;
                    }
                }
            }
            for(j_S=0; j_S<dim_S; j_S++)
            {
                m_sweep[j_S + (repeat_sweep-1)*dim_S] = m[j_S];
            }
            h_sweep[jj_S + (repeat_sweep-1)] = h_jj_S;

            if (repeat_cond == 0)
            {
                fprintf(pFile_1, "sweep %d - sweep %d\n", repeat_sweep, i+1);
                printf("\nsweep %d - sweep %d\n", repeat_sweep, i+1);
            }
            else
            {
                fprintf(pFile_1, "sweep %d\n", repeat_sweep);
                printf("\nsweep %d\n", repeat_sweep);
            }
            fclose(pFile_1);
            char append_string[128];
            char *pos = append_string;
            pos += sprintf(pos, "_sweep_%d", repeat_sweep);
            save_spin_config(append_string, "a", 1);
            repeat_sweep++;
            order[jj_S] = -order[jj_S];
            h_counter = 0;
            if (repeat_sweep-1 == hysteresis_repeat)
            {
                break;
            }
        }
        is_complete = 1;
        delta_h = del_h;
        // h_phi = 0.0;
        // if (update_all_or_checker == 0)
        // {
        //     free(spin_temp);
        // }
        #ifdef CHECKPOINT_TIME
            for (j_S=0; j_S<dim_S; j_S++)
            {
                h[j_S] = h_start[j_S];
            }
            checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
            checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &h_jj_S, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_h, 0); // new
            checkpoint_backup(0, TYPE_DOUBLE, 1, &order[jj_S], 0);
            checkpoint_backup(0, TYPE_INT, 1, &repeat_sweep, 0);
            checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
            checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_sweep, 0);
            checkpoint_backup(0, TYPE_DOUBLE, hysteresis_repeat, h_sweep, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
            checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);
        #endif
        free(m_sweep);
        free(h_sweep);
        return 0;
    }

    int finite_temp_RFXY_hysteresis_rotate_checkerboard_old(int jj_S, double order_start, double h_start, char output_file_name[])
    {
        restore_checkpoint(/* startif */-1, TYPE_DOUBLE, dim_S, h, 0); // reset static variable
        start_time = get_time_if_parallel();
        // CUTOFF_S_SQ = 4.0*pie*pie*del_phi*del_phi;

        // del_phi_cutoff = del_phi/no_of_sites;
        CUTOFF_M = 1.0/no_of_sites;
        CUTOFF_S_SQ = 1.0;
        #ifdef _OPENMP
        #ifdef BUNDLE
            omp_set_num_threads(BUNDLE);
        #else
            omp_set_num_threads(num_of_threads);
        #endif
        #endif
        /* #ifdef _OPENMP
            if (num_of_threads<=16)
            {
                omp_set_num_threads(num_of_threads);
            }
            else 
            {
                if (num_of_threads<=20)
                {
                    omp_set_num_threads(16);
                }
                else
                {
                    omp_set_num_threads(num_of_threads-4);
                }
            }
        #endif */

        #ifdef enable_CUDA_CODE
            #ifdef CUDA_with_managed
            cudaMemcpy(dev_CUTOFF_SPIN, &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_CUTOFF_S_SQ, &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_spin, spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_J, J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpy(dev_J_random, J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpy(dev_h_random, h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_N_N_I, N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_no_of_black_white_sites, no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_black_white_checkerboard, black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #else
            cudaMemcpyToSymbol("dev_CUTOFF_S_SQ", &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_spin", spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_J", J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpyToSymbol("dev_J_random", J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpyToSymbol("dev_h_random", h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_N_N_I", N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_no_of_black_white_sites", no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_black_white_checkerboard", black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #endif
        #endif

        // T != 0;
        double delta_phi = del_phi;

        #ifdef UPDATE_ALL_NON_EQ // if (update_all_or_checker == 0)
        {
            // spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        }
        #endif

        int cutoff_local = 0;
        int j_S, j_L;
        double *m_loop = (double*)malloc(dim_S*hysteresis_repeat*sizeof(double));
        
        for (j_S=0; j_S<dim_S*hysteresis_repeat; j_S++)
        {
            m_loop[j_S] = 2;
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = order_start;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = 0;
        }
        
        double h_phi = 0;
        printf("\nftne RFXY h rotating with |h|=%lf at T=%lf.. \n", h_start, T);

        // long int site_i;
        // int black_or_white = BLACK_WHITE;

        repeat_loop = 1;
        int repeat_cond = 1;
        int restore_chkpt = 1;
        int is_complete = 0;
        
        // print rate
        {
            #ifdef CONST_RATE
                printf("\n ====== CONSTANT RATE ====== \n");
            #endif

            #ifdef DIVIDE_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE ====== \n");
            #endif
            
            #ifdef BINARY_DIVISION
                printf("\n ====== BINARY DIVISION to LOCATE AVALANCHE ====== \n");
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                printf("\n ====== SPEEDUP/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
        }
        
        while (repeat_cond)
        {
            pFile_1 = fopen(output_file_name, "a");
            h_phi = 0.0;
            // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
                delta_phi = del_phi;
            // #endif
            // for (h_phi = 0.0; h_phi * order[jj_S] <= 1.0; h_phi = h_phi + order[jj_S] * delta_phi)
            while (h_phi * order[jj_S] <= 1.0)
            {
                #ifdef CHECKPOINT_TIME
                    if (get_time_if_parallel()-start_time > CHECKPOINT_TIME)
                    {
                        if (jj_S == 0)
                        {
                            h[0] = h_start;
                            h[1] = 0.0;
                        }
                        else
                        {
                            h[0] = 0.0;
                            h[1] = h_start;
                        }
                        checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &h_phi, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_loop, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
                        checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
                        checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);

                        fprintf(pFile_1, "===== Checkpointed here. =====\n");
                        fclose(pFile_1);
                        free(m_loop);
                        return -1;
                    }

                    if (restore_chkpt == RESTORE_CHKPT_VALUE)
                    {
                        if (jj_S == 0)
                        {
                            h[0] = h_start;
                            h[1] = 0.0;
                        }
                        else
                        {
                            h[0] = 0.0;
                            h[1] = h_start;
                        }
                        restore_checkpoint(1, TYPE_DOUBLE, dim_S, h, 0);
                        restore_checkpoint(0, TYPE_INT, 1, &is_complete, 0);
                        restore_checkpoint(0, TYPE_DOUBLE, 1, &h_phi, 0);
                        restore_checkpoint(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
                        restore_checkpoint(0, TYPE_INT, 1, &repeat_loop, 0);
                        restore_checkpoint(0, TYPE_INT, 1, &repeat_cond, 0);
                        restore_checkpoint(0, TYPE_LONGINT, 1, &h_counter, 0);
                        restore_checkpoint(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
                        restore_checkpoint(0, TYPE_DOUBLE, dim_S, m, 0);
                        restore_checkpoint(0, TYPE_VOID, dim_S, h, 1);
                        // ensemble_m();

                        restore_chkpt = !restore_chkpt;
                        if (is_complete == 1)
                        {
                            printf("\n---- Already Completed ----\n");

                            fclose(pFile_1);
                            free(m_loop);
                            return 1;
                        }
                    }
                    
                #endif

                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*h_phi);
                    h[1] = h_start * sin(2*pie*h_phi);
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*h_phi);
                    h[1] = h_start * cos(2*pie*h_phi);
                }
                
                // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE)
                //     backing_up_spin();
                // #endif

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                
                #ifdef CHECK_AVALANCHE
                    #ifndef CONST_RATE
                    if (h_phi == 0.0)
                    {
                        continue_avalanche();
                        Monte_Carlo_Sweep(1, /* MC_algo_type */0, MC_update_type);
                        continue_avalanche();
                    }
                    else
                    {
                        if (delta_phi <= del_phi_cutoff)
                        {
                            continue_avalanche();
                        }
                        else
                        {
                            check_avalanche();
                        }
                    }
                    #else
                    continue_avalanche();
                    #endif
                #else
                    cutoff_local = 0;
                    do
                    {
                        // double cutoff_local_last = cutoff_local;
                        cutoff_local = find_change();
                    }
                    while (cutoff_local > 0); // 10^-14
                #endif
                
                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                    
                #ifdef CONST_RATE
                    const_delta_phi(h_phi, delta_phi, jj_S, h_start);
                #endif

                #ifdef DIVIDE_BY_SLOPE
                    slope_subdivide_phi(h_phi, delta_phi, jj_S, h_start);
                #endif
                
                #ifdef BINARY_DIVISION
                    binary_subdivide_phi(h_phi, delta_phi, jj_S, h_start);
                #endif

                #ifdef DYNAMIC_BINARY_DIVISION
                    dynamic_binary_subdivide_phi(&h_phi, &delta_phi, jj_S, h_start);
                #endif
                
                #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                    dynamic_binary_slope_divide_phi(&h_phi, &delta_phi, jj_S, h_start);
                #endif


                // #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
                    if (h_phi * order[jj_S] + delta_phi >= 1.0 && h_phi * order[jj_S] < 1.0)
                    {
                        delta_phi = 1.0 - h_phi * order[jj_S];
                        h_phi = 1.0;
                    }
                    else
                    {
                        h_phi = h_phi + order[jj_S] * delta_phi;
                    }
                // #else
                //     h_phi = h_phi + order[jj_S] * delta_phi;
                // #endif
            }
            printf("\n=========================");
            printf("\n  phi > 1.0 (%.15e)  ", h_phi);
            printf("\n=========================\n");
            if (jj_S == 0)
            {
                h[0] = h_start;
                h[1] = 0.0;
            }
            else
            {
                h[0] = 0.0;
                h[1] = h_start;
            }

            int i;
            for(i=0; i<repeat_loop-1; i++)
            {
                repeat_cond = 0;
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    if (fabs(m_loop[j_S+i*dim_S] - m[j_S]) > CUTOFF_SPIN )
                    {
                        repeat_cond = 1;
                    }
                }
                if (repeat_cond == 0)
                {
                    break;
                }
            }
            for(j_S=0; j_S<dim_S; j_S++)
            {
                m_loop[j_S + (repeat_loop-1)*dim_S] = m[j_S];
            }
            if (repeat_cond == 0)
            {
                fprintf(pFile_1, "loop %d - loop %d\n", repeat_loop, i+1);
                printf("\nloop %d - loop %d\n", repeat_loop, i+1);
            }
            else
            {
                fprintf(pFile_1, "loop %d\n", repeat_loop);
                printf("\nloop %d\n", repeat_loop);
            }
            fclose(pFile_1);
            char append_string[128];
            char *pos = append_string;
            pos += sprintf(pos, "_loop_%d", repeat_loop);
            save_spin_config(append_string, "a", 1);
            repeat_loop++;
            if (repeat_loop-1 == hysteresis_repeat)
            {
                break;
            }
            // h_counter = 0;
        }
        is_complete = 1;
        h_phi = 0.0;
        // if (update_all_or_checker == 0)
        // {
        //     free(spin_temp);
        // }
        #ifdef CHECKPOINT_TIME
            if (jj_S == 0)
            {
                h[0] = h_start;
                h[1] = 0.0;
            }
            else
            {
                h[0] = 0.0;
                h[1] = h_start;
            }
            checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
            checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &h_phi, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
            checkpoint_backup(0, TYPE_INT, 1, &repeat_loop, 0);
            checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
            checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
            checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);
        #endif
        free(m_loop);
        return 0;
    }

    int finite_temp_RFXY_hysteresis_rotate_checkerboard(int jj_S, double order_start, double h_start, char output_file_name[])
    {
        restore_checkpoint(/* startif */-1, TYPE_DOUBLE, dim_S, h, 0); // reset static variable
        start_time = get_time_if_parallel();
        // CUTOFF_S_SQ = 4.0*pie*pie*del_phi*del_phi;

        // del_phi_cutoff = del_phi/no_of_sites;
        
        CUTOFF_M = 1.0/no_of_sites;
        CUTOFF_S_SQ = 1.0;
        #ifdef _OPENMP
        #ifdef BUNDLE
            omp_set_num_threads(BUNDLE);
        #else
            omp_set_num_threads(num_of_threads);
        #endif
        #endif
        /* #ifdef _OPENMP
            if (num_of_threads<=16)
            {
                omp_set_num_threads(num_of_threads);
            }
            else 
            {
                if (num_of_threads<=20)
                {
                    omp_set_num_threads(16);
                }
                else
                {
                    omp_set_num_threads(num_of_threads-4);
                }
            }
        #endif */

        #ifdef enable_CUDA_CODE
            #ifdef CUDA_with_managed
            cudaMemcpy(dev_CUTOFF_SPIN, &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_CUTOFF_S_SQ, &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_spin, spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_J, J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpy(dev_J_random, J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpy(dev_h_random, h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpy(dev_N_N_I, N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_no_of_black_white_sites, no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_black_white_checkerboard, black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #else
            cudaMemcpyToSymbol("dev_CUTOFF_S_SQ", &CUTOFF_S_SQ, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_spin", spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_J", J, dim_L*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_BOND
            cudaMemcpyToSymbol("dev_J_random", J_random, 2*dim_L*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
            #ifdef RANDOM_FIELD
            cudaMemcpyToSymbol("dev_h_random", h_random, dim_S*no_of_sites*sizeof(double), cudaMemcpyHostToDevice);
            #endif
            cudaMemcpyToSymbol("dev_N_N_I", N_N_I, 2*dim_L*no_of_sites*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_no_of_black_white_sites", no_of_black_white_sites, 2*sizeof(long int), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol("dev_black_white_checkerboard", black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int), cudaMemcpyHostToDevice);
            
            #endif
        #endif

        // T != 0;
        double delta_phi = del_phi;

        #ifdef UPDATE_ALL_NON_EQ // if (update_all_or_checker == 0)
        {
            // spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        }
        #endif

        int cutoff_local = 0;
        int j_S, j_L;
        double *m_loop = (double*)malloc(dim_S*hysteresis_repeat*sizeof(double));
        
        for (j_S=0; j_S<dim_S*hysteresis_repeat; j_S++)
        {
            m_loop[j_S] = 2;
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = order_start;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = 0;
        }

        double h_phi = 0;
        printf("\nftne RFXY h rotating with |h|=%lf at T=%lf.. \n", h_start, T);

        // long int site_i;
        // int black_or_white = BLACK_WHITE;

        repeat_loop = 1;
        int repeat_cond = 1;
        int restore_chkpt = 1;
        int resume_chkpt = 0;
        int is_complete = 0;
        
        int Spin_Order = 0;

        // print rate
        {
            #ifdef CONST_RATE
                printf("\n ====== CONSTANT RATE ====== \n");
            #endif

            #ifdef DIVIDE_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE ====== \n");
            #endif
            
            #ifdef BINARY_DIVISION
                printf("\n ====== BINARY DIVISION to LOCATE AVALANCHE ====== \n");
            #endif

            #ifdef DYNAMIC_BINARY_DIVISION
                printf("\n ====== SPEEDUP/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
            
            #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                printf("\n ====== ADJUSTING to MAGNETIZATION CHANGE/SLOWDOWN by FACTOR of TWO ====== \n");
            #endif
        }

        while (repeat_cond)
        {
            pFile_1 = fopen(output_file_name, "a");
            h_phi = 0.0;
            delta_phi = del_phi;

            #if defined (RESTORE_CHKPT_VALUE) && defined (CHECKPOINT_TIME)
                if (restore_chkpt == RESTORE_CHKPT_VALUE)
                {
                    if (jj_S == 0)
                    {
                        h[0] = h_start;
                        h[1] = 0.0;
                    }
                    else
                    {
                        h[0] = 0.0;
                        h[1] = h_start;
                    }

                    resume_chkpt = restore_checkpoint(1, TYPE_DOUBLE, dim_S, h, 0);
                    restore_checkpoint(0, TYPE_INT, 1, &is_complete, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, 1, &h_phi, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
                    restore_checkpoint(0, TYPE_INT, 1, &repeat_loop, 0);
                    restore_checkpoint(0, TYPE_INT, 1, &repeat_cond, 0);
                    restore_checkpoint(0, TYPE_LONGINT, 1, &h_counter, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
                    restore_checkpoint(0, TYPE_DOUBLE, dim_S, m, 0);
                    restore_checkpoint(0, TYPE_VOID, dim_S, h, 1);
                    // ensemble_m();

                    restore_chkpt = !restore_chkpt;
                    if (is_complete == 1)
                    {
                        printf("\n---- Already Completed ----\n");

                        fclose(pFile_1);
                        free(m_loop);
                        return 1;
                    }
                }
            #endif
            
            printf("\n=========================");
            printf("\n [START] phi(h[%d]) = %.15e  ", jj_S, h_phi);
            printf("\n=========================\n");

            if ( resume_chkpt == 0 )
            {
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*h_phi);
                    h[1] = h_start * sin(2*pie*h_phi);
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*h_phi);
                    h[1] = h_start * cos(2*pie*h_phi);
                }

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                ensemble_m();
                #ifdef EXPLORE_ENERGY_LANDSCAPE
                ensemble_E();
                #endif
                backing_up_spin();
            
                #ifdef CHECK_AVALANCHE
                    continue_avalanche();
                #else
                    continue_avalanche();
                    // cutoff_local = 0;
                    // do
                    // {
                    //     // double cutoff_local_last = cutoff_local;
                    //     cutoff_local = find_change();
                    // }
                    // while (cutoff_local > 0); // 10^-14
                #endif

                #ifdef EXPLORE_ENERGY_LANDSCAPE
                calculate_Energy_change();
                #endif

                #ifdef CHECK_AVALANCHE
                    ensemble_delta_S_squared_max();
                    ensemble_m();
                    backing_up_spin();
                    save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", &h_start, 1);
                #else
                    double delta_m = calculate_magnetization_change();
                    backing_up_spin();
                    save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", &h_start, 1);
                #endif
            }
            else
            {
                backing_up_spin();
                resume_chkpt = 0;
            }


            // while (h_phi * order[jj_S] <= 1.0)
            while (h_phi * order[jj_S] < 1.0)
            {
                #ifdef CHECKPOINT_TIME
                    if (get_time_if_parallel()-start_time > CHECKPOINT_TIME)
                    {
                        if (jj_S == 0)
                        {
                            h[0] = h_start;
                            h[1] = 0.0;
                        }
                        else
                        {
                            h[0] = 0.0;
                            h[1] = h_start;
                        }

                        checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &h_phi, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_loop, 0);
                        checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
                        checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
                        checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
                        checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);

                        fprintf(pFile_1, "===== Checkpointed here. =====\n");
                        fclose(pFile_1);
                        free(m_loop);
                        return -1;
                    }
                #endif


                if (h_phi * order[jj_S] + delta_phi >= 1.0 && h_phi * order[jj_S] < 1.0)
                {
                    delta_phi = 1.0 - h_phi * order[jj_S];
                    h_phi = 1.0;
                }
                else
                {
                    h_phi = h_phi + order[jj_S] * delta_phi;
                }

                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*h_phi);
                    h[1] = h_start * sin(2*pie*h_phi);
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*h_phi);
                    h[1] = h_start * cos(2*pie*h_phi);
                }
                
                
                // #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE)
                //     backing_up_spin();
                // #endif

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                
                #ifdef CHECK_AVALANCHE
                    #ifdef CONST_RATE
                    continue_avalanche();
                    Monte_Carlo_Sweep(1, /* MC_algo_type */0, MC_update_type);
                    continue_avalanche();
                    #else
                    if (delta_phi <= del_phi_cutoff)
                    {
                        continue_avalanche();
                    }
                    else
                    {
                        check_avalanche();
                    }
                    #endif
                #else
                    continue_avalanche();
                    Monte_Carlo_Sweep(1, /* MC_algo_type */0, MC_update_type);
                    continue_avalanche();                    
                    // cutoff_local = 0;
                    // do
                    // {
                    //     // double cutoff_local_last = cutoff_local;
                    //     cutoff_local = find_change();
                    // }
                    // while (cutoff_local > 0); // 10^-14
                #endif
                
                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                    
                #ifdef CONST_RATE
                    const_delta_phi(h_phi, delta_phi, jj_S, h_start);
                #endif

                #ifdef DIVIDE_BY_SLOPE
                    slope_subdivide_phi(h_phi, delta_phi, jj_S, h_start);
                #endif
                
                #ifdef BINARY_DIVISION
                    binary_subdivide_phi(h_phi, delta_phi, jj_S, h_start);
                #endif

                #ifdef DYNAMIC_BINARY_DIVISION
                    dynamic_binary_subdivide_phi(&h_phi, &delta_phi, jj_S, h_start);
                #endif
                
                #ifdef DYNAMIC_BINARY_DIVISION_BY_SLOPE
                    dynamic_binary_slope_divide_phi(&h_phi, &delta_phi, jj_S, h_start);
                #endif
                

                // Start editing here
                // Start editing here
                // Start editing here
                // Start editing here
                // Start editing here
                // Spin_Order = check_spin_direction(jj_S);
            }
            printf("\n=========================");
            printf("\n [END] phi(h[%d]) = %.15e  ", jj_S, h_phi);
            printf("\n=========================\n");
            if (jj_S == 0)
            {
                h[0] = h_start;
                h[1] = 0.0;
            }
            else
            {
                h[0] = 0.0;
                h[1] = h_start;
            }

            int i;
            for(i=0; i<repeat_loop-1; i++)
            {
                repeat_cond = 0;
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    if (fabs(m_loop[j_S+i*dim_S] - m[j_S]) > CUTOFF_SPIN )
                    {
                        repeat_cond = 1;
                    }
                }
                if (repeat_cond == 0)
                {
                    break;
                }
            }
            for(j_S=0; j_S<dim_S; j_S++)
            {
                m_loop[j_S + (repeat_loop-1)*dim_S] = m[j_S];
            }

            if (repeat_cond == 0)
            {
                fprintf(pFile_1, "loop %d - loop %d\n", repeat_loop, i+1);
                printf("\nloop %d - loop %d\n", repeat_loop, i+1);
            }
            else
            {
                fprintf(pFile_1, "loop %d\n", repeat_loop);
                printf("\nloop %d\n", repeat_loop);
            }
            fclose(pFile_1);
            char append_string[128];
            char *pos = append_string;
            pos += sprintf(pos, "_loop_%d", repeat_loop);
            save_spin_config(append_string, "a", 1);
            repeat_loop++;
            h_counter = 0;
            if (repeat_loop-1 == hysteresis_repeat)
            {
                break;
            }
            repeat_cond = 1;
            
        }
        is_complete = 1;
        h_phi = 0.0;
        delta_phi = del_phi;
        
        // if (update_all_or_checker == 0)
        // {
        //     free(spin_temp);
        // }
        #ifdef CHECKPOINT_TIME
            if (jj_S == 0)
            {
                h[0] = h_start;
                h[1] = 0.0;
            }
            else
            {
                h[0] = 0.0;
                h[1] = h_start;
            }
            checkpoint_backup(1, TYPE_DOUBLE, dim_S, h, 0);
            checkpoint_backup(0, TYPE_INT, 1, &is_complete, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &h_phi, 0);
            checkpoint_backup(0, TYPE_DOUBLE, 1, &delta_phi, 0); // new
            checkpoint_backup(0, TYPE_INT, 1, &repeat_loop, 0);
            checkpoint_backup(0, TYPE_INT, 1, &repeat_cond, 0);
            checkpoint_backup(0, TYPE_LONGINT, 1, &h_counter, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S*hysteresis_repeat, m_loop, 0);
            checkpoint_backup(0, TYPE_DOUBLE, dim_S, m, 0);
            checkpoint_backup(0, TYPE_VOID, dim_S, h, 1);
        #endif
        free(m_loop);
        return 0;
    }

//====================  RFXY protocols                ====================//

    int ordered_initialize_and_axis_checkerboard(int jj_S, double order_start, double* h_sweep_abs, int finite_sweep, double zero_or_finite)
    {
        double start_time_func = get_time_if_parallel();
        T = zero_or_finite;
        ax_ro = 0;
        or_ho_ra = 0;
        int cutoff_local = 0;
        int j_S, j_L;
        h_order = 0;
        r_order = 0;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = -order_start;
        initialize_spin_config();
        order[jj_S] = order_start;

        double h_start[dim_S];
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = h_sweep_abs[j_S];
            h_start[j_S] = h_sweep_abs[j_S];
        }

        if (finite_sweep == 0 || finite_sweep == 1)
        {
            if ( fabs(h_start[jj_S]) < h_max )
            {
                h_start[jj_S] = h_max;
            }
            if ( fabs(h_start[jj_S]) < fabs(h_i_max)+0.01 )
            {
                h_start[jj_S] = h_i_max + 0.01;
            }
            if ( fabs(h_start[jj_S]) < fabs(h_i_min-0.01) )
            {
                h_start[jj_S] = h_i_min - 0.01;
            }
            h_start[jj_S] = -order[jj_S]*fabs(h_start[jj_S]);
        }
        else
        {
            h_start[jj_S] = -order[jj_S]*fabs(h_start[jj_S]);
        }
        double h_abs_jj_S = fabs(h_sweep_abs[jj_S]);
        h[jj_S] = h_start[jj_S];

        // start from h[0] or h[1] != 0
        // h[jj_S] = h_start;
        double delta_h = del_h; 
        
        
        ensemble_m();
        ensemble_E();

        // print statements:
        {
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        // set output_variable_name=1 here
        {
            output_prepend = 1;
            output_h = 1;
            output_m = 1;
            #ifdef CHECK_AVALANCHE
            output_delta_S_abs = 1;
            output_delta_S_squared = 1;
            #else
            output_delta_M = 1;
            #endif

            #ifdef EXPLORE_ENERGY_LANDSCAPE
            output_E = 1;
            output_delta_E = 1;
            #endif
            output_append = 1;
        }
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hys_ax_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            /* pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); */
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.6f)", h[jj_S]);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%.3f", order[j_S]);
            // }
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{%.4f}", order[jj_S]*delta_h);
            
            pos += sprintf(pos, "_o_a.dat");
        }

        if( access( output_file_0, F_OK ) != -1 )
        {
            if (RESTORE_CHKPT_VALUE == 0)
            {
                printf("File exists! filename = %s \nExiting..\n", output_file_0);
                return 0; // file exists
            }
            else
            {
                printf("Appending to file = %s \n", output_file_0);

                pFile_1 = fopen(output_file_0, "a");
                #ifdef UPDATE_ALL_NON_EQ
                {
                    printf("\nUpdating all sites simultaneously.. \n");
                    
                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                // else
                #ifdef UPDATE_CHKR_NON_EQ
                {
                    printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                fclose(pFile_1);

                #ifdef CHECK_AVALANCHE
                output_label(output_file_0, "h[%d]\t", "max(|delta{S}|)\t");
                #else
                output_label(output_file_0, "h[%d]\t", "|delta{m[%d]}|\t");
                #endif
            }
        }
        else
        {
            // column labels and parameters
            output_param_file(output_file_0);

            pFile_1 = fopen(output_file_0, "a");
            #ifdef UPDATE_ALL_NON_EQ
            {
                printf("\nUpdating all sites simultaneously.. \n");
                
                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            // else
            #ifdef UPDATE_CHKR_NON_EQ
            {
                printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            fclose(pFile_1);

            #ifdef CHECK_AVALANCHE
            output_label(output_file_0, "h[%d]\t", "max(|delta{S}|)\t");
            #else
            output_label(output_file_0, "h[%d]\t", "|delta{m[%d]}|\t");
            #endif
        }
        int is_chkpt;
        if (zero_or_finite == 0)
        {
            is_chkpt = zero_temp_RFXY_hysteresis_axis_checkerboard(jj_S, order_start, h_start, h_abs_jj_S, finite_sweep, output_file_0);
        }
        else
        {
            is_chkpt = finite_temp_RFXY_hysteresis_axis_checkerboard(jj_S, order_start, h_start, h_abs_jj_S, finite_sweep, output_file_0);
        }

        reset_output_variable_name_0();
        
        double end_time_func = get_time_if_parallel();

        printf("\nTime taken by Sweeping Hysteresis = %lf s : ", end_time_func - start_time_func);
        printf("\n Starting from an ordered spin configuration and sweeping h[%d] with delta_h = %lf ", jj_S, del_h );
        printf("\n------------------------------------------------------\n\n");

        return is_chkpt;
    }
    
    int random_initialize_and_axis_checkerboard(int jj_S, double order_start, double* h_sweep_abs, int finite_sweep, double zero_or_finite)
    {
        double start_time_func = get_time_if_parallel();
        T = zero_or_finite;
        ax_ro = 0;
        or_ho_ra = 2;
        int cutoff_local = 0;
        int j_S, j_L;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = order_start;
        double h_start[dim_S];
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = h_sweep_abs[j_S];
            h_start[j_S] = h_sweep_abs[j_S];
        }
        double h_abs_jj_S = fabs(h_sweep_abs[jj_S]);
        h_start[jj_S] = 0.0;
        h[jj_S] = h_start[jj_S];
        // start from h[0] or h[1] != 0
        // h[jj_S] = h_start;
        double delta_h = del_h;
        h_order = 0;
        r_order = 1;
        initialize_spin_config();
        
        ensemble_m();
        ensemble_E();

        // print statements:
        {
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        // set output_variable_name=1 here
        {
            output_prepend = 1;
            output_h = 1;
            output_m = 1;
            #ifdef CHECK_AVALANCHE
            output_delta_S_abs = 1;
            output_delta_S_squared = 1;
            #else
            output_delta_M = 1;
            #endif

            #ifdef EXPLORE_ENERGY_LANDSCAPE
            output_E = 1;
            output_delta_E = 1;
            #endif
            output_append = 1;
        }
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hys_ax_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            /* pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); */
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.6f)", h[j_S]);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%.3f", order[j_S]);
            // }
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{%.4f}", order[jj_S]*delta_h);
            
            pos += sprintf(pos, "_r_a.dat");
        }

        if( access( output_file_0, F_OK ) != -1 )
        {
            if (RESTORE_CHKPT_VALUE == 0)
            {
                printf("File exists! filename = %s \nExiting..\n", output_file_0);
                return 0; // file exists
            }
            else
            {
                printf("Appending to file = %s \n", output_file_0);

                pFile_1 = fopen(output_file_0, "a");
                #ifdef UPDATE_ALL_NON_EQ
                {
                    printf("\nUpdating all sites simultaneously.. \n");
                    
                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                // else
                #ifdef UPDATE_CHKR_NON_EQ
                {
                    printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                fclose(pFile_1);

                #ifdef CHECK_AVALANCHE
                output_label(output_file_0, "h[%d]\t", "max(|delta{S}|)\t");
                #else
                output_label(output_file_0, "h[%d]\t", "|delta{m[%d]}|\t");
                #endif
            }
        }
        else
        {
            // column labels and parameters
            output_param_file(output_file_0);

            pFile_1 = fopen(output_file_0, "a");
            #ifdef UPDATE_ALL_NON_EQ
            {
                printf("\nUpdating all sites simultaneously.. \n");
                
                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            // else
            #ifdef UPDATE_CHKR_NON_EQ
            {
                printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            fclose(pFile_1);

            #ifdef CHECK_AVALANCHE
            output_label(output_file_0, "h[%d]\t", "max(|delta{S}|)\t");
            #else
            output_label(output_file_0, "h[%d]\t", "|delta{m[%d]}|\t");
            #endif
        }
        int is_chkpt;
        if (zero_or_finite == 0)
        {
            is_chkpt = zero_temp_RFXY_hysteresis_axis_checkerboard(jj_S, order_start, h_start, h_abs_jj_S, finite_sweep, output_file_0);
        }
        else
        {
            is_chkpt = finite_temp_RFXY_hysteresis_axis_checkerboard(jj_S, order_start, h_start, h_abs_jj_S, finite_sweep, output_file_0);
        }

        reset_output_variable_name_0();
        
        double end_time_func = get_time_if_parallel();

        printf("\nTime taken by Sweeping Hysteresis = %lf s : ", end_time_func - start_time_func);
        printf("\n Starting from a random spin configuration and sweeping h[%d] with delta_h = %lf ", jj_S, del_h );
        printf("\n------------------------------------------------------\n\n");

        return is_chkpt;

    }

    int load_spin_and_axis_checkerboard(char spin_file[], int jj_S, double order_start, double* h_sweep_abs, int finite_sweep, double zero_or_finite)
    {
        double start_time_func = get_time_if_parallel();
        T = zero_or_finite;
        ax_ro = 0;
        or_ho_ra = 2;
        int cutoff_local = 0;
        int j_S, j_L;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = order_start;
        double h_start[dim_S];
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = h_sweep_abs[j_S];
            h_start[j_S] = h_sweep_abs[j_S];
        }
        double h_abs_jj_S = fabs(h_sweep_abs[jj_S]);
        h_start[jj_S] = 0.0;
        h[jj_S] = h_start[jj_S];
        // start from h[0] or h[1] != 0
        // h[jj_S] = h_start;
        double delta_h = del_h;
        h_order = 0;
        r_order = 1;
        initialize_spin_config();
        load_spin_config(spin_file);
        
        ensemble_m();
        ensemble_E();

        // print statements:
        {
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        // set output_variable_name=1 here
        {
            output_prepend = 1;
            output_h = 1;
            output_m = 1;
            #ifdef CHECK_AVALANCHE
            output_delta_S_abs = 1;
            output_delta_S_squared = 1;
            #else
            output_delta_M = 1;
            #endif

            #ifdef EXPLORE_ENERGY_LANDSCAPE
            output_E = 1;
            output_delta_E = 1;
            #endif
            output_append = 1;
        }
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hys_ax_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            /* pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); */
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.6f)", h[j_S]);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%.3f", order[j_S]);
            // }
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{%.4f}", order[jj_S]*delta_h);
            
            pos += sprintf(pos, "_l_a.dat");
        }

        if( access( output_file_0, F_OK ) != -1 )
        {
            if (RESTORE_CHKPT_VALUE == 0)
            {
                printf("File exists! filename = %s \nExiting..\n", output_file_0);
                return 0; // file exists
            }
            else
            {
                printf("Appending to file = %s \n", output_file_0);

                pFile_1 = fopen(output_file_0, "a");
                #ifdef UPDATE_ALL_NON_EQ
                {
                    printf("\nUpdating all sites simultaneously.. \n");
                    
                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                // else
                #ifdef UPDATE_CHKR_NON_EQ
                {
                    printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                fclose(pFile_1);

                #ifdef CHECK_AVALANCHE
                output_label(output_file_0, "h[%d]\t", "max(|delta{S}|)\t");
                #else
                output_label(output_file_0, "h[%d]\t", "|delta{m[%d]}|\t");
                #endif
            }
        }
        else
        {
            // column labels and parameters
            output_param_file(output_file_0);

            pFile_1 = fopen(output_file_0, "a");
            #ifdef UPDATE_ALL_NON_EQ
            {
                printf("\nUpdating all sites simultaneously.. \n");
                
                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            // else
            #ifdef UPDATE_CHKR_NON_EQ
            {
                printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            fclose(pFile_1);

            #ifdef CHECK_AVALANCHE
            output_label(output_file_0, "h[%d]\t", "max(|delta{S}|)\t");
            #else
            output_label(output_file_0, "h[%d]\t", "|delta{m[%d]}|\t");
            #endif
        }
        int is_chkpt;
        if (zero_or_finite == 0)
        {
            is_chkpt = zero_temp_RFXY_hysteresis_axis_checkerboard(jj_S, order_start, h_start, h_abs_jj_S, finite_sweep, output_file_0);
        }
        else
        {
            is_chkpt = finite_temp_RFXY_hysteresis_axis_checkerboard(jj_S, order_start, h_start, h_abs_jj_S, finite_sweep, output_file_0);
        }
        
        reset_output_variable_name_0();
        
        double end_time_func = get_time_if_parallel();

        printf("\nTime taken by Sweeping Hysteresis = %lf s : ", end_time_func - start_time_func);
        printf("\n Starting from a given spin configuration and sweeping h[%d] with delta_h = %lf ", jj_S, del_h );
        printf("\n------------------------------------------------------\n\n");

        return is_chkpt;

    }

    int ordered_initialize_and_rotate_checkerboard(int jj_S, double order_start, double h_rotate_dir, double zero_or_finite)
    {
        double start_time_func = get_time_if_parallel();
        T = zero_or_finite;
        ax_ro = 1;
        or_ho_ra = 0;
        int cutoff_local = 0;
        int j_S, j_L;
        
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = custom_double_sign(h_rotate_dir);
        h_order = 0;
        r_order = 0;
        initialize_spin_config();
        order[jj_S] = order_start;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = 0;
        }
        // start from h[0] or h[1] != 0
        double h_start = h_rotate_dir; // order[jj_S]*(h_rotate_dir);
        h[jj_S] = h_start;
        double delta_phi = del_phi; // double delta_phi = 1.0; // for h_start=0


        ensemble_m();
        ensemble_E();
        
        // print statements:
        {
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        // set output_variable_name=1 here
        {
            output_prepend = 1;
            output_h = 1;
            output_m = 1;
            #ifdef CHECK_AVALANCHE
            output_delta_S_abs = 1;
            output_delta_S_squared = 1;
            #else
            output_delta_M = 1;
            #endif

            #ifdef EXPLORE_ENERGY_LANDSCAPE
            output_E = 1;
            output_delta_E = 1;
            #endif
            output_append = 1;
        }
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hys_rot_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            /* pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); */
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.6f)", h_start);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%.3f", order[j_S]);
            // }
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{%.4f}", order[jj_S]*delta_phi);
            
            pos += sprintf(pos, "_o_r.dat");
        }
        if( access( output_file_0, F_OK ) != -1 )
        {
            if (RESTORE_CHKPT_VALUE == 0)
            {
                printf("File exists! filename = %s \nExiting..\n", output_file_0);
                return 0; // file exists
            }
            else
            {
                printf("Appending to file = %s \n", output_file_0);

                pFile_1 = fopen(output_file_0, "a");
                #ifdef UPDATE_ALL_NON_EQ
                {
                    printf("\nUpdating all sites simultaneously.. \n");
                    
                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                // else
                #ifdef UPDATE_CHKR_NON_EQ
                {
                    printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                fclose(pFile_1);

                #ifdef CHECK_AVALANCHE
                output_label(output_file_0, "phi(+phi_0)\t", "max(|delta{S}|)\t");
                #else
                output_label(output_file_0, "phi(+phi_0)\t", "|delta{m[%d]}|\t");
                #endif
            }
        }
        else
        {
            // column labels and parameters
            output_param_file(output_file_0);

            pFile_1 = fopen(output_file_0, "a");
            #ifdef UPDATE_ALL_NON_EQ
            {
                printf("\nUpdating all sites simultaneously.. \n");
                
                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            // else
            #ifdef UPDATE_CHKR_NON_EQ
            {
                printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            fclose(pFile_1);

            #ifdef CHECK_AVALANCHE
            output_label(output_file_0, "phi(+phi_0)\t", "max(|delta{S}|)\t");
            #else
            output_label(output_file_0, "phi(+phi_0)\t", "|delta{m[%d]}|\t");
            #endif
        }
        
        // T = 0;
        int is_chkpt;
        if (zero_or_finite == 0)
        {
            #ifdef OLD_FUNCTION
            is_chkpt = zero_temp_RFXY_hysteresis_rotate_checkerboard_old(jj_S, order_start, h_start, output_file_0);
            #else
            is_chkpt = zero_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start, h_start, output_file_0);
            #endif
        }
        else
        {
            #ifdef OLD_FUNCTION
            is_chkpt = finite_temp_RFXY_hysteresis_rotate_checkerboard_old(jj_S, order_start, h_start, output_file_0);
            #else
            is_chkpt = finite_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start, h_start, output_file_0);
            #endif
        }
        
        reset_output_variable_name_0();
        
        double end_time_func = get_time_if_parallel();

        printf("\nTime taken by Rotating Hysteresis = %lf s : ", end_time_func - start_time_func);
        printf("\n Starting from an ordered spin configuration along initial h[%d] = %lf and rotating with delta_phi = %lf ", jj_S, h_start, order_start*del_phi );
        printf("\n------------------------------------------------------\n\n");

        return is_chkpt;
    }

    int field_cool_and_rotate_checkerboard(int jj_S, double order_start, double h_rotate_dir, double zero_or_finite)
    {
        double start_time_func_1 = get_time_if_parallel();
        int ax_ro = 1;
        int or_ho_ra = 2;
        T = Temp_max;
        // random initialization
        int j_S, j_L, j_SS;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = order_start;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = 0;
        }
        // start from h[0] or h[1] != 0
        double h_start = h_rotate_dir;// order[jj_S]*(h_rotate_dir);
        h[jj_S] = h_start;
        double delta_phi = del_phi; // double delta_phi = 1.0; // for h_start=0
        h_order = 0;
        r_order = 1;
        load_spin_config("");
                
        ensemble_m();
        ensemble_E();
        
        // print statements:
        {
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        // set output_variable_name=1 here
        {
            output_T = 1;
            output_m_avg = 1;
            // output_m_abs_avg = 1;
            // output_m_2_avg = 1;
            // output_m_4_avg = 1;
            // output_m_2_vec_avg = 1;
            // output_m_4_vec_avg = 1;
            // output_m_ab_avg = 1;
            // output_X = 1;
            // output_X_ab = 1;
            // output_B = 1;
            // output_B_a = 1;
            // output_E_avg = 1;
            // output_E_2_avg = 1;
            // output_Cv = 1;
            // output_Y_1_avg = 1;
            // output_Y_2_avg = 1;
            // output_Y_x_mu = 1;
        }
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hys_rot_fcool_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_{%.3f,%.3f}", Temp_max, Temp_min);
            /* for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); */
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.6f)", h_start);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%.3f", order[j_S]);
            // }
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{%.4f}", order[jj_S]*delta_phi);
        }
        char output_file_1[256];
        strcpy(output_file_1, output_file_0);
        strcat(output_file_1, "_c.dat");
        
        // cooling_protocol T_MAX - T_MIN=0
        // column labels and parameters
        output_param_file(output_file_1);
        output_label(output_file_1, "", "");
        
        cooling_heating(output_file_1, 0);
        save_spin_config("", "a", 1);
        
        reset_output_variable_name_0();
        
        double end_time_func_1 = get_time_if_parallel();

        printf("\nTime taken by Field Cooling = %lf s : ", end_time_func_1 - start_time_func_1);
        printf("\n Starting from a random spin configuration at T = %lf and cooling step delta_T = %lf with h[%d] = %lf ", Temp_max, delta_T, jj_S, h_start );
        printf("\n------------------------------------------------------\n\n");

        // rotate field
        // if ( Temp_min > 0.005 )
        if ( Temp_min > zero_or_finite )
        {
            // printf("\nRotating Hysteresis not started since Temp_min > 0.005 ");
            printf("\nRotating Hysteresis not started since Temp_min > %lf ", zero_or_finite);
            printf("\n------------------------------------------------------\n\n");
            
            return 0;
        }
        else
        {
            double start_time_func_2 = get_time_if_parallel();
            // set output_variable_name=1 here
            {
                output_prepend = 1;
                output_h = 1;
                output_m = 1;
                #ifdef CHECK_AVALANCHE
                output_delta_S_abs = 1;
                output_delta_S_squared = 1;
                #else
                output_delta_M = 1;
                #endif

                #ifdef EXPLORE_ENERGY_LANDSCAPE
                output_E = 1;
                output_delta_E = 1;
                #endif
                output_append = 1;
            }
            
            char output_file_2[256];
            strcpy(output_file_2, output_file_0);
            strcat(output_file_2, "_c_r.dat");
            if( access( output_file_2, F_OK ) != -1 )
            {
                if (RESTORE_CHKPT_VALUE == 0)
                {
                    printf("File exists! filename = %s \nExiting..\n", output_file_2);
                    return 0; // file exists
                }
                else
                {
                    printf("Appending to file = %s \n", output_file_2);

                    pFile_1 = fopen(output_file_2, "a");
                    #ifdef UPDATE_ALL_NON_EQ
                    {
                        printf("\nUpdating all sites simultaneously.. \n");
                        
                        fprintf(pFile_1, "==================================================================================\n");
                        fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                        fprintf(pFile_1, "==================================================================================\n");
                    }
                    #endif
                    // else
                    #ifdef UPDATE_CHKR_NON_EQ
                    {
                        printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                        fprintf(pFile_1, "==================================================================================\n");
                        fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                        fprintf(pFile_1, "==================================================================================\n");
                    }
                    #endif
                    fclose(pFile_1);

                    #ifdef CHECK_AVALANCHE
                    output_label(output_file_2, "phi(+phi_0)\t", "max(|delta{S}|)\t");
                    #else
                    output_label(output_file_2, "phi(+phi_0)\t", "|delta{m[%d]}|\t");
                    #endif
                }
            }
            else
            {
                // column labels and parameters
                output_param_file(output_file_2);

                pFile_1 = fopen(output_file_2, "a");
                #ifdef UPDATE_ALL_NON_EQ
                {
                    printf("\nUpdating all sites simultaneously.. \n");
                    
                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                // else
                #ifdef UPDATE_CHKR_NON_EQ
                {
                    printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                fclose(pFile_1);

                #ifdef CHECK_AVALANCHE
                output_label(output_file_2, "phi(+phi_0)\t", "max(|delta{S}|)\t");
                #else
                output_label(output_file_2, "phi(+phi_0)\t", "|delta{m[%d]}|\t");
                #endif
            }
            int is_chkpt;
            if (zero_or_finite == 0)
            {
                #ifdef OLD_FUNCTION
                is_chkpt = zero_temp_RFXY_hysteresis_rotate_checkerboard_old(jj_S, order_start, h_start, output_file_2);
                #else
                is_chkpt = zero_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start, h_start, output_file_2);
                #endif
            }
            else
            {
                #ifdef OLD_FUNCTION
                is_chkpt = finite_temp_RFXY_hysteresis_rotate_checkerboard_old(jj_S, order_start, h_start, output_file_2);
                #else
                is_chkpt = finite_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start, h_start, output_file_2);
                #endif
            }
            
            reset_output_variable_name_0();
            
            double end_time_func_2 = get_time_if_parallel();
            printf("\nTime taken by Rotating Hysteresis = %lf s : ", end_time_func_2 - start_time_func_2);
            printf("\n Starting from a field cooled spin configuration with initial h[%d] = %lf and rotating with delta_phi = %lf ", jj_S, h_start, order_start*del_phi );
            printf("\n------------------------------------------------------\n\n");

            return is_chkpt;
        }

        
        return 0;
    }

    int random_initialize_and_rotate_checkerboard(int jj_S, double order_start, double h_rotate_dir, double zero_or_finite)
    {
        double start_time_func = get_time_if_parallel();
        T = zero_or_finite;
        int ax_ro = 1;
        int or_ho_ra = 2;
        int j_S, j_L;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = custom_double_sign(h_rotate_dir);
        h_order = 0;
        r_order = 1;
        initialize_spin_config();
        order[jj_S] = order_start;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = 0;
        }
        double h_start = h_rotate_dir; // order[jj_S]*(h_rotate_dir);
        h[jj_S] = h_start;
        double delta_phi = del_phi;

        
        ensemble_m();
        ensemble_E();
        
        // print statements:
        {
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        // set output_variable_name=1 here
        {
            output_prepend = 1;
            output_h = 1;
            output_m = 1;
            #ifdef CHECK_AVALANCHE
            output_delta_S_abs = 1;
            output_delta_S_squared = 1;
            #else
            output_delta_M = 1;
            #endif

            #ifdef EXPLORE_ENERGY_LANDSCAPE
            output_E = 1;
            output_delta_E = 1;
            #endif
            output_append = 1;
        }
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hys_rot_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            /* pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); */
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.6f)", h_start);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%.3f", order[j_S]);
            // }
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{%.4f}", order[jj_S]*delta_phi);
            pos += sprintf(pos, "_r_r.dat");
        }
        if( access( output_file_0, F_OK ) != -1 )
        {
            if (RESTORE_CHKPT_VALUE == 0)
            {
                printf("File exists! filename = %s \nExiting..\n", output_file_0);
                return 0; // file exists
            }
            else
            {
                printf("Appending to file = %s \n", output_file_0);

                pFile_1 = fopen(output_file_0, "a");
                #ifdef UPDATE_ALL_NON_EQ
                {
                    printf("\nUpdating all sites simultaneously.. \n");
                    
                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                // else
                #ifdef UPDATE_CHKR_NON_EQ
                {
                    printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                fclose(pFile_1);

                #ifdef CHECK_AVALANCHE
                output_label(output_file_0, "phi(+phi_0)\t", "max(|delta{S}|)\t");
                #else
                output_label(output_file_0, "phi(+phi_0)\t", "|delta{m[%d]}|\t");
                #endif
            }
        }
        else
        {
            // column labels and parameters
            output_param_file(output_file_0);

            pFile_1 = fopen(output_file_0, "a");
            #ifdef UPDATE_ALL_NON_EQ
            {
                printf("\nUpdating all sites simultaneously.. \n");
                
                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            // else
            #ifdef UPDATE_CHKR_NON_EQ
            {
                printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            fclose(pFile_1);
            
            #ifdef CHECK_AVALANCHE
            output_label(output_file_0, "phi(+phi_0)\t", "max(|delta{S}|)\t");
            #else
            output_label(output_file_0, "phi(+phi_0)\t", "|delta{m[%d]}|\t");
            #endif
        }
        int is_chkpt;
        if (zero_or_finite == 0)
        {
            #ifdef OLD_FUNCTION
            is_chkpt = zero_temp_RFXY_hysteresis_rotate_checkerboard_old(jj_S, order_start, h_start, output_file_0);
            #else
            is_chkpt = zero_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start, h_start, output_file_0);
            #endif
        }
        else
        {
            #ifdef OLD_FUNCTION
            is_chkpt = finite_temp_RFXY_hysteresis_rotate_checkerboard_old(jj_S, order_start, h_start, output_file_0);
            #else
            is_chkpt = finite_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start, h_start, output_file_0);
            #endif
        }
        
        reset_output_variable_name_0();
        
        double end_time_func = get_time_if_parallel();

        printf("\nTime taken by Rotating Hysteresis = %lf s : ", end_time_func - start_time_func);
        printf("\n Starting from a random spin configuration with initial h[%d] = %lf and rotating with delta_phi = %lf ", jj_S, h_start, order_start*del_phi );
        printf("\n------------------------------------------------------\n\n");

        return is_chkpt;
    }

    int load_spin_and_rotate_checkerboard(char spin_file[], int jj_S, double order_start, double h_rotate_dir, double zero_or_finite)
    {
        double start_time_func = get_time_if_parallel();
        T = zero_or_finite;
        int ax_ro = 1;
        int or_ho_ra = 2;
        int j_S, j_L;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = custom_double_sign(h_rotate_dir);
        h_order = 0;
        r_order = 1;
        // initialize_spin_config();
        load_spin_config(spin_file);
        order[jj_S] = order_start;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = 0;
        }
        double h_start = h_rotate_dir; // order[jj_S]*(h_rotate_dir);
        h[jj_S] = h_start;
        double delta_phi = del_phi;

        
        ensemble_m();
        ensemble_E();
        
        // print statements:
        {
            printf("\nm = %lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        
        // set output_variable_name=1 here
        {
            output_prepend = 1;
            output_h = 1;
            output_m = 1;
            #ifdef CHECK_AVALANCHE
            output_delta_S_abs = 1;
            output_delta_S_squared = 1;
            #else
            output_delta_M = 1;
            #endif

            #ifdef EXPLORE_ENERGY_LANDSCAPE
            output_E = 1;
            output_delta_E = 1;
            #endif
            output_append = 1;
        }

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hys_rot_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                if (BC[j_L]==0) { pos += sprintf(pos, "%do", lattice_size[j_L]); }
                else if (BC[j_L]==1) { pos += sprintf(pos, "%dp", lattice_size[j_L]); }
                else { pos += sprintf(pos, "%d(%.3f)", lattice_size[j_L], BC[j_L]); }
            }
            pos += sprintf(pos, "_%.3f", T);
            /* pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", J[j_L]);
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_J[j_L]);
            }
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            // }
            // pos += sprintf(pos, "}"); */
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.6f)", h_start);
                }
                else
                {
                    pos += sprintf(pos, "%.3f", h[j_S]);
                }
            }
            pos += sprintf(pos, "}");    
            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            // pos += sprintf(pos, "}");
            // pos += sprintf(pos, "_{");
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%.3f", order[j_S]);
            // }
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{%.4f}", order[jj_S]*delta_phi);
            pos += sprintf(pos, "_l_r.dat");
        }
        
        if( access( output_file_0, F_OK ) != -1 )
        {
            if (RESTORE_CHKPT_VALUE == 0)
            {
                printf("File exists! filename = %s \nExiting..\n", output_file_0);
                return 0; // file exists
            }
            else
            {
                printf("Appending to file = %s \n", output_file_0);
                
                pFile_1 = fopen(output_file_0, "a");
                #ifdef UPDATE_ALL_NON_EQ
                {
                    printf("\nUpdating all sites simultaneously.. \n");
                    
                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                // else
                #ifdef UPDATE_CHKR_NON_EQ
                {
                    printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                    fprintf(pFile_1, "==================================================================================\n");
                    fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                    fprintf(pFile_1, "==================================================================================\n");
                }
                #endif
                fclose(pFile_1);

                #ifdef CHECK_AVALANCHE
                output_label(output_file_0, "phi(+phi_0)\t", "max(|delta{S}|)\t");
                #else
                output_label(output_file_0, "phi(+phi_0)\t", "|delta{m[%d]}|\t");
                #endif
            }
        }
        else
        {
            // column labels and parameters
            output_param_file(output_file_0);

            pFile_1 = fopen(output_file_0, "a");
            #ifdef UPDATE_ALL_NON_EQ
            {
                printf("\nUpdating all sites simultaneously.. \n");
                
                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all sites simultaneously.. \n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            // else
            #ifdef UPDATE_CHKR_NON_EQ
            {
                printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

                fprintf(pFile_1, "==================================================================================\n");
                fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
                fprintf(pFile_1, "==================================================================================\n");
            }
            #endif
            fclose(pFile_1);

            #ifdef CHECK_AVALANCHE
            output_label(output_file_0, "phi(+phi_0)\t", "max(|delta{S}|)\t");
            #else
            output_label(output_file_0, "phi(+phi_0)\t", "|delta{m[%d]}|\t");
            #endif
        }
        int is_chkpt;
        if (zero_or_finite == 0)
        {
            #ifdef OLD_FUNCTION
            is_chkpt = zero_temp_RFXY_hysteresis_rotate_checkerboard_old(jj_S, order_start, h_start, output_file_0);
            #else
            is_chkpt = zero_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start, h_start, output_file_0);
            #endif
        }
        else
        {
            #ifdef OLD_FUNCTION
            is_chkpt = finite_temp_RFXY_hysteresis_rotate_checkerboard_old(jj_S, order_start, h_start, output_file_0);
            #else
            is_chkpt = finite_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start, h_start, output_file_0);
            #endif
        }

        reset_output_variable_name_0();
        
        double end_time_func = get_time_if_parallel();

        printf("\nTime taken by Rotating Hysteresis = %lf s : ", end_time_func - start_time_func);
        printf("\n Starting from a given spin configuration with initial h[%d] = %lf and rotating with delta_phi = %lf ", jj_S, h_start, order_start*del_phi );
        printf("\n------------------------------------------------------\n\n");

        return is_chkpt;
    }

//====================  RFXYZ ZTNE                    ====================//
    
    int checking_O3_spin_with_O2_RF()
    {
        T = 0.0;
        int j_S, j_L;
        int cutoff_local = 0;
        long int counter = 0;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0.0;
        }
        order[0] = 1.0 ;
        r_order = 0;
        h_order = 1;
        initialize_spin_config();
        // set output_variable_name=1 here
        {
            output_m = 1;
            output_append = 1;
        }
        char output_file_1[] = "magnetization_O3_O2_h.dat";

        // char append_string[128];
        // char *pos = append_string;
        // pos += sprintf(pos, "_count_%ld", counter);
        // save_spin_config(append_string, "w", 1);
            
        output_label(output_file_1, "", "cutoff\t");
        ensemble_all();
        output_data(output_file_1, "", "-1\t");

        do
        {
            counter++;
            
            cutoff_local = find_change();
            
            // char append_string[128];
            // char *pos = append_string;
            // pos += sprintf(pos, "_count_%ld", counter);
            // save_spin_config(append_string, "w", 1);

            ensemble_all();
            char str_app[128];
            char *pos_app = str_app;
            pos_app += sprintf(pos_app, "%d\t", cutoff_local);
            output_data(output_file_1, "", str_app);

            printf("\r%d", cutoff_local);
            fflush(stdout);
        }
        while (cutoff_local > 0); // 10^-14
        printf("\n");
        char append_string[128];
        char *pos = append_string;
        pos += sprintf(pos, "_count_%ld_h", counter);
        save_spin_config(append_string, "w", 1);
        
        reset_output_variable_name_0();

        T=0.1;
        
        evolution_at_T(3,100);

        // reset_output_variable_name_0();
        
        return 0;
    }

//========================================================================//

//========================================================================//
//====================  Main                          ====================//
    
    int for_cuda_parallelization()
    {
        printf("\nCUDA Active.\n");
        long int i, j;
        #ifdef _OPENMP
        printf("OpenMP Active.\n");
        num_of_threads = omp_get_max_threads();
        num_of_procs = omp_get_num_procs();
        #else
        num_of_threads = 1;
        num_of_procs = 1;
        #endif
        // random_seed = (unsigned int*)malloc(cache_size*num_of_threads*sizeof(unsigned int));
        init_mt19937_parallel(num_of_threads);
        // init_genrand64( (unsigned long long) rand() );
        init_genrand64( (unsigned long long) rand(), 0 );
        // use CUDA_RANDOM
        
        
        printf("\nNo. of THREADS = %d\n", num_of_threads);
        printf("No. of PROCESSORS = %d\n", num_of_procs);
        for (i=1; i < num_of_threads; i++)
        {
            // random_seed[i] = rand_r(&random_seed[cache_size*(i-1)]);
            // random_seed[cache_size*i] = genrand64_int64();
            init_genrand64( genrand64_int64(i), i );
        }
        double *start_time_loop = (double*)malloc((num_of_threads)*sizeof(double)); 
        double *end_time_loop = (double*)malloc((num_of_threads)*sizeof(double)); 
        
        #ifdef enable_CUDA_CODE
        cudaMalloc((void **)&dev_CUTOFF_SPIN, sizeof(double));
        cudaMalloc((void **)&dev_CUTOFF_S_SQ, sizeof(double));
        cudaMalloc((void **)&dev_spin, dim_S*no_of_sites*sizeof(double));
        cudaMalloc((void **)&dev_spin_temp, dim_S*no_of_sites*sizeof(double));
        cudaMalloc((void **)&dev_spin_bkp, dim_S*no_of_sites*sizeof(double));
        cudaMalloc((void **)&dev_J, dim_L*sizeof(double));
        #ifdef RANDOM_BOND
        cudaMalloc((void **)&dev_J_random, 2*dim_L*no_of_sites*sizeof(double));
        #endif
        cudaMalloc((void **)&dev_h, dim_S*sizeof(double));
        #ifdef RANDOM_FIELD
        cudaMalloc((void **)&dev_h_random, dim_S*no_of_sites*sizeof(double));
        #endif
        cudaMalloc((void **)&dev_N_N_I, 2*dim_L*no_of_sites*sizeof(long int));
        cudaMalloc((void **)&dev_black_white_checkerboard, 2*no_of_black_white_sites[0]*sizeof(long int));
        cudaMalloc((void **)&dev_no_of_black_white_sites, 2*sizeof(long int));
        cudaMalloc((void **)&dev_m, dim_S*sizeof(double));
        cudaMalloc((void **)&dev_m_bkp, dim_S*sizeof(double));
        cudaMalloc((void **)&dev_spin_reduce, dim_S*no_of_sites*sizeof(double));
        cudaMalloc((void **)&dev_delta_spin_abs_reduce, dim_S*no_of_sites*sizeof(double));
        cudaMalloc((void **)&dev_delta_spin_squared_reduce, dim_S*no_of_sites*sizeof(double));
        cudaMalloc((void **)&dev_delta_spin_max_reduce, no_of_sites*sizeof(double));
        #endif
        
        #ifdef enable_CUDA_CODE
        no_of_sites_max_power_2 = 1;
        while (no_of_sites_max_power_2 < no_of_sites-1)
        {
            no_of_sites_max_power_2 = no_of_sites_max_power_2*2;
        }
        no_of_sites_remaining_power_2 = no_of_sites - no_of_sites_max_power_2 ;
        #endif

        free(start_time_loop);
        free(end_time_loop);

        return 0;
    }
    
    int for_omp_parallelization()
    {
        
        long int i, j;
        #ifdef _OPENMP
        printf("\nOpenMP Active.\n");
        num_of_threads = omp_get_max_threads();
        num_of_procs = omp_get_num_procs();
        #else
        num_of_threads = 1;
        num_of_procs = 1;
        #endif
        
        printf("\nNo. of THREADS = %d\n", num_of_threads);
        printf("No. of PROCESSORS = %d\n", num_of_procs);

        // init_genrand64( (unsigned long long) rand() );
        // init_genrand64( (unsigned long long) rand(), 0 );
        // random_seed = (unsigned int*)malloc(cache_size*num_of_threads*sizeof(unsigned int));

        init_mt19937_parallel(num_of_threads);
        int init_key_length_NN = 312;
        int init_key_length_MM = 156;
        unsigned long long *init_keys = (unsigned long long *)malloc(init_key_length_NN*sizeof(unsigned long long));
        for (i=0; i < num_of_threads; i++)
        {
            // printf("%ld\n", time(NULL));
            // usleep(1199294);
            // printf("clock = %le\n", (double)clock()/CLOCKS_PER_SEC);
            // srand(time(NULL)+clock());

            // random_seed[i] = rand_r(&random_seed[cache_size*(i-1)]);
            // random_seed[cache_size*i] = genrand64_int64();

            // init_genrand64( genrand64_int64(i), i );

            int init_key_length = init_key_length_MM + rand()%(init_key_length_NN-init_key_length_MM);
            int j = 0;
            for (j=0; j<init_key_length; j++)
            {
                init_keys[j] = rand();
            }
            init_by_array64(init_keys, init_key_length, i);
        }
        free(init_keys);
        
        
        // double *start_time_loop = (double*)malloc((num_of_threads)*sizeof(double)); 
        // double *end_time_loop = (double*)malloc((num_of_threads)*sizeof(double)); 
        
        // for (i=num_of_threads; i>=1; i++)
        // {
        //     start_time_loop[i-1] = get_time_if_parallel();
        //     omp_set_num_threads(i);
        //     printf("\n\nNo. of THREADS = %ld \n\n", i);
        //     // field_cool_and_rotate_checkerboard(0, 1);
        //     r_order = 1;
        //     initialize_spin_config();
        //     for (j=0; j<100000; j++)
        //     {
        //         spin[genrand64_int64(thread_num_if_parallel())%(dim_S*no_of_sites)] = genrand64_real1(thread_num_if_parallel());
        //         ensemble_m();
        //         // ensemble_E();
        //     }
            
        //     // random_initialize_and_rotate_checkerboard(0, 1);
        //     end_time_loop[i-1] = get_time_if_parallel();
        // }
        
        // for (i=1; i<=num_of_threads; i++)
        // {
        //     printf("No. of THREADS = %ld ,\t Time elapsed = %g s\n", i, end_time_loop[i-1]-start_time_loop[i-1]);
        // }

        // free(start_time_loop);
        // free(end_time_loop);

        return 0;
    }
    
    #ifdef enable_CUDA_CODE
        __global__ void reduce0(int *g_idata, int *g_odata) 
        {
            extern __shared__ int sdata[];
            // each thread loads one element from global to shared mem
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
            sdata[tid] = g_idata[i];
            __syncthreads();
            // do reduction in shared mem
            for(unsigned int s=1; s < blockDim.x; s *= 2) 
            {
                if (tid % (2*s) == 0) 
                {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            // write result for this block to global mem
            if (tid == 0) g_odata[blockIdx.x] = sdata[0];
        }

        __global__ void reduce1(int *g_idata, int *g_odata) 
        {
            extern __shared__ int sdata[];
            // each thread loads one element from global to shared mem
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
            sdata[tid] = g_idata[i];
            __syncthreads();
            // do reduction in shared mem
            for (unsigned int s=1; s < blockDim.x; s *= 2) 
            {
                if (tid % (2*s) == 0) 
                {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            // write result for this block to global mem
            if (tid == 0) g_odata[blockIdx.x] = sdata[0];
        }

        __global__ void reduce2(int *g_idata, int *g_odata) 
        {
            extern __shared__ int sdata[];
            // each thread loads one element from global to shared mem
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
            sdata[tid] = g_idata[i];
            __syncthreads();
            // do reduction in shared mem
            for (unsigned int s=1; s < blockDim.x; s *= 2) 
            {
                int index = 2 * s * tid;
                if (index < blockDim.x) 
                {
                    sdata[index] += sdata[index + s];
                }
                __syncthreads();
            }
            // write result for this block to global mem
            if (tid == 0) g_odata[blockIdx.x] = sdata[0];
        }

        __global__ void reduce3(int *g_idata, int *g_odata) 
        {
            extern __shared__ int sdata[];
            // each thread loads one element from global to shared mem
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
            sdata[tid] = g_idata[i];
            __syncthreads();
            // do reduction in shared mem
            for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
            {
                if (tid < s) 
                {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }            
            // write result for this block to global mem
            if (tid == 0) g_odata[blockIdx.x] = sdata[0];
        }

        __global__ void reduce4(int *g_idata, int *g_odata) 
        {
            extern __shared__ int sdata[];
            // perform first level of reduction,
            // reading from global memory, writing to shared memory
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
            sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
            __syncthreads();
            g_idata[i];
            __syncthreads();
            // do reduction in shared mem
            for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
            {
                if (tid < s) 
                {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }            
            // write result for this block to global mem
            if (tid == 0) g_odata[blockIdx.x] = sdata[0];
        }

        __device__ void warpReduce5(volatile int* sdata, int tid) 
        {
            sdata[tid] += sdata[tid + 32];
            sdata[tid] += sdata[tid + 16];
            sdata[tid] += sdata[tid + 8];
            sdata[tid] += sdata[tid + 4];
            sdata[tid] += sdata[tid + 2];
            sdata[tid] += sdata[tid + 1];
        }
        
        __global__ void reduce5(int *g_idata, int *g_odata) 
        {
            extern __shared__ int sdata[];
            // perform first level of reduction,
            // reading from global memory, writing to shared memory
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
            sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
            __syncthreads();
            g_idata[i];
            __syncthreads();
            // do reduction in shared mem
            for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
            {
                if (tid < s) 
                {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }

            if (tid < 32) warpReduce5(sdata, tid);
            // write result for this block to global mem
            if (tid == 0) g_odata[blockIdx.x] = sdata[0];
        }
        
        template <unsigned int blockSize>
        __device__ void warpReduce6(volatile int* sdata, int tid) 
        {
            sdata[tid] += sdata[tid + 32];
            sdata[tid] += sdata[tid + 16];
            sdata[tid] += sdata[tid + 8];
            sdata[tid] += sdata[tid + 4];
            sdata[tid] += sdata[tid + 2];
            sdata[tid] += sdata[tid + 1];
        }

        template <unsigned int blockSize>
        __global__ void reduce6(int *g_idata, int *g_odata) 
        {
            extern __shared__ int sdata[];
            // perform first level of reduction,
            // reading from global memory, writing to shared memory
            unsigned int tid = threadIdx.x;
            unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
            sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
            __syncthreads();
            g_idata[i];
            __syncthreads();
            // do reduction in shared mem
            for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
            {
                if (tid < s) 
                {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            
            if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
            if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
            if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

            if (tid < 32) warpReduce(sdata, tid);
            // write result for this block to global mem
            if (tid == 0) g_odata[blockIdx.x] = sdata[0];
        }
        
        /* void call_reduce6()
        switch (threads)
        {
            case 1024:
                reduce5<1024><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
            case 512:
                reduce5<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
            case 256:
                reduce5<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
            case 128:
                reduce5<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
            case 64:
                reduce5< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
            case 32:
                reduce5< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
            case 16:
                reduce5< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
            case 8:
                reduce5< 8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
            case 4:
                reduce5< 4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
            case 2:
                reduce5< 2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
            case 1:
                reduce5< 1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
        } */
    #endif
    
    int set_output(int j)
    {
        if (j==0) { output_J = 1; }
        else if (j==1) { output_sigma_J = 1; }
        else if (j==2) { output_h = 1; }
        else if (j==3) { output_sigma_h = 1; }
        else if (j==4) { output_T = 1; }
        else if (j==5) { output_abs_m = 1; }
        else if (j==6) { output_m = 1; }
        else if (j==7) { output_m_avg = 1; }
        else if (j==8) { output_m_abs_avg = 1; }
        else if (j==9) { output_m_2_avg = 1; }
        else if (j==10) { output_m_4_avg = 1; }
        else if (j==11) { output_m_abs_vec_avg = 1; }
        else if (j==12) { output_m_2_vec_avg = 1; }
        else if (j==13) { output_m_4_vec_avg = 1; }
        else if (j==14) { output_m_ab_avg = 1; }
        else if (j==15) { output_delta_S_squared = 1; }
        else if (j==16) { output_delta_S_abs = 1; }
        else if (j==17) { output_delta_S_max = 1; }
        else if (j==18) { output_delta_M = 1; }
        else if (j==19) { output_X = 1; }
        else if (j==20) { output_X_ab = 1; }
        else if (j==21) { output_B = 1; }
        else if (j==22) { output_B_a = 1; }
        else if (j==23) { output_E = 1; }
        else if (j==24) { output_E_avg = 1; }
        else if (j==25) { output_E_2_avg = 1; }
        else if (j==26) { output_delta_E = 1; }
        else if (j==27) { output_Cv = 1; }
        else if (j==28) { output_Y_1 = 1; }
        else if (j==29) { output_Y_2 = 1; }
        else if (j==30) { output_Y_1_avg = 1; }
        else if (j==31) { output_Y_2_avg = 1; }
        else if (j==32) { output_Y_x_mu = 1; }
        return 0;
    }

    int print_help(char *function_list[], int func_len, char *output_list[], int out_len)
    {
        printf("Usage:\n$ ./a.out\n");
        printf("  -L");
        int j_L;
        for(j_L=0; j_L<dim_L;j_L++){
            printf(" <L[%d]>",j_L+1);
        }
        printf(" (Lattice dimensions)\n");
        printf("  -BC");
        for(j_L=0; j_L<dim_L;j_L++){
            printf(" <BC[%d]>",j_L+1);
        }
        printf(" (Boundary Conditions)\n");

        printf("  -th_step <Thermalizing MC Steps>\n");
        printf("  -th_algo <Glauber(0)/Metropolis(1)/Wolff(2)>\n");
        printf("  -th_updt <Checkerboard(0)/Random(1)/Linear(2)>\n");
        printf("  -av_step <Averaging MC Steps>\n");
        printf("  -av_smpl <Sampling/Measurement Interval>\n");
        printf("  -av_algo <Glauber(0)/Metropolis(1)/Wolff(2)>\n");
        printf("  -av_updt <Checkerboard(0)/Random(1)/Linear(2)>\n");
        
        printf("  -T <Temperature>\n");
        printf("  -Tmax <Maximum Temperature>\n");
        printf("  -Tmin <Minimum Temperature>\n");
        printf("  -dT <Temperature Step>\n");
        
        printf("  -h");
        int j_S;
        for(j_S=0; j_S<dim_S; j_S++){
            printf(" <h[%d]>",j_S+1);
        }
        printf(" (Uniform Field)\n");
        // printf("  -dh <field step>\n");
        #ifdef RANDOM_FIELD
        printf("  -RF");
        for(j_S=0; j_S<dim_S; j_S++){
            printf(" <RF[%d]>",j_S+1);
        }
        printf(" (Random Field Strength)\n");
        #endif

        printf("  -J");
        for(j_L=0; j_L<dim_L; j_L++){
            printf(" <J[%d]>",j_L+1);
        }
        printf(" (Uniform Bond)\n");
        // printf("  -dJ <bond step>\n");
        #ifdef RANDOM_BOND
        printf("  -RB");
        for(j_L=0; j_L<dim_L; j_L++){
            printf(" <RB[%d]>",j_L+1);
        }
        printf(" (Random Bond Strength)\n");
        #endif

        printf("  -fn <");
        while (func_len>0) {
            func_len--;
            if(func_len>0) {
                printf("%s/",function_list[func_len]);
            }
            else {
                printf("%s",function_list[func_len]);
            }

        }
        printf("> (Function Type)\n");

        printf("  -out <");
        while (out_len>0) {
            out_len--;
            if(out_len>0) {
                printf("%s/",output_list[out_len]);
            }
            else {
                printf("%s",output_list[out_len]);
            }

        }
        printf("> (Output Columns)\n");
        return 0;
    }

    void print_model_name()
    {
        char model[128];
        char *pos_end=model;
        
        #if !defined(RANDOM_FIELD) && !defined(RANDOM_BOND)
        pos_end += sprintf(pos_end, "| %dD Clean ", dim_L);
        #endif
        #if defined(RANDOM_FIELD) && !defined(RANDOM_BOND)
        pos_end += sprintf(pos_end, "| %dD Random Field ", dim_L);
        #endif
        #if !defined(RANDOM_FIELD) && defined(RANDOM_BOND)
        pos_end += sprintf(pos_end, "| %dD Random Bond ", dim_L);
        #endif
        #if defined(RANDOM_FIELD) && defined(RANDOM_BOND)
        pos_end += sprintf(pos_end, "| %dD Random Field + Bond ", dim_L);
        #endif

        #if !(dim_S-1)
        pos_end += sprintf(pos_end,"Ising Model |");
        #else
        #if !(dim_S-2)
        pos_end += sprintf(pos_end,"XY Model |");
        #else
        #if !(dim_S-3)
        pos_end += sprintf(pos_end,"Heisenberg Model |");
        #else
        pos_end += sprintf(pos_end,"O(%d) Model |", dim_S);
        #endif
        #endif
        #endif

        char *pos;
        for(pos=model; pos<pos_end; pos++) { printf("_"); } printf("\n");
        for(pos=model; pos<pos_end; pos++) { printf("%c", *pos); } printf("\n");
        for(pos=model; pos<pos_end; pos++) { printf("^"); } printf("\n");
        
    }

    int main(int argc, char *argv[])
    {
        print_model_name();
        char *function_list[] = {
            "ZTNE_inc", 
            "ZTNE_dec", 
            "EQ_init_Ordrd",
            "EQ_init_Randm"
        };
        int func_len = 4;
        int *run_this_function = calloc(0, sizeof(int));
        int no_of_functions=0;
        
        char *output_list[] = {
            "J",
            "sigma_J",
            "h",
            "sigma_h",
            "T",
            "abs_m",
            "m",
            "m_avg",
            "m_abs_avg",
            "m_2_avg",
            "m_4_avg",
            "m_abs_vec_avg",
            "m_2_vec_avg",
            "m_4_vec_avg",
            "m_ab_avg",
            "delta_S_squared",
            "delta_S_abs",
            "delta_S_max",
            "delta_M",
            "X",
            "X_ab",
            "B",
            "B_a",
            "E",
            "E_avg",
            "E_2_avg",
            "delta_E",
            "Cv",
            "Y_1",
            "Y_2",
            "Y_1_avg",
            "Y_2_avg",
            "Y_x_mu"
        };
        int out_len = 33;
        
        if (argc==1){
            print_help(function_list, func_len, output_list, out_len);
            return 0;
        }
        else{
            int i=1;
            while (i<argc){
                if ( strcmp("-th_step", argv[i])==0 ){
                    i++;
                    thermal_i = atoi(argv[i]);
                }
                else if ( strcmp("-av_step", argv[i])==0 ){
                    i++;
                    average_j = atoi(argv[i]);
                }
                else if ( strcmp("-av_smpl", argv[i])==0 ){
                    i++;
                    sampling_inter = atoi(argv[i]);
                }
                else if ( strcmp("-av_algo", argv[i])==0 ){
                    i++;
                    MC_algo_type_avg = atoi(argv[i]);
                }
                else if ( strcmp("-th_algo", argv[i])==0 ){
                    i++;
                    MC_algo_type_thrm = atoi(argv[i]);
                }
                else if ( strcmp("-av_updt", argv[i])==0 ){
                    i++;
                    MC_update_type_avg = atoi(argv[i]);
                }
                else if ( strcmp("-th_updt", argv[i])==0 ){
                    i++;
                    MC_update_type_thrm = atoi(argv[i]);
                }
                else if ( strcmp("-out", argv[i])==0 ){
                    i++;
                    int j=0;
                    while ( strcmp(output_list[j], argv[i])!=0 ){
                        j++;
                        if (j==out_len){
                            printf("Output variable not in list: %s, %s\n", argv[i], output_list[j-1]);
                            return 0;
                        }
                    }
                    set_output(j);
                }
                else if ( strcmp("-L", argv[i])==0 ){
                    int j_L;
                    for(j_L=0; j_L<dim_L; j_L++){
                        i++;
                        lattice_size[j_L] = atoi(argv[i]);
                    }
                }
                else if ( strcmp("-BC", argv[i])==0 ){
                    int j_L;
                    for(j_L=0; j_L<dim_L; j_L++){
                        i++;
                        BC[j_L] = (atoi(argv[i])==0 ? 0 : 1);
                    }
                }
                else if ( strcmp("-T", argv[i])==0 ){
                    i++;
                    T = atof(argv[i]);
                }
                else if ( strcmp("-Tmax", argv[i])==0 ){
                    i++;
                    Temp_max = atof(argv[i]);
                }
                else if ( strcmp("-Tmin", argv[i])==0 ){
                    i++;
                    Temp_min = atof(argv[i]);
                }
                else if ( strcmp("-dT", argv[i])==0 ){
                    i++;
                    delta_T = atof(argv[i]);
                }
                else if ( strcmp("-J", argv[i])==0 ){
                    int j_L;
                    for(j_L=0; j_L<dim_L; j_L++){
                        i++;
                        J[j_L] = atof(argv[i]);
                    }
                }
                else if ( strcmp("-h", argv[i])==0 ){
                    int j_S;
                    for(j_S=0; j_S<dim_S; j_S++){
                        i++;
                        h[j_S] = atof(argv[i]);
                    }
                }
                else if ( strcmp("-RB", argv[i])==0 ){
                    int j_L;
                    for(j_L=0; j_L<dim_L; j_L++){
                        i++;
                        sigma_J[j_L] = atof(argv[i]);
                    }
                }
                else if ( strcmp("-RF", argv[i])==0 ){
                    int j_S;
                    for(j_S=0; j_S<dim_S; j_S++){
                        i++;
                        sigma_h[j_S] = atof(argv[i]);
                    }
                }
                else if ( strcmp("-fn", argv[i])==0 ){
                    i++;
                    int j=0;
                    while ( strcmp(function_list[j], argv[i])!=0 ){
                        j++;
                        if (j==func_len){
                            printf("Function not in list: %s, %s\n", argv[i], function_list[j-1]);
                            return 0;
                        }
                    }
                    no_of_functions+=1;
                    run_this_function = realloc(run_this_function,no_of_functions*sizeof(int));
                    run_this_function[no_of_functions-1] = j;
                }
                i++;
            }
        }
        
        
        // return 0;
        // #ifdef enable_CUDA_CODE
        // cudaDeviceReset();
        // #endif
        srand(time(NULL)+clock());
        double abs_start_time = get_time_if_parallel();
        printf("\n---- BEGIN ----\n");
        printf("L = %d, dim_L = %d, dim_S = %d\n", lattice_size[0], dim_L, dim_S); 
        // srand(time(NULL));
        printf("RAND_MAX = %lf,\n sizeof(int) = %ld,\n sizeof(long) = %ld,\n sizeof(double) = %ld,\n sizeof(long int) = %ld,\n sizeof(short int) = %ld,\n sizeof(unsigned int) = %ld,\n sizeof(RAND_MAX) = %ld\n", (double)RAND_MAX, sizeof(int), sizeof(long), sizeof(double), sizeof(long int), sizeof(short int), sizeof(unsigned int), sizeof(RAND_MAX));
        
        allocate_memory();
        
        int j_L, j_S;
        // no_of_sites = custom_int_pow(lattice_size, dim_L);
        #ifdef enable_CUDA_CODE
        for_cuda_parallelization();
        #else
            #ifdef _OPENMP
            for_omp_parallelization();
            #else
            for_omp_parallelization();
            #endif
        #endif

        // initialize_nearest_neighbor_index();
        initialize_nearest_neighbor_index_2();
        printf("[t=%lf s] \n", get_time_if_parallel()-abs_start_time);
        // printf("nearest neighbor initialized. \n");
        
        #if defined (UPDATE_CHKR_NON_EQ) || defined (UPDATE_CHKR_EQ_MC)
            initialize_checkerboard_sites();
            // j_L = initialize_checkerboard_sites_2();
            // if (j_L==-1){return -1;}
            printf("[t=%lf s] \n", get_time_if_parallel()-abs_start_time);
        #endif
        
        #ifdef RANDOM_BOND
        load_J_config("");
        // printf("J loaded. \n");
        printf("[t=%lf s] \n", get_time_if_parallel()-abs_start_time);
        #endif
        
        #ifdef RANDOM_FIELD
        // initialize_h_random_gaussian();
        load_h_config("");
        // printf("h loaded. \n");
        printf("[t=%lf s] \n", get_time_if_parallel()-abs_start_time);
        #endif

        // mergeSort(h_random, 0, no_of_sites-1);
        // long int * h_sorted = sort_h_index(h_random, no_of_sites);

        long int i, j;
        
        int fi;
        for (fi=0; fi<no_of_functions; fi++)
        {
            if (run_this_function[fi] == 0){
                zero_temp_IM_hysteresis_with_changing_field(-1);
            }
            else if (run_this_function[fi] == 1){
                zero_temp_IM_hysteresis_with_changing_field(1);
            }
            else if (run_this_function[fi] == 2){
                evo_diff_ini_config_temp(0);
            }
            else if (run_this_function[fi] == 3){
                evo_diff_ini_config_temp(2);
            }
        }
        // thermal_i = thermal_i*lattice_size[0];
        // average_j = average_j*lattice_size[0];
        
        // checking_O3_spin_with_O2_RF();
        // for (i=0; i<10; i++){
        //     #ifdef GAUSSIAN_FIELD // for random gaussian fields
        //     initialize_h_random_gaussian();
        //     #endif
        //     #ifdef BIMODAL_FIELD // for random bimodal fields
        //     initialize_h_random_bimodal();
        //     #endif
        //     zero_temp_spin_at_hc(-1, i);
        //     zero_temp_spin_at_hc(1, i);
        // }
        // zero_temp_IM_hysteresis_with_changing_field(1);
        // zero_temp_RFIM_hysteresis();
        // zero_temp_RFIM_ringdown(0.1);
        // zero_temp_RFIM_return_point_memory();
        // hysteresis_protocol(0, -1);
        // fc_fh_or_both(2, 0);
        // evo_diff_ini_config_temp(0);
        // evo_diff_ini_config_temp(2);
        // return 0;

        int is_chkpt = 0;
        
        
        /* // double h_sweep_vals[] = { 0.1000, 0.2000, 0.3000, 0.4000, 0.5000 };
        double h_sweep_vals[] = { 0.5000 };
        int len_h_sweep_vals = sizeof(h_sweep_vals) / sizeof(h_sweep_vals[0]);
        for (i=0; i<len_h_sweep_vals; i++)
        {
            double start_time_loop[2];
            double end_time_loop[2];
            
            sigma_h[0] = 2.270;
            // sigma_h[1] = 1.000;
            load_h_config("");

            double h_sweep[dim_S] = { 0.0 };
            h_sweep[0] = 0.1*h_sweep_vals[i];
            // h_sweep[1] = 5*h_sweep_vals[i];

            start_time_loop[0] = get_time_if_parallel();
            
            // is_chkpt = random_initialize_and_axis_checkerboard(1, 1, h_sweep, 2);
            is_chkpt = ordered_initialize_and_axis_checkerboard(0, 1, h_sweep, 0, 0);
            // is_chkpt = zero_temp_RFXY_hysteresis_axis_checkerboard_old(1, 1, h_sweep);
            // is_chkpt = finite_temp_RFXY_hysteresis_axis_checkerboard_old(1, 1, h_sweep);
            
            end_time_loop[0] = get_time_if_parallel();
            
            if (is_chkpt == -1)
            {
                printf("is_chkpt = %d\n", is_chkpt);
                printf("\nIncomplete sweeping hysteresi : Exit after = %lf s\n", end_time_loop[0] - start_time_loop[0] );
                break;
            }
            // printf("\nSweeping hysteresis along x ( |h|=%lf ) = %lf \n", h[0], end_time_loop[0] - start_time_loop[0] );
            // printf("\nSweeping hysteresis along y ( |h|=%lf ) = %lf \n", h[1], end_time_loop[0] - start_time_loop[0] );
        } */ 
        
        /* is_chkpt = 0;
        // double h_field_vals[] = { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15 };
        // double h_field_vals[] = { 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.048, 0.050, 0.052, 0.054, 0.056, 0.058, 0.060, 0.064, 0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150 };
        // double h_field_vals[] = { 0.100, 0.500, 1.000, 0.800, 0.300, 2.000 };
        double h_field_vals[] = { 0.1480 };
        int len_h_field_vals = sizeof(h_field_vals) / sizeof(h_field_vals[0]);
        for (i=0; i<len_h_field_vals; i++)
        {
            is_chkpt = -1;
            while (is_chkpt == -1)
            {
                
                double start_time_loop[2];
                double end_time_loop[2];
                
                
                // sigma_h[0] = h_field_vals[i];
                // sigma_h[0] = 0.500;
                // sigma_h[1] = h_field_vals[i];
                // sigma_h[1] = 0.000;
                load_h_config("");
                
                start_time_loop[0] = get_time_if_parallel();
                // field_cool_and_rotate_checkerboard(0, 1);
                // random_initialize_and_rotate_checkerboard(0, 1);
                // zero_temp_RFXY_hysteresis_axis_checkerboard_old(0, -1);
                // zero_temp_RFXY_hysteresis_axis_checkerboard_old(1, -1);
                
                start_time = get_time_if_parallel();
                is_chkpt = ordered_initialize_and_rotate_checkerboard(1, 1, h_field_vals[i], 0.000);
                
                
                end_time_loop[0] = get_time_if_parallel();
                
                printf("is_chkpt = %d\n", is_chkpt);
                if (is_chkpt == -1)
                {
                    printf("\nIncomplete rotating hysteresis : Exit after = %lf s\n", end_time_loop[0] - start_time_loop[0] );
                }
                else
                {
                    printf("\nCompleted rotating hysteresis : Exit after = %lf s\n", end_time_loop[0] - start_time_loop[0] );
                }
                
                start_time_loop[1] = get_time_if_parallel();
                // start_time = get_time_if_parallel();
                // zero_temp_RFXY_hysteresis_axis_checkerboard(1, 1);
                // evolution_at_T(100);
                end_time_loop[1] = get_time_if_parallel();
                
                // printf("\nCooling protocol time (from T=%lf to T=%lf) = %lf \n", Temp_max, Temp_min, end_time_loop[0] - start_time_loop[0] );
                // printf("\nRotating hysteresis starting from x ( |h|=%lf ) = %lf \n", h[0], end_time_loop[0] - start_time_loop[0] );
                
                // printf("\nRotating hysteresis starting from y ( |h|=%lf ) = %lf \n", h[1], end_time_loop[0] - start_time_loop[0] );
                
                // printf("\nHysteresis along x ( Max(|sigma_h|)=%lf ) = %lf \n", sigma_h[0], end_time_loop[0] - start_time_loop[0] );
                // printf("\nHysteresis along y ( Max(|h|)=%lf ) = %lf \n", h_max+h_i_max, end_time_loop[1] - start_time_loop[1] );
                // printf("\nEvolution time (at T=%lf) = %lf \n", T, end_time_loop[1] - start_time_loop[1] );
                #ifdef WALLTIME_LIMIT
                if ( end_time_loop[1] - abs_start_time > WALLTIME_LIMIT )
                {
                    printf("\n %lf s Elapsed: To be continued... \n", end_time_loop[1] - abs_start_time );
                    break;
                }
                #endif
            }
        } */
        
        // zero_temp_RFIM_hysteresis();
        // zero_temp_RFIM_ringdown(0.1);
        // zero_temp_RFIM_return_point_memory();
        
        /* 
        MC_algo_type = 1;
        MC_update_type = 0;
        h_order = 0;
        r_order = 0;
        evo_diff_ini_config_temp(0);
        */
        
        // h[0] = 0.1;
        // MC_algo_type = 2;
        // MC_update_type = 0;
        // h_order = 0;
        // r_order = 1;
        // fc_fh_or_both(2,0);
        // evolution_at_T(1);
        // evo_diff_ini_config_temp(2);
        
        
        /* 
        MC_algo_type = 1;
        MC_update_type = 0;
        h_order = 0;
        r_order = 1;
        fc_fh_or_both(2,1);
        */
        
        /* 
        MC_algo_type = 0;
        MC_update_type = 1;
        h_order = 0;
        r_order = 0;
        {
            long int hysteresis_MCS_min = 1; 
            long int hysteresis_MCS_max = 100;
            int hysteresis_MCS_multiplier = 10;
            for(T=Temp_min; T<=Temp_max; T=T+delta_T)
            {
                for(hysteresis_MCS=hysteresis_MCS_min; hysteresis_MCS<=hysteresis_MCS_max; hysteresis_MCS=hysteresis_MCS*hysteresis_MCS_multiplier)
                {
                    for(j_S=0; j_S<dim_S; j_S++)
                    {
                        for(order[j_S]=-1; order[j_S]<2; order[j_S]=order[j_S]+2)
                        {
                            hysteresis_protocol(j_S, order[j_S]);
                        }
                        order[j_S] = 0;
                    }
                }
            }
        } */
        /* 
        MC_algo_type = 0;
        MC_update_type = 0;
        h_order = 0;
        r_order = 0;
        {
            int hysteresis_MCS_multiplier = 10;
            long int hysteresis_MCS_min = 1; 
            long int hysteresis_MCS_max = 100;
            for(T=Temp_min; T<=Temp_max; T=T+delta_T)
            {
                for(hysteresis_MCS=hysteresis_MCS_min; hysteresis_MCS<=hysteresis_MCS_max; hysteresis_MCS=hysteresis_MCS*hysteresis_MCS_multiplier)
                {
                    for (j_S=0; j_S<dim_S; j_S++)
                    {
                        order[j_S] = 0;
                    }
                    order[0] = 1;
                    hysteresis_protocol(0, order[0]);
                }
            }
        } */
        
        
        
        free_memory();
        double abs_end_time = get_time_if_parallel();
        printf("\nCPU Time elapsed total = %lf s\n", abs_end_time-abs_start_time);
        printf("\n----- END -----\n");
        // is_chkpt = -1;
        return -is_chkpt;
    }

//========================================================================//

/*
    volatile bool flag=false;

    #pragma omp parallel for shared(flag)
    for(int i=0; i<=100000; ++i)
    {
        if(flag) continue;
        if(element[i] ...)
        {
            ...
            flag=true;
        }
    }
*/

// output_J = 1;
// output_sigma_J = 1;
// output_h = 1;
// output_sigma_h = 1;
// output_T = 1;
// output_abs_m = 1;
// output_m = 1;
// output_m_avg = 1;
// output_m_abs_avg = 1;
// output_m_2_avg = 1;
// output_m_4_avg = 1;
// output_m_abs_vec_avg = 1;
// output_m_2_vec_avg = 1;
// output_m_4_vec_avg = 1;
// output_m_ab_avg = 1;
// output_delta_S_squared = 1;
// output_delta_S_abs = 1;
// output_delta_S_max = 1;
// output_delta_M = 1;
// output_X = 1;
// output_X_ab = 1;
// output_B = 1;
// output_B_a = 1;
// output_E = 1;
// output_E_avg = 1;
// output_E_2_avg = 1;
// output_delta_E = 1;
// output_Cv = 1;
// output_Y_1 = 1;
// output_Y_2 = 1;
// output_Y_1_avg = 1;
// output_Y_2_avg = 1;
// output_Y_x_mu = 1;
// output_prepend = 1;
// output_append = 1;
