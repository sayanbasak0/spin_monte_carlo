// using bitbucket
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "mt19937-64.h"
#include <omp.h>
#include <unistd.h> // chdir 
#include <errno.h> // strerror
#include <sys/types.h>
#include <sys/stat.h>
#ifdef enable_CUDA_CODE
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

#define CUDA_with_managed 1

#define MARSAGLIA 1 // uncomment only one
// #define REJECTION 1 // uncomment only one
// #define BOX_MULLER 1 // uncomment only one
// #define OLD_COMPILER 1
// #define BUNDLE 4

// #define CONST_RATE 1 // uncomment only one
// #define DIVIDE_BY_SLOPE 1 // uncomment only one
#define BINARY_DIVISION 1 // uncomment only one
// #define DYNAMIC_BINARY_DIVISION 1 // uncomment only one
// #define DYNAMIC_BINARY_DIVISION_BY_SLOPE 1 // uncomment only one

#define RANDOM_FIELD 1
// #define RANDOM_BOND 1

// #define RFIM 1

#define dim_L 2
#define dim_S 2

// #define SAVE_SPIN_AFTER 250

#define TYPE_VOID 0
#define TYPE_INT 1
#define TYPE_LONGINT 2
#define TYPE_FLOAT 3
#define TYPE_DOUBLE 4

#define CHECKPOINT_TIME 60.00 // in seconds
#define RESTORE_CHKPT_VALUE 1 // 0 for initialization, 1 for restoring

// #define UPDATE_ALL_NON_EQ 1 // uncomment only one
#define UPDATE_CHKR_NON_EQ 1 // uncomment only one

#define UPDATE_CHKR_EQ_MC 1 

#define CHECK_AVALANCHE 1

//===============================================================================//
//====================      Variables                        ====================//

    FILE *pFile_1, *pFile_2, *pFile_output, *pFile_chkpt;
    char output_file_0[256];

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
    

//===============================================================================//
//====================      Lattice size                     ====================//
    int lattice_size[dim_L] = { 128, 128 }; // lattice_size[dim_L]
    long int no_of_sites;
    long int no_of_black_sites;
    long int no_of_white_sites;
    long int no_of_black_white_sites[2];

//====================      Checkerboard variables           ====================//
    // long int *black_white_checkerboard[2]; 
    long int *black_white_checkerboard; 
    #if defined (UPDATE_CHKR_NON_EQ) || defined (UPDATE_CHKR_EQ_MC)
        int black_white_checkerboard_reqd = 1;
    #else
        int black_white_checkerboard_reqd = 0;
    #endif
    long int *black_checkerboard; int black_checkerboard_reqd = 0;
    long int *white_checkerboard; int white_checkerboard_reqd = 0;
    int site_to_dir_index[dim_L];

//====================      Ising hysteresis sorted list     ====================//
    long int *sorted_h_index; 
    #if defined (RFIM)
        int sorted_h_index_reqd = 1; 
    #else
        int sorted_h_index_reqd = 0; 
    #endif
    long int *next_in_queue; 
    #if defined (RFIM)
        int next_in_queue_reqd = 1;
    #else
        int next_in_queue_reqd = 0;
    #endif
    long int remaining_sites;

//====================      Wolff/Cluster variables          ====================//
    double reflection_plane[dim_S];
    int *cluster; int cluster_reqd = 0;

//====================      Near neighbor /Boundary Cond     ====================//
    long int *N_N_I; int N_N_I_reqd = 1;
    double BC[dim_L] = { 1, 1 }; // 1 -> Periodic | 0 -> Open | -1 -> Anti-periodic -- Boundary Condition

//====================      Spin variable                    ====================//
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

//====================      Initialization type              ====================//
    double order[dim_S] = { 1.0, 0.0 }; // order[dim_S]
    int h_order = 0; // 0/1
    int r_order = 0; // 0/1
    int or_ho_ra = 0;
    char o_h_r[] = "ohr";
    int ax_ro = 1;
    char a_r[] = "ar";

//====================      MC-update type                   ====================//
    int Gl_Me_Wo = 1;
    char G_M_W[] = "GMW";
    int Ch_Ra_Li = 0;
    char C_R_L[] = "CRL";

//====================      NN-interaction (J)               ====================//
    double J[dim_L] = { 1.0, 1.0 }; 
    double sigma_J[dim_L] = { 0.0, 0.0 };
    double *J_random;
    #ifdef RANDOM_BOND
        int J_random_reqd = 1;
    #else
        int J_random_reqd = 0;
    #endif
    double J_max = 0.0;
    double J_min = 0.0;
    double delta_J = 0.01, J_i_max = 0.0, J_i_min = 0.0; // for hysteresis
    double J_dev_net[dim_L];
    double J_dev_avg[dim_L];

//====================      on-site field (h)                ====================//
    double h[dim_S] = { 0.0, 0.0 }; // h[0] = 0.1; // h[dim_S]
    double sigma_h[dim_S] = { 2.00, 0.00 }; 
    double *h_random;
    #ifdef RANDOM_FIELD
        int h_random_reqd = 1;
    #else
        int h_random_reqd = 0;
    #endif
    double h_max = 4.01; double h_min = -4.01;
    double del_h = 0.001, del_h_cutoff = 0.000001, h_i_max = 0.0, h_i_min = 0.0; // for hysteresis (axial)
    double del_phi = 0.0001, del_phi_cutoff = 0.00000001; // for hysteresis (rotating)
    double h_dev_net[dim_S];
    double h_dev_avg[dim_S];
    double *field_site; int field_site_reqd = 0; // field experienced by spin due to nearest neighbors and on-site field

//====================      Temperature                      ====================//
    double T = 3.00;
    double Temp_min = 0.02;
    double Temp_max = 2.00;
    double delta_T = 0.05;

//====================      Avalanche delta(S)               ====================//
    double delta_S_squared[dim_S] = {0.0, 0.0 };
    double delta_S_abs[dim_S] = { 0.0, 0.0 };
    double delta_S_max = 0.0 ;
    double delta_M[dim_S] = { 0.0, 0.0 };

//====================      Magnetisation <M>                ====================//
    double m[dim_S];
    double m_bkp[dim_S];
    
    double abs_m[dim_S];
    double m_sum[dim_S];
    double m_avg[dim_S];

    double m_abs_sum = 0;
    double m_abs_avg = 0;
    double m_2_sum = 0;
    double m_2_avg = 0;
    double m_2_vec_sum[dim_S] = { 0 };
    double m_2_vec_avg[dim_S] = { 0 };

    double m_4_sum = 0;
    double m_4_avg = 0;
    double m_4_vec_sum[dim_S] = { 0 };
    double m_4_vec_avg[dim_S] = { 0 };

    double m_ab[dim_S*dim_S] = { 0 };
    double m_ab_sum[dim_S*dim_S] = { 0 };
    double m_ab_avg[dim_S*dim_S] = { 0 };

//====================      Energy <E>                       ====================//
    double E = 0;
    double E_bkp[dim_S];
    double E_sum = 0;
    double E_avg = 0;
    double E_2_sum = 0;
    double E_2_avg = 0;

//====================      Helicity <Y>                     ====================//
    double Y_1[dim_S*dim_S*dim_L] = { 0 };
    double Y_2[dim_S*dim_S*dim_L] = { 0 };
    double Y_1_sum[dim_S*dim_S*dim_L] = { 0 };
    double Y_2_sum[dim_S*dim_S*dim_L] = { 0 };
    double Y_1_avg[dim_S*dim_S*dim_L] = { 0 };
    double Y_2_avg[dim_S*dim_S*dim_L] = { 0 };
    double Y_ab_mu[dim_S*dim_S*dim_L] = { 0 };

//====================      Specific heat Cv                 ====================//
    double Cv = 0;

//====================      Susceptibility (Tensor) X        ====================//
    double X = 0;
    double X_ab[dim_S*dim_S] = { 0 };

//====================      Binder Parameter B               ====================//
    double B = 0;
    double Bx = 0;
    double By = 0;

//====================      MC-update iterations             ====================//
    long int thermal_i = 128*10*10*10; // ! *=lattice_size
    long int average_j = 128*10*10; // ! *=lattice_size
    long int sampling_inter = 16; // *=sampling_inter-rand()%sampling_inter

//====================      Hysteresis                       ====================//
    long int hysteresis_MCS = 1; 
    long int hysteresis_MCS_min = 1; 
    long int hysteresis_MCS_max = 100;
    int hysteresis_repeat = 32;
    long int hysteresis_MCS_multiplier = 10;
    // #define CUTOFF_BY_SUM 1 // for find_change sum
    #ifdef CUTOFF_BY_SUM
    double CUTOFF_SPIN = 1.0000000000000e-10; 
    #endif
    #define CUTOFF_BY_MAX 1 // for find_change max
    #ifdef CUTOFF_BY_MAX
    double CUTOFF_SPIN = 1.00000000000000e-14; 
    #endif
    double CUTOFF_M_SQ = 1.00000000e-08;
    double CUTOFF_M_SQ_BY_4 = 2.50000000e-09;
    double cutoff_check[3] = { 0.0, 0.0, 0.0 };

//====================      CUDA device ptr                  ====================//
    #ifdef enable_CUDA_CODE
        #ifdef CUDA_with_managed
        __managed__ __device__ double *dev_spin;
        __managed__ __device__ double *dev_spin_temp;
        __managed__ __device__ double *dev_spin_bkp;
        __managed__ __device__ double *dev_CUTOFF_SPIN;
        __managed__ __device__ double *dev_J;
        __managed__ __device__ double *dev_J_random;
        __managed__ __device__ double *dev_h;
        __managed__ __device__ double *dev_h_random;
        __managed__ __device__ long int *dev_N_N_I;
        __managed__ __device__ double *dev_m;
        __managed__ __device__ double *dev_m_bkp;
        __managed__ __device__ double *dev_spin_reduce;
        #else
        __device__ double *dev_spin;
        __device__ double *dev_spin_temp;
        __device__ double *dev_spin_bkp;
        __device__ double *dev_CUTOFF_SPIN;
        __device__ double *dev_J;
        __device__ double *dev_J_random;
        __device__ double *dev_h;
        __device__ double *dev_h_random;
        __device__ long int *dev_N_N_I;
        __device__ double *dev_m;
        __device__ double *dev_m_bkp;
        __device__ double *dev_spin_reduce;
        #endif 
    #endif 
    
    long int no_of_sites_max_power_2;
    long int no_of_sites_remaining_power_2;

//===============================================================================//

//===============================================================================//
//====================      Functions                        ====================//
//===============================================================================//
//====================      Custom                           ====================//

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

    int direction_index(long int xyzi)
    {
        int j_L;
        // long int temp_1 = pow(lattice_size, j);
        long int temp_1 = 1;
        for(j_L=0; j_L<dim_L; j_L++)
        {
            site_to_dir_index[j_L] = ((long int)(xyzi/temp_1))%lattice_size[j_L];
            temp_1 = temp_1 * lattice_size[j_L];
        }
        // long int temp_1 = custom_int_pow(lattice_size, j);
        return 0;
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
                X = (-10 + 20 * genrand64_real1());
                r = genrand64_real1();
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
                U1 = -1.0 + 2.0 * genrand64_real2() ; // ((double) rand () / RAND_MAX) * 2;
                U2 = -1.0 + 2.0 * genrand64_real2() ; // ((double) rand () / RAND_MAX) * 2;
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
            rho = genrand64_real2();
            theta = genrand64_real2();
            
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

//====================      Initialization                   ====================//

    int initialize_h_zero()
    {
        long int i;

        for(i=0; i<dim_S*no_of_sites; i=i+1)
        {
            h_random[i] = 0;
            // h_total[i] = h; 
        }
        return 0;
    }

    int initialize_h_random_gaussian()
    {
        long int i, r_i;
        int j_S;
        
        #ifdef RANDOM_FIELD
        initialize_h_zero();
        for(j_S=0; j_S<dim_S; j_S=j_S+1)
        {
            h_dev_net[j_S] = 0;
            for(i=0; i<no_of_sites; i=i+1)
            {
                do
                {
                    r_i = rand()%no_of_sites;
                } while(h_random[dim_S*r_i + j_S]!=0);
                h_random[dim_S*r_i + j_S] = sigma_h[j_S] * generate_gaussian();
                h_dev_net[j_S] += h_random[dim_S*r_i + j_S];

                if (h_random[dim_S*r_i + j_S]>h_i_max)
                {
                    h_i_max = h_random[dim_S*r_i + j_S];
                }
                else if (h_random[dim_S*r_i + j_S]<h_i_min)
                {
                    h_i_min = h_random[dim_S*r_i + j_S];
                }
            }
            
            h_dev_avg[j_S] = h_dev_net[j_S] / no_of_sites;
        }
        // if (fabs(h_i_max) < fabs(h_i_min))
        // {
        //     h_i_max = fabs(h_i_min);
        // }
        // else
        // {
        //     h_i_max = fabs(h_i_max);
        // }
        // h_i_min = -h_i_max;
        #endif

        return 0;
    }

    int initialize_J_zero()
    {
        long int i;
        for(i=0; i<2*dim_L*no_of_sites; i=i+1)
        {
            J_random[i] = 0;
            // h_total[i] = h; 
        }
        return 0;
    }

    int initialize_J_random_gaussian()
    {
        long int i, r_i;
        int j_L, k_L;
        
        #ifdef RANDOM_BOND
        initialize_J_zero();
        for(j_L=0; j_L<dim_L; j_L=j_L+1)
        {
            J_dev_net[j_L] = 0;
            for(i=0; i<no_of_sites; i=i+1)
            {
                do
                {
                    r_i = rand()%no_of_sites;
                } while(J_random[2*dim_L*r_i + 2*j_L]!=0);
                J_random[2*dim_L*r_i + 2*j_L] = sigma_J[j_L] * generate_gaussian();
                J_dev_net[j_L] += J_random[2*dim_L*r_i + 2*j_L];

                if (J_random[2*dim_L*r_i + 2*j_L]>J_i_max)
                {
                    J_i_max = J_random[2*dim_L*r_i + 2*j_L];
                }
                else if (J_random[2*dim_L*r_i + 2*j_L]<J_i_min)
                {
                    J_i_min = J_random[2*dim_L*r_i + 2*j_L];
                }
                J_random[2*dim_L*N_N_I[2*dim_L*r_i + 2*j_L] + 2*j_L + 1] = J_random[2*dim_L*r_i + 2*j_L];
            }
            J_dev_avg[j_L] = J_dev_net[j_L] / no_of_sites;
        }
        #endif
        
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
        for(i=0; i<no_of_sites; i=i+1)
        {
            for(j_L=0; j_L<dim_L; j_L=j_L+1)
            {
                for(k_L=0; k_L<2; k_L=k_L+1)
                {
                    N_N_I[i*2*dim_L + 2*j_L + k_L] = nearest_neighbor(i, j_L, k_L);
                }
            }
        }
        printf("Nearest neighbor initialized. \n");
        return 0;
    }

    int initialize_checkerboard_sites()
    {
        long int i = 0;
        int j_L;
        
        int dir_index_sum;
        
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

        for (i=0; i<no_of_sites; i++)
        {
            direction_index(i);
            dir_index_sum = 0;
            for (j_L=0; j_L<dim_L; j_L++)
            {
                dir_index_sum = dir_index_sum + site_to_dir_index[j_L];
            }
            if (dir_index_sum % 2 == black_white[0])
            {
                // black_checkerboard[black_index] = i;
                black_white_checkerboard[0+black_white_index[0]] = i;
                // if ( black_checkerboard[black_white_index[0]] - black_white_checkerboard[0][black_white_index[0]] != 0 )
                // {
                //     printf("black_checkerboard[i] = %ld, black_white_checkerboard[0][i] = %ld; %ld\n", black_checkerboard[black_white_index[0]], black_white_checkerboard[0][black_white_index[0]], black_checkerboard[black_white_index[0]] - black_white_checkerboard[0][black_white_index[0]] );
                // }
                black_white_index[0]++;
                // black_index++;
                
            }
            else //if (dir_index_sum % 2 == white)
            {
                // white_checkerboard[white_index] = i;
                black_white_checkerboard[no_of_black_white_sites[0]+black_white_index[1]] = i;
                // if ( white_checkerboard[black_white_index[1]] - black_white_checkerboard[1][black_white_index[1]] != 0 )
                // {
                //     printf("white_checkerboard[i] = %ld, black_white_checkerboard[1][i] = %ld; %ld\n", white_checkerboard[black_white_index[1]], black_white_checkerboard[1][black_white_index[1]], white_checkerboard[black_white_index[1]] - black_white_checkerboard[1][black_white_index[1]] );
                // }
                black_white_index[1]++;
                // white_index++;
            }
        }
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
                    spin[dim_S*i+j_S] = (-1.0 + 2.0 * (double)rand_r(&random_seed[cache_size*omp_get_thread_num()])/(double)(RAND_MAX));
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

    
//====================      Save J, h, Spin                  ====================//

    int save_spin_config(char append_string[], char write_mode[])
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

        char output_file_1[256];
        char *pos = output_file_1;
        pos += sprintf(pos, "Spin_%lf_", T);
        for (j_S = 0 ; j_S != dim_S ; j_S++) 
        {
            if (j_S) 
            {
                pos += sprintf(pos, "-");
            }
            pos += sprintf(pos, "%lf", h[j_S]);
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

        printf("Saved spin config. Output file name: %s\n", output_file_1);

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
        

        char output_file_1[256];
        char *pos = output_file_1;
        pos += sprintf(pos, "h_config_");
        for (j_S = 0 ; j_S != dim_S ; j_S++) 
        {
            if (j_S) 
            {
                pos += sprintf(pos, "-");
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

        /* for (i = 0; i < no_of_sites; i++)
        {
            for (j_S = 0; j_S<dim_S; j_S++)
            {
                printf("|%lf|", h_random[dim_S*i + j_S]);
            }
            printf("\n");
        }
        printf("\n"); */
        
        return 0;
    }

    int save_J_config(char append_string[])
    {
        long int i;
        int j_L, k_L;
        
        
        char output_file_1[256];
        char *pos = output_file_1;
        pos += sprintf(pos, "J_config_");
        for (j_L = 0 ; j_L != dim_L ; j_L++) 
        {
            if (j_L) 
            {
                pos += sprintf(pos, "-");
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
        
        return 0;
    }

//====================      Load J, h, Spin                  ====================//

    int load_spin_config(char append_string[])
    {
        
        long int i;
        int j_S, j_L;

        char input_file_1[256];
        char *pos = input_file_1;
        pos += sprintf(pos, "Spin_%lf_", T);
        for (j_S = 0 ; j_S != dim_S ; j_S++) 
        {
            if (j_S) 
            {
                pos += sprintf(pos, "-");
            }
            pos += sprintf(pos, "%lf", h[j_S]);
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
        char input_file_1[256];
        char *pos = input_file_1;
        pos += sprintf(pos, "h_config_");
        for (j_S = 0 ; j_S != dim_S ; j_S++) 
        {
            if (j_S) 
            {
                pos += sprintf(pos, "-");
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
        strcat(input_file_1, append_string);
        strcat(input_file_1, ".dat");
        
        pFile_1 = fopen(input_file_1, "r"); // opens file for reading

        if (pFile_1 == NULL)
        {
            // h_random = (double*)malloc(dim_S*no_of_sites*sizeof(double));
            initialize_h_random_gaussian();

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

        return 0;
    }

    int load_J_config(char append_string[])
    {
        //---------------------------------------------------------------------------------------//
        
        // J_random = (double*)malloc(2*dim_L*no_of_sites*sizeof(double));
        long int i;
        int j_L, k_L;
        char input_file_1[256];
        char *pos = input_file_1;
        pos += sprintf(pos, "J_config_");
        for (j_L = 0 ; j_L != dim_L ; j_L++) 
        {
            if (j_L) 
            {
                pos += sprintf(pos, "-");
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
        strcat(input_file_1, append_string);
        strcat(input_file_1, ".dat");
        
        pFile_1 = fopen(input_file_1, "r"); // opens file for reading

        if (pFile_1 == NULL)
        {
            // J_random = (double*)malloc(2*dim_L*no_of_sites*sizeof(double));
            initialize_J_random_gaussian();

            save_J_config(append_string); // creates file for later
            printf("Initialized J_random config. Output file name: %s\n", input_file_1);
        }
        else
        {
            // J_random = (double*)malloc(2*dim_L*no_of_sites*sizeof(double));
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

        return 0;
    }

//====================      Checkpoint                       ====================//

    int restore_checkpoint(int startif, int array_type, int array_length, void *voidarray, int stopif)
    {
        int j_arr, j_L;
        static int restore_point_exist = 1;
        if (restore_point_exist == 0)
        {
            return 0;
        }
        if (startif == 1)
        {
            char chkpt_file_1[256];
            char *pos = chkpt_file_1;
            pos += sprintf(pos, "Data_%lf_", T);
            if (array_type == TYPE_INT)
            {
                int *array = (int*)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, "-");
                    }
                    pos += sprintf(pos, "%d", array[j_arr]);
                }
            }
            if (array_type == TYPE_LONGINT)
            {
                long int *array = (long int*)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, "-");
                    }
                    pos += sprintf(pos, "%ld", array[j_arr]);
                }
            }
            if (array_type == TYPE_FLOAT)
            {
                float *array = (float*)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, "-");
                    }
                    pos += sprintf(pos, "%f", array[j_arr]);
                }
            }
            if (array_type == TYPE_DOUBLE)
            {
                double *array = (double*)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, "-");
                    }
                    pos += sprintf(pos, "%lf", array[j_arr]);
                }
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
            // strcat(chkpt_file_1, append_string);
            strcat(chkpt_file_1, "_chkpt.dat");
                
            pFile_chkpt = fopen(chkpt_file_1, "r"); // opens new file for reading
            if (pFile_chkpt == NULL)
            {
                restore_point_exist = 0;
                printf("\n---- Starting from Initial Conditions ----\n");
                return 0;
            }
            load_spin_config("_chkpt");
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
        int j_arr, j_L;
        if (startif == 1)
        {
            char chkpt_file_1[256];
            char *pos = chkpt_file_1;
            pos += sprintf(pos, "Data_%lf_", T);
            if (array_type == TYPE_INT)
            {
                int *array = (int *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, "-");
                    }
                    pos += sprintf(pos, "%d", array[j_arr]);
                }
            }
            if (array_type == TYPE_LONGINT)
            {
                long int *array = (long int *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, "-");
                    }
                    pos += sprintf(pos, "%ld", array[j_arr]);
                }
            }
            if (array_type == TYPE_FLOAT)
            {
                float *array = (float *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, "-");
                    }
                    pos += sprintf(pos, "%f", array[j_arr]);
                }
            }
            if (array_type == TYPE_DOUBLE)
            {
                double *array = (double *)voidarray;
                for (j_arr = 0 ; j_arr != array_length ; j_arr++) 
                {
                    if (j_arr) 
                    {
                        pos += sprintf(pos, "-");
                    }
                    pos += sprintf(pos, "%lf", array[j_arr]);
                }
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
            // strcat(chkpt_file_1, append_string);
            strcat(chkpt_file_1, "_chkpt.dat");
                
            pFile_chkpt = fopen(chkpt_file_1, "w"); // opens new file for writing
            
            save_spin_config("_chkpt", "w");
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
            double time_now = omp_get_wtime();
            printf("\n---- Checkpoint after %lf seconds ----\n", time_now - start_time);
            return 1;
        }

        return 0;
    }

//====================      Print Output                     ====================//

    int print_header_column(char output_file_name[])
    {
        int j_L, j_S;
        printf("Output file name: %s\n", output_file_name);
        pFile_output = fopen(output_file_name, "a");
        fprintf(pFile_output, "----------------------------------------------------------------------------------\n");
        
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

        fprintf(pFile_output, "T = %.12e ,\n", T);
        
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

        fprintf(pFile_output, "sigma_J = (");
        for (j_L=0; j_L<dim_L; j_L++)
        {
            if (j_L)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.12e ", sigma_J[j_L]);
        }
        fprintf(pFile_output, ") ,\n");

        fprintf(pFile_output, "<J_{ij}> = (");
        for (j_L=0; j_L<dim_L; j_L++)
        {
            if (j_L)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.12e ", J_dev_avg[j_L]);
        }
        fprintf(pFile_output, ") ,\n");

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

        fprintf(pFile_output, "sigma_h = (");
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.12e ", sigma_h[j_S]);
        }
        fprintf(pFile_output, ") ,\n");

        fprintf(pFile_output, "<h_i> = (");
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.12e ", h_dev_avg[j_S]);
        }
        fprintf(pFile_output, ") ,\n");

        fprintf(pFile_output, "order = (");
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S)
            {
                fprintf(pFile_output, ",");
            }
            fprintf(pFile_output, " %.12e ", order[j_S]);
        }
        fprintf(pFile_output, ") ,\n");

        fprintf(pFile_output, "order_field = %d ,\n", h_order);

        fprintf(pFile_output, "order_random = %d ,\n", r_order);

        fprintf(pFile_output, "Thermalizing-MCS = %ld ,\n", thermal_i);

        fprintf(pFile_output, "Averaging-MCS = %ld ,\n", average_j);

        fprintf(pFile_output, "hysteresis-MCS/step = %ld, \n", hysteresis_MCS);
        
        fprintf(pFile_output, "del_h = range_h x %.12e ,\n", del_h);
        fprintf(pFile_output, "delta_{phi} = %.12e ,\n", del_phi);

        fprintf(pFile_output, "==================================================================================\n");

        // for (i=0; i<whatever; i++)
        // {
        //     fprintf(pFile_output, "%s", column_head[i]);
        // }

        // fprintf(pFile_output, "----------------------------------------------------------------------------------\n");

        
        fclose(pFile_output);
        return 0;
    }

    int print_to_file()
    {


        return 0;
    }

//====================      Avalanche delta(S)               ====================//

    #ifdef enable_CUDA_CODE

    #else
    int ensemble_delta_S_squared_max()
    {
        long int i; 
        int j_S;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            delta_S_squared[j_S] = 0;
        }
        delta_S_max = 0.0;
        
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
        
        delta_S_max = sqrt(delta_S_max);
        
        return 0;
    }

    int ensemble_delta_S_squared()
    {
        long int i; 
        int j_S;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            delta_S_squared[j_S] = 0;
        }
        
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
        
        delta_S_max = sqrt(delta_S_max);
        
        return 0;
    }
    #endif

//====================      Magnetisation                    ====================//

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

    int set_sum_of_moment_m_0()
    {
        int j_S;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_sum[j_S] = 0;
            m_avg[j_S] = 1;
        }

        return 0;
    }

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
            dev_m[index_j_S] = dev_spin_reduce[index_j_S] / (double) sites;
        }
        return; 
    }
    
    __global__ void copy_spin(long int sites)
    {
        int index = threadIdx.x + blockIdx.x*blockDim.x;
        if (index < sites*dim_S)
        {
            dev_spin_reduce[index] = dev_spin_temp[index];
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
    
    int ensemble_m_vec_abs()
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
            m[j_S] = fabs(m[j_S]) / no_of_sites;
        }
        
        return 0;
    }

    int sum_of_moment_m()
    {
        int j_S;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_sum[j_S] += m[j_S];
        }
        
        return 0;
    }

    int average_of_moment_m(double MCS_counter)
    {
        int j_S;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_avg[j_S] = m_sum[j_S] / MCS_counter;
        }
        
        return 0;
    }

    int set_sum_of_moment_m_abs_0()
    {
        m_abs_sum = 0;
        m_abs_avg = 1;

        return 0;
    }

    int sum_of_moment_m_abs()
    {
        int j_S, j_SS, j_L;
        double m_2_persite = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_2_persite += m[j_S] * m[j_S];
        }

        m_abs_sum += sqrt(m_2_persite);

        return 0;
    }

    int average_of_moment_m_abs(double MCS_counter)
    {
        m_abs_avg = m_abs_sum / MCS_counter;

        return 0;
    }

    int set_sum_of_moment_m_2_0()
    {
        m_2_sum = 0;
        m_2_avg = 1;

        return 0;
    }

    int sum_of_moment_m_2()
    {
        int j_S, j_SS, j_L;
        double m_2_persite = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_2_persite += m[j_S] * m[j_S];
        }

        m_2_sum += m_2_persite;

        return 0;
    }

    int average_of_moment_m_2(double MCS_counter)
    {
        m_2_avg = m_2_sum / MCS_counter;

        return 0;
    }

    int set_sum_of_moment_m_4_0()
    {
        m_4_sum = 0;
        m_4_avg = 1;

        return 0;
    }

    int sum_of_moment_m_4()
    {
        int j_S, j_SS, j_L;
        double m_2_persite = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_2_persite += m[j_S] * m[j_S];
        }

        m_4_sum += m_2_persite * m_2_persite;

        return 0;
    }

    int average_of_moment_m_4(double MCS_counter)
    {
        m_4_avg = m_4_sum / MCS_counter;

        return 0;
    }

    int set_sum_of_moment_m_ab_0()
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
    }

    int sum_of_moment_m_ab()
    {
        int j_S, j_SS;
        double m_2_persite = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {    
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                m_ab_sum[j_S*dim_S + j_SS] += m[j_S] * m[j_SS];            
            }
        }

        return 0;
    }

    int average_of_moment_m_ab(double MCS_counter)
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
    }

    int set_sum_of_moment_m_higher_0()
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
    }

    int sum_of_moment_m_higher()
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
    }

    int average_of_moment_m_higher(double MCS_counter)
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
    }

    int set_sum_of_moment_m_all_0()
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
    }

    int sum_of_moment_m_all()
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
    }

    int average_of_moment_m_all(double MCS_counter)
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
    }

    int set_sum_of_moment_m_vec_0()
    {
        int j_S, j_SS;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_2_vec_sum[j_S] = 0;
            m_2_vec_avg[j_S] = 1;
            m_4_vec_sum[j_S] = 0;
            m_4_vec_avg[j_S] = 1;
        }

        return 0;
    }

    int sum_of_moment_m_vec()
    {
        int j_S, j_SS, j_L;
        double m_2_vec_persite = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_2_vec_persite = m[j_S] * m[j_S];
            m_2_vec_sum[j_S] += m_2_vec_persite;
            m_4_vec_sum[j_S] += m_2_vec_persite * m_2_vec_persite;;
        }

        return 0;
    }

    int average_of_moment_m_vec(double MCS_counter)
    {
        int j_S, j_SS, j_L;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_2_vec_avg[j_S] = m_2_vec_sum[j_S] / MCS_counter;
            m_4_vec_avg[j_S] = m_4_vec_sum[j_S] / MCS_counter;
        }
        
        return 0;
    }


//====================      Binder Parameter                 ====================//

    int set_sum_of_moment_B_0()
    {
        set_sum_of_moment_m_2_0();
        set_sum_of_moment_m_4_0();

        return 0;
    }

    int ensemble_B()
    {
        ensemble_m();
        
        return 0;
    }

    int sum_of_moment_B()
    {
        sum_of_moment_m_2();
        sum_of_moment_m_4();
        
        return 0;
    }

    int average_of_moment_B(double MCS_counter)
    {
        average_of_moment_m_2(MCS_counter);
        average_of_moment_m_4(MCS_counter);
        
        B = (1.0 / 2.0) * ( 3.0 - ( m_4_avg / (m_2_avg * m_2_avg) ) );
        
        return 0;
    }

//====================      Susceptibity                     ====================//

    int set_sum_of_moment_X_0()
    {
        set_sum_of_moment_m_abs_0();
        set_sum_of_moment_m_2_0();

        return 0;
    }

    int ensemble_X()
    {
        ensemble_m();

        return 0;
    }

    int sum_of_moment_X()
    {
        sum_of_moment_m_abs();
        sum_of_moment_m_2();
        
        return 0;
    }

    int average_of_moment_X(double MCS_counter)
    {
        average_of_moment_m_abs(MCS_counter);
        average_of_moment_m_2(MCS_counter);
        
        X = (m_2_avg - (m_abs_avg * m_abs_avg)) / T;
        
        return 0;
    }

//====================      Susceptibility tensor            ====================//

    int set_sum_of_moment_X_ab_0()
    {
        set_sum_of_moment_m_0();
        set_sum_of_moment_m_ab_0();

        return 0;
    }

    int ensemble_X_ab()
    {
        ensemble_m();
        
        return 0;
    }

    int sum_of_moment_X_ab()
    {
        sum_of_moment_m();
        sum_of_moment_m_ab();
        
        return 0;
    }

    int average_of_moment_X_ab(double MCS_counter)
    {
        average_of_moment_m(MCS_counter);
        average_of_moment_m_ab(MCS_counter);
        
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

//====================      Energy                           ====================//

    int set_sum_of_moment_E_0()
    {
        E_sum = 0;
        E_avg = 1;

        return 0;
    }

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

    int sum_of_moment_E()
    {
        E_sum += E;
        
        return 0;
    }

    int average_of_moment_E(double MCS_counter)
    {
        E_avg = E_sum / MCS_counter;
        
        return 0;
    }

    int set_sum_of_moment_E_2_0()
    {
        E_2_sum = 0;
        E_2_avg = 1;

        return 0;
    }

    int sum_of_moment_E_2()
    {
        E_2_sum += E * E;
        
        return 0;
    }

    int average_of_moment_E_2(double MCS_counter)
    {
        E_2_avg = E_2_sum / MCS_counter;
        
        return 0;
    }

//====================      Specific Heat                    ====================//

    int set_sum_of_moment_Cv_0()
    {
        set_sum_of_moment_E_0();
        set_sum_of_moment_E_2_0();

        return 0;
    }

    int ensemble_Cv()
    {
        ensemble_E();

        return 0;
    }

    int sum_of_moment_Cv()
    {
        sum_of_moment_E();
        sum_of_moment_E_2();
        
        return 0;
    }

    int average_of_moment_Cv(double MCS_counter)
    {
        average_of_moment_E(MCS_counter);
        average_of_moment_E_2(MCS_counter);
        
        Cv = (E_2_avg - (E_avg * E_avg)) / (T * T);
        
        return 0;
    }

//====================      Helicity                         ====================//

    int set_sum_of_moment_Y_0()
    {
        int j_S, j_SS, j_L;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {       
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    Y_1_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = 0;
                    Y_1_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = 1;
                    Y_2_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = 0;
                    Y_2_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = 1;
                }
            }
        }

        return 0;
    }

    int ensemble_Y()
    {
        long int i;
        int j_L, j_S, j_SS, j_L_j_S_j_SS;

        /* for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = 0;
                    Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = 0;
                }
            }
        } */

        for (j_L_j_S_j_SS=0; j_L_j_S_j_SS < dim_S*dim_S*dim_L; j_L_j_S_j_SS++)
        {
            Y_1[j_L_j_S_j_SS] = 0;
            Y_2[j_L_j_S_j_SS] = 0;
        }

        #ifndef OLD_COMPILER
        #pragma omp parallel for private(j_L, j_S, j_SS) reduction(+:Y_1[:dim_S*dim_S*dim_L], Y_2[:dim_S*dim_S*dim_L])
        for(i=0; i<no_of_sites; i++)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        #ifdef RANDOM_BOND
                        Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += - ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        #else
                        Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += - ( J[j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        #endif
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
                    double Y_1_j_L_j_S_j_SS = 0.0;
                    double Y_2_j_L_j_S_j_SS = 0.0;
                    #pragma omp parallel for private(i) reduction(+:Y_1_j_L_j_S_j_SS, Y_2_j_L_j_S_j_SS)
                    for(i=0; i<no_of_sites; i++)
                    {
                        #ifdef RANDOM_BOND
                        Y_1_j_L_j_S_j_SS += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        Y_1_j_L_j_S_j_SS += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        Y_2_j_L_j_S_j_SS += - ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        Y_2_j_L_j_S_j_SS += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        #else
                        Y_1_j_L_j_S_j_SS += ( J[j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        Y_1_j_L_j_S_j_SS += ( J[j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        Y_2_j_L_j_S_j_SS += - ( J[j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        Y_2_j_L_j_S_j_SS += ( J[j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        #endif
                    }
                    Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_1_j_L_j_S_j_SS;
                    Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_2_j_L_j_S_j_SS;
                }
            }
        }
        #endif

        for (j_L_j_S_j_SS=0; j_L_j_S_j_SS < dim_S*dim_S*dim_L; j_L_j_S_j_SS++)
        {
            Y_1[j_L_j_S_j_SS] = Y_1[j_L_j_S_j_SS] / no_of_sites;
            Y_2[j_L_j_S_j_SS] = (Y_2[j_L_j_S_j_SS] * Y_2[j_L_j_S_j_SS]) / no_of_sites;
        }
        // for (j_S=0; j_S<dim_S; j_S++)
        // {
        //     for (j_SS=0; j_SS<dim_S; j_SS++)
        //     {
        //         for (j_L=0; j_L<dim_L; j_L++)
        //         {
        //             double Y_1_j_L_j_S_j_SS = 0;
        //             double Y_2_j_L_j_S_j_SS = 0;
        //             #pragma omp parallel for reduction(+:Y_1_j_L_j_S_j_SS, Y_2_j_L_j_S_j_SS)
        //             for(i=0; i<no_of_sites; i++)
        //             {
        //                 Y_1_j_L_j_S_j_SS += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
        //                 Y_1_j_L_j_S_j_SS += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
        //                 Y_2_j_L_j_S_j_SS += - ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
        //                 Y_2_j_L_j_S_j_SS += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
        //             }
        //             Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_1_j_L_j_S_j_SS / no_of_sites;
        //             Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = custom_double_pow(Y_2_j_L_j_S_j_SS, 2) / no_of_sites;
        //         }
        //     }
        // }

        /* for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] / no_of_sites;
                    Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = custom_double_pow(Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS], 2) / no_of_sites;
                }
            }
        } */
        
        return 0;
    }

    int sum_of_moment_Y()
    {
        int j_S, j_SS, j_L;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    Y_1_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS];
                    Y_2_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS];
                }
            }
        }
        
        return 0;
    }

    int average_of_moment_Y(double MCS_counter)
    {
        int j_S, j_SS, j_L;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    Y_1_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_1_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] / MCS_counter;
                    Y_2_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_2_sum[dim_S*dim_S*j_L + dim_S*j_S + j_SS] / MCS_counter;
                }
            }
        }
        
        return 0;
    }

//====================      Helicity tensor                  ====================//

    int set_sum_of_moment_Y_ab_mu_0()
    {
        set_sum_of_moment_Y_0();

        return 0;
    }

    int ensemble_Y_ab_mu()
    {
        ensemble_Y();

        return 0;
    }

    int sum_of_moment_Y_ab_mu()
    {
        sum_of_moment_Y();
        
        return 0;
    }

    int average_of_moment_Y_ab_mu(double MCS_counter)
    {
        average_of_moment_Y(MCS_counter);
        
        int j_S, j_SS, j_L;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    Y_ab_mu[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_1_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] - Y_2_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] / T;
                }
            }
        }
        
        return 0;
    }

//====================      All moments                      ====================//

    int set_sum_of_moment_all_0()
    {
        set_sum_of_moment_m_0();
        set_sum_of_moment_m_ab_0();
        set_sum_of_moment_m_abs_0();
        set_sum_of_moment_m_2_0();
        set_sum_of_moment_m_4_0();
        set_sum_of_moment_E_0();
        set_sum_of_moment_E_2_0();
        set_sum_of_moment_Y_0();

        return 0;
    }

    int ensemble_all()
    {
        ensemble_m();
        ensemble_E();
        ensemble_Y();

        return 0;
    }

    int sum_of_moment_all()
    {
        sum_of_moment_m();
        sum_of_moment_m_ab();
        sum_of_moment_m_abs();
        sum_of_moment_m_2();
        sum_of_moment_m_4();
        sum_of_moment_E();
        sum_of_moment_E_2();
        sum_of_moment_Y();
        
        return 0;
    }

    int average_of_moment_all(double MCS_counter)
    {
        average_of_moment_m(MCS_counter);
        average_of_moment_m_ab(MCS_counter);
        average_of_moment_m_abs(MCS_counter);
        average_of_moment_m_2(MCS_counter);
        average_of_moment_m_4(MCS_counter);
        average_of_moment_E(MCS_counter);
        average_of_moment_E_2(MCS_counter);
        average_of_moment_Y(MCS_counter);
        
        int j_S, j_SS, j_L;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                X_ab[j_S*dim_S + j_SS] = (m_ab_avg[j_S*dim_S + j_SS] - m_avg[j_S] * m_avg[j_SS]) / T;
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    Y_ab_mu[dim_S*dim_S*j_L + dim_S*j_S + j_SS] = Y_1_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] - Y_2_avg[dim_S*dim_S*j_L + dim_S*j_S + j_SS] / T;
                }
            }
        }
        
        Cv = (E_2_avg - (E_avg * E_avg)) / (T * T);

        X = (m_2_avg - (m_abs_avg * m_abs_avg)) / T;

        B = (1.0 / 2.0) * ( 3.0 - ( m_4_avg / (m_2_avg * m_2_avg) ) );
        
        
        return 0;
    }

//====================      MonteCarlo-tools                 ====================//

    #ifdef enable_CUDA_CODE
    // __global__ void update_spin_all(long int sites, double* spin_local, double* cutoff_bool)
    __global__ void update_spin_all(long int sites, double* cutoff_bool)
    {
        long int index = threadIdx.x + blockIdx.x*blockDim.x;
        long int stride = blockDim.x*gridDim.x;
        long int xyzi = index;
        
        if (index < sites)
        {
            int j_S;
            for (j_S=0; j_S<dim_S; j_S++)
            {
                // if ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] )
                {
                    cutoff_bool[0] += (double) ( fabs(dev_spin[dim_S*xyzi + j_S] - dev_spin_temp[dim_S*xyzi + j_S]) > dev_CUTOFF_SPIN[0] );
                }
                dev_spin[dim_S*xyzi + j_S] = dev_spin_temp[dim_S*xyzi + j_S];
            }
        }

        // return 0;
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
        
        do
        {
            s_mod=0.0;
            for(j_S=0; j_S<dim_S; j_S=j_S+1)
            {
                spin_local[j_S] = (1.0 - 2.0 * (double)rand_r(&random_seed[cache_size*omp_get_thread_num()])/(double)(RAND_MAX));
                // spin_local[j_S] = (-1.0 + 2.0 * 0.75);
                s_mod = s_mod + spin_local[j_S] * spin_local[j_S];
            }
        }
        while(s_mod >= 1 || s_mod <= limit);
        s_mod = sqrt(s_mod);
        
        // spin_new[0] = spin_new[0] / s_mod;
        // if (spin_old[0] == spin_new[0])
        // {
        //     spin_new[0] = -spin_new[0];
        // }
        // Energy_nu = Energy_nu + field_site[0] * spin_new[0];
        for(j_S=0; j_S<dim_S; j_S++)
        {
            spin_local[j_S] = spin_local[j_S] / s_mod;
            // spin_new[dim_S*xyzi + j_S] = spin_local[j_S];
            // Energy_nu = Energy_nu + field_site[dim_S*xyzi + j_S] * spin_local[j_S];
            Energy_nu = Energy_nu + field_local[j_S] * spin_local[j_S];
        }
        
        return Energy_nu;
    }

//====================      Metropolis                       ====================//

    double update_probability_Metropolis(long int xyzi)
    {
        double update_prob;

        double spin_local[dim_S];
        double field_local[dim_S];
        double E_old = Energy_old(xyzi, spin_local, field_local);
        double E_new = Energy_new(xyzi, spin_local, field_local);
        if (E_new - E_old <= 0)
        {
            update_prob = 1.0;
        }
        else 
        {
            if (T == 0)
            {
                update_prob = 0.0;
            }
            else 
            {
                update_prob = exp(-(E_new-E_old)/T);
            }
        }
        double r = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()])/ (double) RAND_MAX;
        if(r < update_prob)
        {
            update_spin_single(xyzi, spin_local);
            return 1.0;
        }
        
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

                // double r = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()])/ (double) RAND_MAX;
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
        long int xyzi;

        do
        {
            xyzi = rand_r(&random_seed[cache_size*omp_get_thread_num()])%no_of_sites;

            double update_prob = update_probability_Metropolis(xyzi);

            // double r = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()])/ (double) RAND_MAX;
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
        return 0;
    }

    int checkerboard_Metropolis_sweep(long int iter)
    {
        // THIS PART IS PARALLELIZABLE. *modifications may be needed
        static int black_or_white = 0;
        long int i;
        while(iter > 0)
        {
            #pragma omp parallel for 
            for (i=0; i < no_of_black_white_sites[black_or_white]; i++)
            {
                long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*black_or_white + i];

                double update_prob = update_probability_Metropolis(site_index);
            }
            
            black_or_white = !black_or_white;
            iter--;
            // printf("Iteration=%ld\n", iter);
        } 
        
        
        return 0;
    }

//====================      Glauber                          ====================//

    double update_probability_Glauber(long int xyzi)
    {
        double update_prob;
        double spin_local[dim_S];
        double field_local[dim_S];
        double E_old = Energy_old(xyzi, spin_local, field_local);
        double E_new = Energy_new(xyzi, spin_local, field_local);
        if (T == 0)
        {
            if (E_new - E_old > 0)
            {
                update_prob = 0.0;
            }
            else if (E_new - E_old < 0)
            {
                update_prob = 1.0;
            }
            else
            {
                update_prob = 0.5;
            }
        }
        else 
        {
            update_prob = 1/(1+exp((E_new-E_old)/T));
        }
        double r = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()])/ (double) RAND_MAX;
        if(r < update_prob)
        {
            update_spin_single(xyzi, spin_local);
            return 1.0;
        }
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

                // double r = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()])/ (double) RAND_MAX;
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
        long int xyzi;
        do
        {
            xyzi = rand_r(&random_seed[cache_size*omp_get_thread_num()])%no_of_sites;

            double update_prob = update_probability_Glauber(xyzi);

            // double r = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()])/ (double) RAND_MAX;
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
        return 0;
    }

    int checkerboard_Glauber_sweep(long int iter)
    {
        // THIS PART IS PARALLELIZABLE. *modifications may be needed
        static int black_or_white = 0;
        long int i;
        while(iter > 0)
        {
            #pragma omp parallel for 
            for (i=0; i < no_of_black_white_sites[black_or_white]; i++)
            {
                long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*black_or_white + i];

                double update_prob = update_probability_Metropolis(site_index);
            }

            black_or_white = !black_or_white;
            iter--;
        } 
        
        return 0;
    }

//====================      MonteCarlo-Wolff/Cluster         ====================//

    int generate_random_axis()
    {
        int j_S;
        double s_mod = 0.0;
        double limit = 0.01 * dim_S;
        do
        {
            s_mod = 0.0;
            for(j_S=0; j_S<dim_S; j_S=j_S+1)
            {
                reflection_plane[j_S] = (-1.0 + 2.0 * (double)rand_r(&random_seed[cache_size*omp_get_thread_num()])/(double)(RAND_MAX));
                s_mod = s_mod + (reflection_plane[j_S] * reflection_plane[j_S]);
            }
        }
        while(s_mod >= 1 || s_mod <= limit);
        s_mod = sqrt(s_mod);
        
        for(j_S=0; j_S<dim_S; j_S++)
        {
            reflection_plane[j_S] = reflection_plane[j_S] / s_mod;
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
        double energy_site = 0;
        double Si_dot_ref = 0;
        int j_S;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            #ifdef RANDOM_BOND
            energy_site = -(J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*xyzi + j_S] * spin[dim_S*xyzi_nn + j_S];
            #else
            energy_site = -(J[j_L]) * spin[dim_S*xyzi + j_S] * spin[dim_S*xyzi_nn + j_S];
            #endif
        }

        return energy_site;
    }

    double E_bond_new(long int xyzi, int j_L, int k_L, double* __restrict__ spin_local)
    {
        double energy_site = 0;
        double Si_dot_ref = 0;
        int j_S;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            #ifdef RANDOM_BOND
            energy_site = -(J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*xyzi + j_S] * spin_local[j_S];
            #else
            energy_site = -(J[j_L]) * spin[dim_S*xyzi + j_S] * spin_local[j_S];
            #endif
        }

        return energy_site;
    }

    int nucleate_from_site(long int xyzi)
    {

        cluster[xyzi] = 1;
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
                    double delta_E_bond = -E_bond_old(xyzi, j_L, k_L, xyzi_nn);
                    delta_E_bond += E_bond_new(xyzi, j_L, k_L, spin_reflected);
                    if (delta_E_bond < 0)
                    {
                        if (T > 0)
                        {
                            p_bond = 1 - exp(delta_E_bond/T);
                        }
                        else
                        {
                            if (delta_E_bond < 0)
                            {
                                p_bond = 1;
                            }
                        }
                        double r_bond = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()]) / (double) RAND_MAX;
                        /* if (r_bond < p_bond)
                        {
                            nucleate_from_site(xyzi_nn);
                        } */

                        if (p_bond > 0)
                        {
                            double p_site = 1;
                            double delta_E_site = -E_site_old(xyzi_nn);
                            delta_E_site += E_site_new(xyzi_nn, spin_reflected);
                            // if (delta_E_site > 0)
                            // {
                                if (T > 0)
                                {
                                    p_site = exp(-delta_E_site/T);
                                }
                                else
                                {
                                    if (delta_E_site > 0)
                                    {
                                        p_site = 0;
                                    }
                                }
                                double r_site = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()]) / (double) RAND_MAX;
                                if (r_site < p_site*p_bond)
                                {
                                    update_spin_single(xyzi, spin_reflected);
                                    nucleate_from_site(xyzi_nn);
                                }
                            // }
                        }
                    }

                    // double r = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()]) / (double) RAND_MAX;
                    // if (r < p)
                    // {
                    //     nucleate_from_site(xyzi_nn);
                    // }
                }
            }
        }

        return 0;
    }

    int set_cluster_s(int s)
    {
        long int i;
        for (i=0; i<no_of_sites; i++)
        {
            cluster[i] = s;
        }
        return 0;
    }

    int random_Wolff_sweep(long int iter)
    {
        long int xyzi;
        int i, j_S;
        for (i=0; i<iter; i++)
        {
            set_cluster_s(0);
            
            // do
            // {
                generate_random_axis();
                xyzi = rand_r(&random_seed[cache_size*omp_get_thread_num()]) % no_of_sites;
                double delta_E_site = -E_site_old(xyzi);
                double spin_reflected[dim_S];
                transform_spin(xyzi, spin_reflected);
                delta_E_site += E_site_new(xyzi, spin_reflected);
                if (delta_E_site <= 0)
                {
                    update_spin_single(xyzi, spin_reflected);
                    nucleate_from_site(xyzi);
                }
                else
                {
                    double r = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()]) / (double) RAND_MAX;
                    if (r<exp(-delta_E_site/T))
                    {
                        update_spin_single(xyzi, spin_reflected);
                        nucleate_from_site(xyzi);
                    }
                }
            // }
            // while (cluster[xyzi] != 1);
        }

        return 0;
    }

//====================      MonteCarlo-Sweep                 ====================//

    int Monte_Carlo_Sweep(long int sweeps)
    {
        if (Gl_Me_Wo == 0)
        {
            if (Ch_Ra_Li == 0)
            {
                checkerboard_Glauber_sweep(2*sweeps);
            }
            else
            {
                if (Ch_Ra_Li == 1)
                {
                    random_Glauber_sweep(no_of_sites*sweeps);
                }
                else
                {
                    linear_Glauber_sweep(sweeps);
                }
            }
        }
        else
        {
            if (Gl_Me_Wo == 1)
            {
                if (Ch_Ra_Li == 0)
                {
                    checkerboard_Metropolis_sweep(2*sweeps);
                }
                else
                {
                    if (Ch_Ra_Li == 1)
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
                random_Wolff_sweep(sweeps);
            }
        }
        return 0;
    }

//====================      T!=0                             ====================//

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

    int thermalizing_iteration(long int thermal_iter)
    {
        int j_S;
        printf("Thermalizing iterations... \n");

        /* while(thermal_iter)
        {
            // linear_Glauber_sweep();
            // checkerboard_Glauber_sweep(2);
            // random_Glauber_sweep(no_of_sites);
            // checkerboard_Glauber_sweep(2);
            
            Monte_Carlo_Sweep(1);
            // random_Wolff_sweep(1);
            thermal_iter = thermal_iter - 1;
            printf("Iteration=%ld\n", thermal_iter);
        } */

        Monte_Carlo_Sweep(thermal_iter);
        printf("Done.\n");
        // ensemble_all();
        ensemble_E();
        ensemble_m();
        printf("Thermalized Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);

        return 0;
    }

    int averaging_iteration(long int average_iter)
    {
        double MCS_counter = 0;
        int j_S, j_SS, j_L;
        
        // set_sum_of_moment_Y_ab_mu_0();
        
        set_sum_of_moment_m_0();
        set_sum_of_moment_m_higher_0();
        // set_sum_of_moment_m_vec_0();
        // set_sum_of_moment_m_abs_0();
        // set_sum_of_moment_E_0();

        printf("Averaging iterations... h=%lf", h[0]);
        for (j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", h[j_S]);
        }
        printf("\n");

        while(average_iter)
        {
            Monte_Carlo_Sweep(sampling_inter-genrand64_int64()%sampling_inter);
            // random_Wolff_sweep(1);
            ensemble_m();
            // ensemble_m_vec_abs();
            // ensemble_B();
            // ensemble_E();
            // ensemble_Y_ab_mu();
            // sum_of_moment_m_vec();
            sum_of_moment_m();
            sum_of_moment_m_higher();
            // sum_of_moment_B();
            // sum_of_moment_m_abs();
            // sum_of_moment_E();
            // sum_of_moment_Y_ab_mu();
            MCS_counter = MCS_counter + 1;
            
            average_iter = average_iter - 1;
        }
        printf("Done.\n");

        average_of_moment_m(MCS_counter);
        average_of_moment_m_higher(MCS_counter);
        // average_of_moment_m_vec(MCS_counter);
        // average_of_moment_B(MCS_counter);
        // average_of_moment_m_abs(MCS_counter);
        // average_of_moment_E(MCS_counter);
        // average_of_moment_Y_ab_mu(MCS_counter);

        printf("Final: Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        // printf("), Energy = %lf \n", E);
        printf("), Energy = %lf \n", E);

        printf("<M> = (%lf", m_avg[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m_avg[j_S]);
        }
        printf("), <E> = %lf \n", E_avg);
        printf(", Binder = %lf \n", B);
        /* 
        for (j_S=0; j_S<dim_S; j_S++)
        {
            for (j_SS=0; j_SS<dim_S; j_SS++)
            {
                for (j_L=0; j_L<dim_L; j_L++)
                {
                    printf("Y[%d,%d][%d]=%lf\t \n", j_S, j_SS, j_L, Y_ab_mu[dim_S*dim_S*j_L + dim_S*j_S + j_SS]);
                }
            }
        } */

        // printf("<M> = %lf, <|M|> = %lf, <E> = %lf, Cv = %lf, X = %lf, X_abs = %lf, B = %lf .\n", m_avg, m_abs_avg, E_avg, Cv, X, X_abs, B);
        
        return 0;
    }

    int evolution_at_T(int repeat_for_same_T)
    {
        // repeat with different initial configurations
        int j_S, j_SS, j_L;

        printf("\norder = ({%lf", order[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", order[j_S]);
        }
        printf("}, %d, %d)\n", h_order, r_order);

        // ensemble_all();
        ensemble_E();
        ensemble_m();

        printf("Initial Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_evo(t)_at_T=%lf_%c_%c_", dim_S, dim_L, T, G_M_W[Gl_Me_Wo], C_R_L[Ch_Ra_Li]);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            // pos += sprintf(pos, "_(%lf,%lf)_{", Temp_min, Temp_max);
            // pos += sprintf(pos, "_{");
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", J[j_L]);
            // }
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_L = 0 ; j_L != dim_L ; j_L++) 
            // {
            //     if (j_L)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", sigma_J[j_L]);
            // }
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
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h[j_S]);
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
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld_%ld.dat", h_order, r_order, thermal_i, average_j);
        }

        // column labels and parameters
        print_header_column(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        {
            fprintf(pFile_1, "step\t");
            fprintf(pFile_1, "<|m|>\t");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t", j_S);
            }
            fprintf(pFile_1, "<E>\t");
            
            fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
        }
        fclose(pFile_1);

        int i;
        for (i=0; i<repeat_for_same_T; i++)
        {
            thermalizing_iteration(thermal_i);
            averaging_iteration(average_j);
            pFile_1 = fopen(output_file_0, "a");
            fprintf(pFile_1, "%d\t", i);
            fprintf(pFile_1, "%.12e\t", m_abs_avg);
            
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m_avg[j_S]);
            }
            
            fprintf(pFile_1, "%.12e\t", E_avg);

            fprintf(pFile_1, "\n");
            fclose(pFile_1);
        }
        printf("------------------------\n");

        return 0;
    }

    int evolution_at_T_h()
    {
        // repeat with different initial configurations
        int j_S, j_SS, j_L;
        
        // print statements:
        {
            printf("\norder = ({%lf", order[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", order[j_S]);
            }
            printf("}, %d, %d)\n", h_order, r_order);
        }
        ensemble_all();
        // ensemble_E();
        // ensemble_m();

        printf("Initial Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_evo(t)_at_T=%lf_%c_%c_", dim_S, dim_L, T, G_M_W[Gl_Me_Wo], C_R_L[Ch_Ra_Li]);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_(%lf,%lf)_{", Temp_min, Temp_max);
            pos += sprintf(pos, "_{");
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
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h[j_S]);
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
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld_%ld.dat", h_order, r_order, thermal_i, average_j);
        }
        // column labels and parameters
        print_header_column(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        {
            fprintf(pFile_1, "<|m|>\t");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t", j_S);
            }
            fprintf(pFile_1, "<E>\t");
            fprintf(pFile_1, "T\t");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "h[%d]\t", j_S);
            }

            fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
        }
        fclose(pFile_1);

        thermalizing_iteration(thermal_i);
        averaging_iteration(average_j);
        pFile_1 = fopen(output_file_0, "a");
        fprintf(pFile_1, "%.12e\t", m_abs_avg);
        
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", m_avg[j_S]);
        }
        
        fprintf(pFile_1, "%.12e\t", E_avg);
        fprintf(pFile_1, "%.12e\t", T);

        for (j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", h[j_S]);
        }
        fprintf(pFile_1, "\n");
        fclose(pFile_1);

        printf("------------------------\n");

        return 0;
    }

    int initialize_spin_and_evolve_at_T()
    {

        // repeat with different initial configurations
        int j_S, j_SS, j_L;

        initialize_spin_config();

        printf("\norder = ({%lf", order[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", order[j_S]);
        }
        printf("}, %d, %d)\n", h_order, r_order);

        ensemble_all();
        // ensemble_E();
        // ensemble_m();

        printf("Initial Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);

        thermalizing_iteration(thermal_i);
        averaging_iteration(average_j);

        printf("------------------------\n");

        return 0;
    }

    int evo_diff_ini_config_temp()
    {
        int j_S, j_L, j_SS;

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

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_Temp_evo_%c_%c_", dim_S, dim_L, G_M_W[Gl_Me_Wo], C_R_L[Ch_Ra_Li]);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_(%lf,%lf)_{", Temp_min, Temp_max);
            pos += sprintf(pos, "_{");
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
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h[j_S]);
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
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld_%ld.dat", h_order, r_order, thermal_i, average_j);
        }
        // column labels and parameters
        print_header_column(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        {
            fprintf(pFile_1, "T\t");
            fprintf(pFile_1, "<|m|>\t");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t", j_S);
            }
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "<Y[%d,%d][%d]>\t", j_S, j_SS, j_L);
                    }
                }
            } */
            fprintf(pFile_1, "<E>\t");
            fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
        }
        fclose(pFile_1);
        
        for (T=Temp_min; T<=Temp_max; T=T+delta_T)
        {
            printf("\nT=%lf\t ", T);
            initialize_spin_and_evolve_at_T(); 

            pFile_1 = fopen(output_file_0, "a");
            fprintf(pFile_1, "%.12e\t", T);
            fprintf(pFile_1, "%.12e\t", m_abs_avg);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m_avg[j_S]);
            }
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "%.12e\t", Y_ab_mu[dim_S*dim_S*j_L + dim_S*j_S + j_SS]);
                    }
                }
            } */
            fprintf(pFile_1, "%.12e\t", E_avg);

            fprintf(pFile_1, "\n");
            fclose(pFile_1);
        }
        
        
        return 0;
    }

    int cooling_protocol(char output_file_name[])
    {
        #ifdef _OPENMP
        #ifdef BUNDLE
            omp_set_num_threads(BUNDLE);
        #else
            omp_set_num_threads(num_of_threads);
        #endif
        #endif
        
        // #ifdef _OPENMP
        // if (num_of_threads<=16)
        // {
        //     omp_set_num_threads(num_of_threads);
        // }
        // else 
        // {
        //     if (num_of_threads<=24)
        //     {
        //         omp_set_num_threads(16);
        //     }
        //     else
        //     {
        //         omp_set_num_threads(num_of_threads-8);
        //     }
        // }
        // #endif
        
        int j_S, j_SS, j_L;

        // ensemble_all();
        ensemble_E();
        ensemble_m();

        printf("Initial Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);
        printf("Cooling... ");

        pFile_1 = fopen(output_file_name, "a");
        fprintf(pFile_1, "Cooling... \n");
        fclose(pFile_1);

        for (T=Temp_max; T>Temp_min; T=T-delta_T)
        {
            printf("\nT=%lf\t ", T);
            
            thermalizing_iteration(thermal_i);
            averaging_iteration(average_j);

            pFile_1 = fopen(output_file_name, "a");

            fprintf(pFile_1, "%.12e\t", T);
            // fprintf(pFile_1, "%.12e\t", m_abs_avg);

            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m_avg[j_S]);
            } 

            // for(j_S=0; j_S<dim_S; j_S++)
            // {
            //     fprintf(pFile_1, "%.12e\t", m_2_vec_avg[j_S]);
            //     fprintf(pFile_1, "%.12e\t", m_4_vec_avg[j_S]);
            // } 
            
            for(j_S=0; j_S<dim_S; j_S++)
            {
                for(j_SS=0; j_SS<dim_S; j_SS++)
                {
                    fprintf(pFile_1, "%.12e\t", m_ab_avg[j_S*dim_S+j_SS]);
                }
            }
            fprintf(pFile_1, "%.12e\t", m_2_avg);
            fprintf(pFile_1, "%.12e\t", m_4_avg);

            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "%.12e\t", Y_ab_mu[dim_S*dim_S*j_L + dim_S*j_S + j_SS]);
                    }
                }
            } */

            // fprintf(pFile_1, "%.12e\t", E_avg);    
            fprintf(pFile_1, "\n");
            fclose(pFile_1);
        }

        // ensemble_all();
        ensemble_E();
        ensemble_m();

        printf("Final Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);
        printf("------------------------\n");
        
        
        return 0;
    }

    int heating_protocol(char output_file_name[])
    {
        #ifdef _OPENMP
        #ifdef BUNDLE
            omp_set_num_threads(BUNDLE);
        #else
            omp_set_num_threads(num_of_threads);
        #endif
        #endif
        // #ifdef _OPENMP
        // if (num_of_threads<=16)
        // {
        //     omp_set_num_threads(num_of_threads);
        // }
        // else 
        // {
        //     if (num_of_threads<=24)
        //     {
        //         omp_set_num_threads(16);
        //     }
        //     else
        //     {
        //         omp_set_num_threads(num_of_threads-8);
        //     }
        // }
        // #endif

        int j_S, j_SS, j_L;

        // ensemble_all();
        ensemble_E();
        ensemble_m();

        printf("Initial Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);
        printf("Heating... ");

        pFile_1 = fopen(output_file_name, "a");
        fprintf(pFile_1, "Heating... \n");
        fclose(pFile_1);
        
        for (T=Temp_min; T<=Temp_max; T=T+delta_T)
        {
            printf("\nT=%lf\t ", T);
            
            thermalizing_iteration(thermal_i);
            averaging_iteration(average_j);

            pFile_1 = fopen(output_file_name, "a");

            fprintf(pFile_1, "%.12e\t", T);
            fprintf(pFile_1, "%.12e\t", m_abs_avg);

            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m_avg[j_S]);
            }
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "%.12e\t", Y_ab_mu[dim_S*dim_S*j_L + dim_S*j_S + j_SS]);
                    }
                }
            } */
            fprintf(pFile_1, "%.12e\t", E_avg);
            fprintf(pFile_1, "\n");
            fclose(pFile_1);
        }

        // ensemble_all();
        ensemble_E();
        ensemble_m();

        printf("Final Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);
        printf("------------------------\n");
        
        return 0;
    }

    int zfc_zfh_or_both(int c_h_ch_hc)
    {
        int j_S, j_SS, j_L;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = 0;
        }
        
        if (c_h_ch_hc == 0 || c_h_ch_hc == 2)
        {
            T = Temp_max;
        }
        if (c_h_ch_hc == 1 || c_h_ch_hc == 3)
        {
            T = Temp_min;
        }
        load_spin_config("");
        // print statements:
        {
            printf("order = ({%lf", order[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", order[j_S]);
            }
            printf("}, %d, %d)\n", h_order, r_order);

            printf("h = {%lf", h[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", h[j_S]);
            }
            printf("}\n");

            printf("J = {%lf", J[0]);
            for(j_L=1; j_L<dim_L; j_L++)
            {
                printf(",%lf", J[j_L]);
            }
            printf("}\n");
        }

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_ZFC-ZFH_%c_%c_", dim_S, dim_L, G_M_W[Gl_Me_Wo], C_R_L[Ch_Ra_Li]);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_(%lf,%lf)-[%lf]", Temp_min, Temp_max, delta_T);
            /* pos += sprintf(pos, "_{");
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}");     */
            /* pos += sprintf(pos, "_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            } */
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
            /* pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h[j_S]);
            }
            pos += sprintf(pos, "}"); */    
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
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld_%ld.dat", h_order, r_order, thermal_i, average_j);
        }
        // column labels and parameters
        print_header_column(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        {
            fprintf(pFile_1, "T\t|m|\t");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t", j_S);
            } 
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "<Y[%d,%d][%d]>\t", j_S, j_SS, j_L);
                    }
                }
            } */
            fprintf(pFile_1, "<E>\t");
            fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
        }
        fclose(pFile_1);

        if (c_h_ch_hc == 0 || c_h_ch_hc == 2)
        {
            cooling_protocol(output_file_0);
        }
        if (c_h_ch_hc == 1 || c_h_ch_hc == 2 || c_h_ch_hc == 3)
        {
            heating_protocol(output_file_0);
        }
        if (c_h_ch_hc == 3)
        {
            cooling_protocol(output_file_0);
        }
        
        return 0;
    }

    int hysteresis_average(long int hysteresis_iter)
    {
        int j_S;
        double MCS_counter = 0;
        
        set_sum_of_moment_m_0();
        set_sum_of_moment_E_0();
        
        // printf("Averaging iterations... h=%lf\n", h);

        while(hysteresis_iter)
        {  

            Monte_Carlo_Sweep(1);

            // ensemble_all();
            ensemble_m();
            ensemble_E();
            
            sum_of_moment_m();
            sum_of_moment_E();
            MCS_counter = MCS_counter + 1;
            
            hysteresis_iter = hysteresis_iter - 1;
        }
        // printf("Done.\n");
        average_of_moment_m(MCS_counter);
        average_of_moment_E(MCS_counter);
        
        // printf("Final: Magnetisation = %lf, Energy = %lf \n", m, E);
        // printf("<M> = %lf, <E> = %lf \n", m_avg, E_avg);
        
        return 0;
    }

    int hysteresis_protocol(int jj_S, double order_start)
    {
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
        printf("\n%lf", m[0]);
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
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hysteresis_%c_%c_", dim_S, dim_L, G_M_W[Gl_Me_Wo], C_R_L[Ch_Ra_Li]);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
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
                    pos += sprintf(pos, "(%lf,%lf)-[%lf]", h_start, h_end, delta_h);
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
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld.dat", h_order, r_order, hysteresis_MCS);
        }
        // column labels and parameters
        print_header_column(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        {
            fprintf(pFile_1, "h[%d]\t", jj_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t", j_S);
            }
            fprintf(pFile_1, "<E>\t");
            fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
        }
        fclose(pFile_1);
        int i;
        for (i=0; i<hysteresis_repeat; i=i+1)
        {
            printf("h = %lf --> %lf\t ", h_start, h_end);
            for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
            {
                hysteresis_average(hysteresis_MCS);

                pFile_1 = fopen(output_file_0, "a");
                fprintf(pFile_1, "%.12e\t", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m_avg[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", E_avg);

                fprintf(pFile_1, "\n");
                fclose(pFile_1);
            }
            printf("..(%d) \nh = %lf <-- %lf\t ", i+1, h_start, h_end);
            for (h[jj_S] = h_end; order[jj_S] * h[jj_S] <= order[jj_S] * h_start; h[jj_S] = h[jj_S] + order[jj_S] * delta_h)
            {
                hysteresis_average(hysteresis_MCS);

                pFile_1 = fopen(output_file_0, "a");
                fprintf(pFile_1, "%.12e\t", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m_avg[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", E_avg);

                fprintf(pFile_1, "\n");
                fclose(pFile_1);
            }
            printf("..(%d) \n", i+1);
            h[jj_S] = 0;
        }
        
        // delta_h = (2*order[jj_S]-1)*delta_h;
        fclose(pFile_1);
        printf("Finished. \n");
        return 0;
    }

//====================      RFIM ZTNE                        ====================//

    long int find_extreme(double s, long int remaining_sites)
    {
        h[0] = 0;
        long int i_1 = 0;
        while (spin[dim_S*i_1 + 0] == -s)
        {
            i_1++;
        }
        double field_local[dim_S];
        double spin_local[dim_S];
        double E_1 = Energy_old(i_1, spin_local, field_local);
        double E_2;
        remaining_sites--;
        long int i_2 = i_1;
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
                i_1 = i_2;
                E_1 = E_2;
            }
        }
        h[0] = s*E_1;
        return i_1;
    }

    long int flip_unstable(long int nucleation_site, long int remaining_sites)
    {
        int j_SS = 0, j_L, k_L;
        long int next_site = nucleation_site;
        spin[dim_S*nucleation_site + j_SS] = -spin[dim_S*nucleation_site + j_SS];
        long int i_1=0, i_2=1;
        next_in_queue[i_1] = nucleation_site;
        double field_local[dim_S];
        double spin_local[dim_S];
        do
        {
            for (j_L=0; j_L<dim_L; j_L++)
            {
                for (k_L=0; k_L<2; k_L++)
                {
                    next_site = N_N_I[2*dim_L*next_in_queue[i_1] + 2*j_L + k_L];
                    if (spin[dim_S*next_site + j_SS] != -order[0])
                    {
                        if (Energy_old(next_site, spin_local, field_local)>=0)
                        {
                            next_in_queue[i_2] = next_site;
                            i_2++;
                            spin[dim_S*next_site + j_SS] = -spin[dim_S*next_site + j_SS];
                            
                            // remaining_sites = flip_unstable(next_site, remaining_sites);
                        }
                    }
                }
            }
            i_1++;
        }
        while(i_1 != i_2);

        return remaining_sites - i_2;
    }

    int zero_temp_RFIM_hysteresis()
    {
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
        int jj_S=0, j_S, j_L;

        printf("\nztne RFIM looping  at T=%lf.. \n",  T);

        ensemble_m();
        ensemble_E();
        
        // print statements:
        {
            printf("\n%lf", m[0]);
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
                pos += sprintf(pos, "%d", lattice_size[j_L]);
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
                    pos += sprintf(pos, "(%lf,%lf)", h_i_max, h_i_min);
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
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}.dat");
        }
        // column labels and parameters
        print_header_column(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        {
            fprintf(pFile_1, "h[%d]\t", jj_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t", j_S);
            }
            fprintf(pFile_1, "<E>\t");
            fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
        }
        
        long int nucleation_site;

        long int remaining_sites = no_of_sites;
        ensemble_m();
        ensemble_E();
        fprintf(pFile_1, "%.12e\t", h[jj_S]);
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", m[j_S]);
        }
        fprintf(pFile_1, "%.12e\t", E);

        fprintf(pFile_1, "\n");
        
        while (remaining_sites)
        {
            nucleation_site = find_extreme(order[0], remaining_sites);

            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%.12e\t", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\t", E);

            fprintf(pFile_1, "\n");
            
            remaining_sites = flip_unstable(nucleation_site, remaining_sites);
            printf("h=%lf, ", h[0]);
            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%.12e\t", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\t", E);

            fprintf(pFile_1, "\n");
        }

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
        remaining_sites = no_of_sites;
        printf("\n%lf", m[0]);
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

        ensemble_m();
        ensemble_E();
        fprintf(pFile_1, "%.12e\t", h[jj_S]);
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", m[j_S]);
        }
        fprintf(pFile_1, "%.12e\t", E);

        fprintf(pFile_1, "\n");

        while (remaining_sites)
        {
            nucleation_site = find_extreme(order[0], remaining_sites);

            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%.12e\t", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\t", E);

            fprintf(pFile_1, "\n");
            
            remaining_sites = flip_unstable(nucleation_site, remaining_sites);
            printf("h=%lf, ", h[0]);
            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%.12e\t", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\t", E);

            fprintf(pFile_1, "\n");
        }
        printf("\n----------\nDone.\n");

        fclose(pFile_1);
        return 0;
    }

    int zero_temp_RFIM_ringdown()
    {
        // call ztne RFIM here.
        
        double delta_m = 0.1;
        
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
        int jj_S=0, j_S, j_L;

        printf("\nztne RFIM looping  at T=%lf.. \n",  T);
        
        ensemble_m();
        ensemble_E();

        // print statements:
        {
            printf("\n%lf", m[0]);
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
                pos += sprintf(pos, "%d", lattice_size[j_L]);
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
                    pos += sprintf(pos, "(%lf,%lf)", h_i_max, h_i_min);
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
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}.dat");
        }
        // column labels and parameters
        print_header_column(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        {
            fprintf(pFile_1, "h[%d]\t", jj_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t", j_S);
            }
            fprintf(pFile_1, "<E>\t");
            fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
        }

        long int remaining_sites = 0;
        long int nucleation_site;
        float M_compare;
        for (M_compare=1.0; M_compare>0.0; M_compare -= delta_m)
        {
            // Put modified ztne RFIM hysteresis code here.
            
            order[0] = 1;
            remaining_sites = no_of_sites - remaining_sites;
            ensemble_m();
            ensemble_E();
            
            fprintf(pFile_1, "%.12e\t", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\t", E);
            
            fprintf(pFile_1, "\n");

            double old_h=0.0, new_h=0.0;
            

            while (old_h == new_h || M_compare > -m[0])
            {
                nucleation_site = find_extreme(order[0], remaining_sites);

                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%.12e\t", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", E);

                fprintf(pFile_1, "\n");
                
                remaining_sites = flip_unstable(nucleation_site, remaining_sites);
                printf("h=%lf, ", h[0]);
                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%.12e\t", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", E);
                old_h = h[0];
                find_extreme(order[0], remaining_sites);
                new_h = h[0];

                fprintf(pFile_1, "\n");
            }

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
            remaining_sites = no_of_sites - remaining_sites;
            printf("\n%lf", m[0]);
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

            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%.12e\t", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\t", E);

            fprintf(pFile_1, "\n");

            old_h = 0.0, new_h = 0.0;

            while (old_h == new_h || M_compare > m[0])
            {
                nucleation_site = find_extreme(order[0], remaining_sites);

                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%.12e\t", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", E);

                fprintf(pFile_1, "\n");
                
                remaining_sites = flip_unstable(nucleation_site, remaining_sites);
                printf("h=%lf, ", h[0]);
                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%.12e\t", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", E);
                old_h = h[0];
                find_extreme(order[0], remaining_sites);
                new_h = h[0];

                fprintf(pFile_1, "\n");
            }
            printf("\n----------\nDone.\n");
        }

        fclose(pFile_1);
        return 0;
    }

    int zero_temp_RFIM_return_point_memory()
    {
        // zero_temp_RFIM_hysteresis();

        double delta_m[] = { 0.2, 0.1 , 0.05, 0.025 };
        double m_start = 0.0;
        int depth_of_subloop = sizeof(delta_m) / sizeof(delta_m[0]);
        double *h_ext = (double *)malloc((depth_of_subloop+1)*sizeof(double));
        double *mag_rpm = (double *)malloc((depth_of_subloop+1)*sizeof(double));
        
        T = 0;
        h[0] = h_i_max;
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
        int jj_S=0, j_S, j_L;

        printf("\nztne RFIM looping  at T=%lf.. \n",  T);
        
        ensemble_m();
        ensemble_E();
        
        // print statements:
        {
            printf("\n%lf", m[0]);
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
                pos += sprintf(pos, "%d", lattice_size[j_L]);
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
                    pos += sprintf(pos, "(%lf,%lf)", h_i_max, h_i_min);
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
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}.dat");
        }
        // column labels and parameters
        print_header_column(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        {
            fprintf(pFile_1, "h[%d]\t", jj_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t", j_S);
            }
            fprintf(pFile_1, "<E>\t");
            fprintf(pFile_1, "\tRPM_error");
            fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
        }

        long int remaining_sites = 0;
        long int nucleation_site;
        float M_compare;
        int i, one = 1;
        double old_h = 0.0, new_h = 0.0;

        order[0] = 1;
        remaining_sites = no_of_sites - remaining_sites;
        ensemble_m();
        ensemble_E();
        
        fprintf(pFile_1, "%.12e\t", h[jj_S]);
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", m[j_S]);
        }
        fprintf(pFile_1, "%.12e\t", E);
        
        fprintf(pFile_1, "\n");

        while (old_h == new_h || m_start < m[jj_S])
        {
            nucleation_site = find_extreme(order[0], remaining_sites);

            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%.12e\t", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\t", E);

            fprintf(pFile_1, "\n");
            
            remaining_sites = flip_unstable(nucleation_site, remaining_sites);
            // printf("h=%lf, ", h[0]);
            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%.12e\t", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\t", E);
            old_h = h[0];
            find_extreme(order[0], remaining_sites);
            new_h = h[0];

            fprintf(pFile_1, "\n");
        }
        mag_rpm[0] = m[jj_S];
        h_ext[0] = old_h;
        printf("mag_rpm[0]=%lf, h_ext[0]=%lf \n", mag_rpm[0], h_ext[0]);

        for (i=0; i<depth_of_subloop; i++)
        {
            order[0] = - order[0];
            remaining_sites = no_of_sites - remaining_sites;
            ensemble_m();
            ensemble_E();
            
            fprintf(pFile_1, "%.12e\t", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\t", E);
            
            fprintf(pFile_1, "\n");
            while ( old_h == new_h || delta_m[i] > fabs( mag_rpm[i] - m[jj_S] ) )
            {
                nucleation_site = find_extreme(order[0], remaining_sites);

                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%.12e\t", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", E);

                fprintf(pFile_1, "\n");
                
                remaining_sites = flip_unstable(nucleation_site, remaining_sites);
                // printf("h=%lf, ", h[0]);
                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%.12e\t", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", E);
                old_h = h[0];
                find_extreme(order[0], remaining_sites);
                new_h = h[0];

                fprintf(pFile_1, "\n");
            }
            mag_rpm[i+1] = m[jj_S];
            h_ext[i+1] = old_h;
            printf("mag_rpm[%d]=%lf, h_ext[%d]=%lf \n", i+1, mag_rpm[i+1], i+1, h_ext[i+1]);
        }

        for (i=depth_of_subloop-1; i>=0; i--)
        {
            order[0] = - order[0];
            remaining_sites = no_of_sites - remaining_sites;
            ensemble_m();
            ensemble_E();
            
            fprintf(pFile_1, "%.12e\t", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\t", E);
            
            fprintf(pFile_1, "\n");

            while ( old_h == new_h || !( ( h_ext[i] >= old_h && h_ext[i] < new_h ) || ( h_ext[i] <= old_h && h_ext[i] > new_h ) ) )
            {
                nucleation_site = find_extreme(order[0], remaining_sites);

                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%.12e\t", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", E);

                fprintf(pFile_1, "\n");
                
                remaining_sites = flip_unstable(nucleation_site, remaining_sites);
                // printf("h=%lf, ", h[0]);
                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%.12e\t", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", E);
                old_h = h[0];
                find_extreme(order[0], remaining_sites);
                new_h = h[0];

                fprintf(pFile_1, "\n");
            }

            
            {
                fprintf(pFile_1, "%.12e\t", old_h);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", E);

                fprintf(pFile_1, "%.12e\t", (mag_rpm[i] - m[jj_S]) );

                fprintf(pFile_1, "\n");

                printf("m[0]=%lf, h_ext[%d]=%lf \n", m[jj_S], i, h_ext[i]);
            }
        }
        fclose(pFile_1);

        free(h_ext);
        free(mag_rpm);
        return 0;
    }

//====================      RFXY ZTNE                        ====================//

    #ifdef enable_CUDA_CODE
    // __global__ void Energy_minimum_old_XY(long int sites, double* spin_local)
    __global__ void Energy_minimum_old_XY(long int sites)
    {
        long int index = threadIdx.x + blockIdx.x*blockDim.x;
        long int stride = blockDim.x*gridDim.x;
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
            // // else
            
            // if(Energy_min>0)
            {
                Energy_min = -sqrt(Energy_min);
                double energy_bool = (double)(Energy_min>=0);
                Energy_min += energy_bool;
                // Energy_min = ~(Energy_min); // unary bitflip
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    dev_spin_temp[dim_S*xyzi + j_S] = (field_local[j_S] / Energy_min) + energy_bool * dev_spin_temp[dim_S*xyzi + j_S];
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
    double update_to_minimum_checkerboard(long int xyzi, double* __restrict__ spin_local)
    {
        int j_S;
        
        Energy_minimum_new_XY(xyzi, spin_local);

        double spin_diff_abs = 0.0;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            spin_diff_abs += fabs(spin[dim_S*xyzi + j_S] - spin_local[j_S]);
        }
        update_spin_single(xyzi, spin_local);
        return spin_diff_abs;
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
    double find_change()
    {
        static int print_first = 0;
        if (print_first == 0)
        {
            printf("\n Using find change max.. \n");
            print_first = !print_first;
        }
        double cutoff_local;
        double* dev_cutoff_local;
        // double* dev_spin_temp;
        cudaMalloc(&dev_cutoff_local, sizeof(double));
        // cudaMallocManaged(&dev_spin_temp, dim_S*no_of_sites*sizeof(double));
        long int site_i;
        static int black_or_white = 0;

        // if (update_all_or_checker == 0)
        #ifdef UPDATE_ALL_NON_EQ
        {
            cutoff_local = 0.0;
            // cudaMemcpy(dev_cutoff_local, &cutoff_local, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemset(dev_cutoff_local, 0, sizeof(double));
            
            Energy_minimum_old_XY<<< no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites);
            // Energy_minimum_old_XY<<< 1, no_of_sites >>>(no_of_sites);
            cudaDeviceSynchronize();

            update_spin_all<<< no_of_sites/gpu_threads + 1, gpu_threads >>>(no_of_sites, dev_cutoff_local);
            // update_spin_all<<< 1, no_of_sites >>>(no_of_sites, dev_cutoff_local);
            cudaDeviceSynchronize();

            cudaMemcpy(&cutoff_local, dev_cutoff_local, sizeof(double), cudaMemcpyDeviceToHost);
        }
        #endif
        // else
        #ifdef UPDATE_CHKR_NON_EQ
        {
            // #pragma omp parallel 
            // {
            //     cutoff_local = 0.0;

            //     #pragma omp for reduction(+:cutoff_local)
            //     for (site_i=0; site_i<no_of_black_white_sites[black_or_white]; site_i++)
            //     {
                    long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*black_or_white + site_i];
                    double spin_local[dim_S];

                    cutoff_local += (double) ( update_to_minimum_checkerboard(site_index, spin_local) > CUTOFF_SPIN );
                    
                // }

                // #pragma omp for reduction(+:cutoff_local)
                // for (site_i=0; site_i<no_of_black_white_sites[!black_or_white]; site_i++)
                // {
                    long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*(!black_or_white) + site_i];
                    double spin_local[dim_S];
                    
                    cutoff_local += (double) ( update_to_minimum_checkerboard(site_index, spin_local) > CUTOFF_SPIN );
                    
                // }            
            // }
        }
        #endif
        cudaFree(dev_cutoff_local);
        // cudaFree(dev_spin_temp);
        return cutoff_local;
    }
    #else
    double find_change()
    {
        static int print_first = 0;
        if (print_first == 0)
        {
            printf("\n Using find change max.. \n");
            print_first = !print_first;
        }
        double cutoff_local;
        
        long int site_i;
        static int black_or_white = 0;

        // if (update_all_or_checker == 0)
        #ifdef UPDATE_ALL_NON_EQ
        {
            cutoff_local = 0.0;

            #pragma omp parallel 
            {
                #pragma omp for
                for (site_i=0; site_i<no_of_sites; site_i++)
                {
                    Energy_minimum_old_XY(site_i, &spin_temp[dim_S*site_i + 0]);
                }
                #pragma omp for reduction(+:cutoff_local)
                for (site_i=0; site_i<no_of_sites*dim_S; site_i++)
                {
                    cutoff_local += (double) ( fabs(spin[site_i] - spin_temp[site_i]) > CUTOFF_SPIN );
                    
                    spin[site_i] = spin_temp[site_i];
                }
            }
        }
        #endif
        // else
        #ifdef UPDATE_CHKR_NON_EQ
        {
            #pragma omp parallel 
            {
                cutoff_local = 0.0;

                #pragma omp for reduction(+:cutoff_local)
                for (site_i=0; site_i<no_of_black_white_sites[black_or_white]; site_i++)
                {
                    long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*black_or_white + site_i];
                    double spin_local[dim_S];

                    cutoff_local += (double) ( update_to_minimum_checkerboard(site_index, spin_local) > CUTOFF_SPIN );
                    
                }

                #pragma omp for reduction(+:cutoff_local)
                for (site_i=0; site_i<no_of_black_white_sites[!black_or_white]; site_i++)
                {
                    long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*(!black_or_white) + site_i];
                    double spin_local[dim_S];
                    
                    cutoff_local += (double) ( update_to_minimum_checkerboard(site_index, spin_local) > CUTOFF_SPIN );
                    
                }            
            }
        }
        #endif

        return cutoff_local;
    }
    #endif
    #endif

    #ifdef CUTOFF_BY_SUM
    double find_change()
    {
        static int print_first = 0;
        if (print_first == 0)
        {
            printf("\n Using find change sum.. \n");
            print_first = !print_first;
        }
        double cutoff_local;
        
        long int site_i;
        static int black_or_white = 0;

        // if (update_all_or_checker == 0)
        #ifdef UPDATE_ALL_NON_EQ
        {
            cutoff_local = 0.0;

            #pragma omp parallel 
            {
                #pragma omp for
                for (site_i=0; site_i<no_of_sites; site_i++)
                {
                    Energy_minimum_new_XY(site_i, &spin_temp[dim_S*site_i + 0]);
                }
                #pragma omp for reduction(+:cutoff_local)
                for (site_i=0; site_i<no_of_sites*dim_S; site_i++)
                {
                    cutoff_local += fabs(spin[site_i] - spin_temp[site_i]);
                    spin[site_i] = spin_temp[site_i];
                }
            }
        }
        #endif
        // else
        #ifdef UPDATE_CHKR_NON_EQ
        {
            #pragma omp parallel 
            {
                cutoff_local = 0.0;

                #pragma omp for reduction(+:cutoff_local)
                for (site_i=0; site_i<no_of_black_white_sites[black_or_white]; site_i++)
                {
                    long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*black_or_white + site_i];
                    double spin_local[dim_S];

                    cutoff_local += update_to_minimum_checkerboard(site_index, spin_local);
                }

                #pragma omp for reduction(+:cutoff_local)
                for (site_i=0; site_i<no_of_black_white_sites[!black_or_white]; site_i++)
                {
                    long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*(!black_or_white) + site_i];
                    double spin_local[dim_S];

                    cutoff_local += update_to_minimum_checkerboard(site_index, spin_local);
                }            
            }
        }
        #endif

        return cutoff_local;
    }
    #endif

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
        #else
        for(i=0; i<no_of_sites*dim_S; i++)
        {
            spin_bkp[i] = spin[i];
        }
        for(j_S=0; j_S<dim_S; j_S++)
        {
            m_bkp[j_S] = m[j_S];
        }
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
        #else
        for(i=0; i<no_of_sites*dim_S; i++)
        {
            spin[i] = spin_bkp[i];
        }
        for(j_S=0; j_S<dim_S; j_S++)
        {
            m[j_S] = m_bkp[j_S];
        }
        #endif
        // if(reqd_to_print == 1)
        // {
            // printf(  "\n============================\n");
            printf(  "\r=1=     %s = %.15e ", text, h_text );
            #ifndef CHECK_AVALANCHE
            printf(    ",   delta_m = %.15e ", delta_m );
            #else
            printf(    ",   delta_S = %.15e ", delta_m );
            #endif
            printf(    ", delta_%s = %.15e ", text, order[jj_S]*delta_text );
            printf(    " - [ restore ] | ");
            printf(    " Time elapsed = %le s | ", omp_get_wtime() - start_time );
            fflush(stdout);
            // printf(  "\n============================\n");
        // }
        return 0;
    }
    
    int save_to_file(double h_text, double delta_text, int jj_S, double delta_m, char text[], int reqd_to_print)
    {
        int j_S;
        fprintf(pFile_1, "%.12e\t", h_text);
        // fprintf(pFile_1, "%.12e\t%.12e\t", h[0], h[1]);
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", h[j_S]);
        }
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", m[j_S]);
        }
        // fprintf(pFile_1, "%.12e\t", E);
        #ifndef CHECK_AVALANCHE
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", delta_M[j_S]);
        }
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", delta_M[j_S]*delta_M[j_S]);
        }
        fprintf(pFile_1, "%.12e\t", delta_m);
        #else
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", delta_S_abs[j_S]);
        }
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", delta_S_squared[j_S]);
        }
        fprintf(pFile_1, "%.12e\t", delta_S_max);
        #endif
        fprintf(pFile_1, "\n");
        
        // if(reqd_to_print == 1)
        // {
            // printf(  "\n============================\n");
            printf(  "\r=1=     %s = %.15e ", text, h_text );
            #ifndef CHECK_AVALANCHE
            printf(    ",   delta_m = %.15e ", delta_m );
            #else
            printf(    ",   delta_S = %.15e ", delta_m );
            #endif
            printf(    ", delta_%s = %.15e ", text, order[jj_S]*delta_text );
            printf(    " - [ backup ]  | ");
            printf(    " Time elapsed = %le s | ", omp_get_wtime() - start_time );
            fflush(stdout);
            // printf(  "\n============================\n");
        // }
        return 0;
    }
    
    int check_avalanche()
    {
        cutoff_check[0] = 0.0;
        cutoff_check[1] = 0.0;

        
        int j_S;
        long int site_i;
        static int black_or_white = 0;
        do
        {
            cutoff_check[0] = 0.0;
            cutoff_check[1] = 0.0;

            // if (update_all_or_checker == 0)
            #ifdef UPDATE_ALL_NON_EQ
            {
                // cutoff_check[0] = 0.0;

                #pragma omp parallel 
                {
                    #pragma omp for
                    for (site_i=0; site_i<no_of_sites; site_i++)
                    {
                        Energy_minimum_old_XY_temp(site_i);
                    }
                    #pragma omp for private(j_S) reduction(+:cutoff_check[:3])
                    for (site_i=0; site_i<no_of_sites; site_i++)
                    {
                        double delta_S = 0.0;
                        for (j_S=0; j_S<dim_S; j_S++)
                        {
                            delta_S += (spin_temp[site_i*dim_S+j_S] - spin_bkp[site_i*dim_S+j_S])*(spin_temp[site_i*dim_S+j_S] - spin_bkp[site_i*dim_S+j_S]);
                            cutoff_check[0] += (double) ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN );
                            
                            spin[site_i*dim_S+j_S] = spin_temp[site_i*dim_S+j_S];
                        }
                        cutoff_check[1] += (double) ( delta_S > CUTOFF_M_SQ );
                    }
                }
            }
            #endif
            // else
            #ifdef UPDATE_CHKR_NON_EQ
            {
                #pragma omp parallel 
                {
                    // cutoff_check[0] = 0.0;

                    #pragma omp for private(j_S) reduction(+:cutoff_check[:3])
                    for (site_i=0; site_i<no_of_black_white_sites[black_or_white]; site_i++)
                    {
                        long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*black_or_white + site_i];
                        // double spin_local[dim_S];
                        Energy_minimum_old_XY_temp(site_index);

                        double delta_S = 0.0;
                        for (j_S=0; j_S<dim_S; j_S++)
                        {
                            delta_S += (spin_temp[site_index*dim_S+j_S] - spin_bkp[site_index*dim_S+j_S])*(spin_temp[site_index*dim_S+j_S] - spin_bkp[site_index*dim_S+j_S]);
                            cutoff_check[0] += (double) ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                            
                            spin[site_index*dim_S+j_S] = spin_temp[site_index*dim_S+j_S];
                        }
                        cutoff_check[1] += (double) ( delta_S > CUTOFF_M_SQ );
                    }

                    #pragma omp for private(j_S) reduction(+:cutoff_check[:3])
                    for (site_i=0; site_i<no_of_black_white_sites[!black_or_white]; site_i++)
                    {
                        long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*(!black_or_white) + site_i];
                        // double spin_local[dim_S];
                        Energy_minimum_old_XY_temp(site_index);

                        double delta_S = 0.0;
                        for (j_S=0; j_S<dim_S; j_S++)
                        {
                            delta_S += (spin_temp[site_index*dim_S+j_S] - spin_bkp[site_index*dim_S+j_S])*(spin_temp[site_index*dim_S+j_S] - spin_bkp[site_index*dim_S+j_S]);
                            cutoff_check[0] += (double) ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                            
                            spin[site_index*dim_S+j_S] = spin_temp[site_index*dim_S+j_S];
                        }
                        cutoff_check[1] += (double) ( delta_S > CUTOFF_M_SQ );
                    }         
                }
            }
            #endif
        }
        while ( cutoff_check[0] > 0.0 && cutoff_check[1] == 0.0 );

        return 0;
    }

    int continue_avalanche()
    {
        cutoff_check[0] = 0.0;
        cutoff_check[1] = 0.0;

        
        int j_S;
        long int site_i;
        static int black_or_white = 0;
        do
        {
            cutoff_check[0] = 0.0;
            // cutoff_check[1] = 0.0;

            // if (update_all_or_checker == 0)
            #ifdef UPDATE_ALL_NON_EQ
            {
                // cutoff_check[0] = 0.0;

                #pragma omp parallel 
                {
                    #pragma omp for 
                    for (site_i=0; site_i<no_of_sites; site_i++)
                    {
                        Energy_minimum_old_XY_temp(site_i);
                    }
                    #pragma omp for private(j_S) reduction(+:cutoff_check[:3])
                    for (site_i=0; site_i<no_of_sites; site_i++)
                    {
                        // double delta_S = 0.0;
                        for (j_S=0; j_S<dim_S; j_S++)
                        {
                            cutoff_check[0] += (double) ( fabs(spin[site_i*dim_S+j_S] - spin_temp[site_i*dim_S+j_S]) > CUTOFF_SPIN );
                            
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
                    // cutoff_check[0] = 0.0;

                    #pragma omp for private(j_S) reduction(+:cutoff_check[:3])
                    for (site_i=0; site_i<no_of_black_white_sites[black_or_white]; site_i++)
                    {
                        long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*black_or_white + site_i];
                        // double spin_local[dim_S];
                        Energy_minimum_old_XY_temp(site_index);

                        // double delta_S = 0.0;
                        for (j_S=0; j_S<dim_S; j_S++)
                        {
                            cutoff_check[0] += (double) ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                            
                            spin[site_index*dim_S+j_S] = spin_temp[site_index*dim_S+j_S];
                        }
                        
                    }

                    #pragma omp for private(j_S) reduction(+:cutoff_check[:3])
                    for (site_i=0; site_i<no_of_black_white_sites[!black_or_white]; site_i++)
                    {
                        long int site_index = black_white_checkerboard[no_of_black_white_sites[0]*(!black_or_white) + site_i];
                        // double spin_local[dim_S];
                        Energy_minimum_old_XY_temp(site_index);

                        // double delta_S = 0.0;
                        for (j_S=0; j_S<dim_S; j_S++)
                        {
                            cutoff_check[0] += (double) ( fabs(spin[site_index*dim_S+j_S] - spin_temp[site_index*dim_S+j_S]) > CUTOFF_SPIN );
                            
                            spin[site_index*dim_S+j_S] = spin_temp[site_index*dim_S+j_S];
                        }
                        
                    }         
                }
            }
            #endif
        }
        while ( cutoff_check[0] > 0.0 );

        return 0;
    }
    
    #ifdef CHECK_AVALANCHE
    int const_delta_phi(double h_phi, double delta_phi, int jj_S, double h_start)
    {
        static int binary_or_slope = 1;
        static long int counter = 1;
        if (binary_or_slope)
        {
            printf("\n ====== CONSTANT RATE ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
        
        ensemble_m();
        ensemble_delta_S_squared();

        
        // printf(  "\n============================\n");
        if(counter % 100 == 0)
        {
            save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", 1);
            counter = 0;
        }
        else
        {
            save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", 0);
        }
        // printf(  "\n============================\n");
        counter++;

        return 0;
    }
    
    int const_delta_h_axis(double h_jj_S, double delta_h, int jj_S, double h_start)
    {
        static int binary_or_slope = 1;
        static long int counter = 1;
        if (binary_or_slope)
        {
            printf("\n ====== CONSTANT RATE ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
        
        ensemble_m();
        ensemble_delta_S_squared();
        find_delta_S_max();
        // printf(  "\n============================\n");
        if(counter % 10 == 0)
        {
            save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_jj_S", 1);
            counter = 0;
        }
        else
        {
            save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_jj_S", 0);
        }
        // printf(  "\n============================\n");
        counter++;

        return 0;
    }
    
    int slope_subdivide_phi(double h_phi, double delta_phi, int jj_S, double h_start)
    {
        static int binary_or_slope = 1;
        if (binary_or_slope)
        {
            printf("\n ====== SLOPE ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
        
        
        if (cutoff_check[1] == 0.0 && cutoff_check[0] == 0.0)
        {
            ensemble_m();
            ensemble_delta_S_squared();
            // ensemble_E();
            backing_up_spin();
            
            save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", 1);
            
            return 0;
        }

        double h_phi_k, delta_phi_k, cutoff_local;
        long int slope;
        if (cutoff_check[1] > 0.0)
        {
            find_delta_S_max();
            restoring_spin(h_phi, delta_phi, jj_S, delta_S_max, "phi", 1);
            // ensemble_m();
            slope = delta_S_max/del_phi + 1;
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

        //     save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", 1);

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
        static int binary_or_slope = 1;
        if (binary_or_slope)
        {
            printf("\n ====== SLOPE ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
        
        
        if (cutoff_check[1] == 0.0 && cutoff_check[0] == 0.0)
        {
            ensemble_m();
            ensemble_delta_S_squared();
            // ensemble_E();
            backing_up_spin();
            
            save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_jj_S", 1);
            
            return 0;
        }

        

        double h_jj_S_k, delta_h_k, cutoff_local;
        long int slope;
        if (cutoff_check[1] > 0.0)
        {
            find_delta_S_max();
            restoring_spin(h_jj_S, delta_h, jj_S, delta_S_max, "h_jj_S", 1);
            // ensemble_m();
            slope = delta_S_max/del_h + 1;
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

        //     save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 1);

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
        static int binary_or_slope = 1;
        if (binary_or_slope)
        {
            printf("\n ====== BINARY ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
        
        
        if (cutoff_check[1] == 0.0 && cutoff_check[0] == 0.0)
        {
            ensemble_m();
            ensemble_delta_S_squared_max();
            // ensemble_E();
            backing_up_spin();
            
            save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", 1);
            
            return 0;
        }
        
        // if (cutoff_check[1] > 0.0 && cutoff_check[0] > 0.0)
        // if (cutoff_check[1] > 0.0 && cutoff_check[0] == 0.0)
        // if (cutoff_check[1] == 0.0 && cutoff_check[0] > 0.0)

        double h_phi_k, delta_phi_k, cutoff_local;
        
        if (cutoff_check[1] > 0.0)
        {
            restoring_spin(h_phi, delta_phi, jj_S, delta_S_max, "phi", 1);
            // ensemble_m();
            delta_phi_k = delta_phi / 2.0;
        }
        // else 
        // {
        //     // ensemble_E();
        //     backing_up_spin();

        //     save_to_file(h_phi, delta_phi, jj_S, delta_S_max, "phi", 1);

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
        static int binary_or_slope = 1;
        if (binary_or_slope)
        {
            printf("\n ====== BINARY ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
        
        if (cutoff_check[1] == 0.0 && cutoff_check[0] == 0.0)
        {
            ensemble_m();
            ensemble_delta_S_squared_max();
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_jj_S, delta_h, jj_S, delta_S_max, "h_jj_S", 1);
            
            return 0;
        }

        double h_jj_S_k, delta_h_k, cutoff_local;
        
        if (cutoff_check[1] > 0.0)
        {
            restoring_spin(h_jj_S, delta_h, jj_S, delta_S_max, "h_jj_S", 1);
            // ensemble_m();
            delta_h_k = delta_h / 2.0;
        }
        // else 
        // {
        //     // ensemble_E();
        //     backing_up_spin();

        //     save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 1);

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
        static int binary_or_slope = 1;
        static long int counter = 1;
        static int last_phi_restored = 0;
        if (binary_or_slope)
        {
            printf("\n ====== DYNAMIC BINARY DIVISION RATE ====== \n");
            binary_or_slope = !binary_or_slope;
        }
        // double h_phi_k, delta_phi_k;
        // h_phi_k = *h_phi;
        // delta_phi_k = *delta_phi;
        int j_S;

        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        double ratio_delta_m = del_phi/del_phi_cutoff;
        if (delta_m > del_phi_cutoff)
        {
            ratio_delta_m = del_phi/delta_m;
        }

        if (delta_phi[0] <= del_phi_cutoff)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
            
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

                    save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);

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
                        save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
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
                    save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
                    return 5;
                }
            }
        }
        return 6;
    }

    int dynamic_binary_subdivide_h_axis(double *h_jj_S, double *delta_h, int jj_S, double h_start)
    {
        static int binary_or_slope = 1;
        static long int counter = 1;
        static int last_h_restored = 0;
        if (binary_or_slope)
        {
            printf("\n ====== DYNAMIC BINARY DIVISION RATE ====== \n");
            binary_or_slope = !binary_or_slope;
        }
        // double h_phi_k, delta_phi_k;
        // h_phi_k = *h_phi;
        // delta_phi_k = *delta_phi;
        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        double ratio_delta_m = del_h/del_h_cutoff;
        if (delta_m > del_h_cutoff)
        {
            ratio_delta_m = del_h/delta_m;
        }

        if (delta_h[0] <= del_h_cutoff*h_start)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
            
            if (ratio_delta_m > 2 && last_h_restored == 0)
            {
                delta_h[0] = delta_h[0] * 2;
                if (delta_h[0] >= del_h*h_start)
                {
                    delta_h[0] = del_h*h_start;
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
            if (delta_h[0] < del_h*h_start)
            {
                if (ratio_delta_m > 2 && last_h_restored == 0)
                {
                    backing_up_spin();

                    save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);

                    delta_h[0] = delta_h[0] * 2;
                    if (delta_h[0] >= del_h*h_start)
                    {
                        delta_h[0] = del_h*h_start;
                    }
                    return 1;
                }
                else
                {
                    if (ratio_delta_m <= 1)
                    {
                        restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                        last_h_restored = 1;
                        h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                        
                        delta_h[0] = delta_h[0] / 2;
                        return 2;
                    }
                    else
                    {
                        backing_up_spin();
                        last_h_restored = 0;
                        save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                        return 3;
                    }
                }
            }
            else
            {
                if (ratio_delta_m <= 1)
                {
                    restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                    last_h_restored = 1;
                    h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                    
                    delta_h[0] = delta_h[0] / 2;
                    return 4;
                }
                else
                {
                    backing_up_spin();
                    last_h_restored = 0;
                    save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                    return 5;
                }
            }
        }
        return 6;
    }

    int dynamic_binary_slope_divide_phi(double *h_phi, double *delta_phi, int jj_S, double h_start)
    {
        static int binary_or_slope = 1;
        static int last_phi_restored = 0;
        static const double reqd_ratio = 1.1;
        static long int counter = 1;
        if (binary_or_slope)
        {
            printf("\n ====== DYNAMIC BINARY DIVISION TO ADJUST TO SLOPE ====== \n");
            binary_or_slope = !binary_or_slope;
        }
        // double h_phi_k, delta_phi_k;
        // h_phi_k = *h_phi;
        // delta_phi_k = *delta_phi;
        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        double ratio_delta_m = del_phi/del_phi_cutoff;
        if (delta_m > del_phi_cutoff)
        {
            ratio_delta_m = del_phi/delta_m;
        }

        if (delta_phi[0] <= del_phi_cutoff)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
            
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

                    save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);

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

                        save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
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

                    save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
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
        static int binary_or_slope = 1;
        static int last_h_restored = 0;
        static const double reqd_ratio = 1.1;
        static long int counter = 1;
        if (binary_or_slope)
        {
            printf("\n ====== DYNAMIC BINARY DIVISION TO ADJUST TO SLOPE ====== \n");
            binary_or_slope = !binary_or_slope;
        }
        // double h_phi_k, delta_phi_k;
        // h_phi_k = *h_phi;
        // delta_phi_k = *delta_phi;
        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        double ratio_delta_m = del_h/del_h_cutoff;
        if (delta_m > del_h_cutoff)
        {
            ratio_delta_m = del_h/delta_m;
        }

        if (delta_h[0] <= del_h_cutoff*h_start)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
            
            if (ratio_delta_m > 2 && last_h_restored == 0)
            {
                delta_h[0] = delta_h[0] * 2;
                if (delta_h[0] >= del_h*h_start)
                {
                    delta_h[0] = del_h*h_start;
                }
            }
            else
            {
                if (ratio_delta_m > reqd_ratio && last_h_restored == 0)
                {
                    delta_h[0] = delta_h[0] * ratio_delta_m / reqd_ratio;
                    if (delta_h[0] >= del_h*h_start)
                    {
                        delta_h[0] = del_h*h_start;
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
            if (delta_h[0] < del_h*h_start)
            {
                if (ratio_delta_m > 2 && last_h_restored == 0)
                {
                    backing_up_spin();

                    save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);

                    delta_h[0] = delta_h[0] * 2;
                    if (delta_h[0] >= del_h*h_start)
                    {
                        delta_h[0] = del_h*h_start;
                    }
                    return 1;
                }
                else
                {
                    if (ratio_delta_m <= 1)
                    {
                        restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                        last_h_restored = 1;
                        h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                        
                        delta_h[0] = delta_h[0] / 2;
                        return 2;
                    }
                    else
                    {
                        backing_up_spin();

                        save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                        if (last_h_restored == 0)
                        {
                            delta_h[0] = delta_h[0] * ratio_delta_m / reqd_ratio;
                            if (delta_h[0] >= del_h*h_start)
                            {
                                delta_h[0] = del_h*h_start;
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
                    restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                    last_h_restored = 1;
                    h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                    
                    delta_h[0] = delta_h[0] / 2;
                    return 4;
                }
                else
                {
                    backing_up_spin();

                    save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
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

    #ifndef CHECK_AVALANCHE
    int const_delta_phi(double h_phi, double delta_phi, int jj_S, double h_start)
    {
        static int binary_or_slope = 1;
        static long int counter = 1;
        if (binary_or_slope)
        {
            printf("\n ====== CONSTANT RATE ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        
        // printf(  "\n============================\n");
        if(counter % 100 == 0)
        {
            save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", 1);
            counter = 0;
        }
        else
        {
            save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", 0);
        }
        // printf(  "\n============================\n");
        counter++;

        return 0;
    }
    
    int const_delta_h_axis(double h_jj_S, double delta_h, int jj_S, double h_start)
    {
        static int binary_or_slope = 1;
        static long int counter = 1;
        if (binary_or_slope)
        {
            printf("\n ====== CONSTANT RATE ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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
        
        // printf(  "\n============================\n");
        if(counter % 10 == 0)
        {
            save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 1);
            counter = 0;
        }
        else
        {
            save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 0);
        }
        // printf(  "\n============================\n");
        counter++;

        return 0;
    }

    int slope_subdivide_phi(double h_phi, double delta_phi, int jj_S, double h_start)
    {
        static int binary_or_slope = 1;
        if (binary_or_slope)
        {
            printf("\n ====== SLOPE ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        if (delta_phi <= del_phi_cutoff)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", 1);

            return 0;
        }

        double h_phi_k, delta_phi_k, cutoff_local;
        long int slope;
        if (delta_m > del_phi)
        {
            restoring_spin(h_phi, delta_phi, jj_S, delta_m, "phi", 1);
            // ensemble_m();
            slope = delta_m/del_phi + 1;
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

            save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", 1);

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
            
            cutoff_local = -0.1;
            
            do
            {
                // double cutoff_local_last = cutoff_local;
                cutoff_local = find_change();
            }
            while (cutoff_local > CUTOFF_SPIN); // 10^-14


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
            
            cutoff_local = -0.1;
            
            do
            {
                // double cutoff_local_last = cutoff_local;
                cutoff_local = find_change();
            }
            while (cutoff_local > CUTOFF_SPIN); // 10^-14


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
        static int binary_or_slope = 1;
        if (binary_or_slope)
        {
            printf("\n ====== SLOPE ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        if (delta_h <= del_h_cutoff*h_start)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 1);

            return 0;
        }

        double h_jj_S_k, delta_h_k, cutoff_local;
        long int slope;
        if (delta_m > del_h)
        {
            restoring_spin(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 1);
            // ensemble_m();
            slope = delta_m/del_h + 1;
            delta_h_k = delta_h / (double) slope;
            if (delta_h_k < del_h_cutoff*h_start)
            {
                slope = delta_h/del_h_cutoff + 1;
                delta_h_k = delta_h / (double) slope;
            }
        }
        else 
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 1);

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
            
            cutoff_local = -0.1;

            do
            {
                // double cutoff_local_last = cutoff_local;
                cutoff_local = find_change();
            }
            while (cutoff_local > CUTOFF_SPIN); // 10^-14


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
            
            cutoff_local = -0.1;

            do
            {
                // double cutoff_local_last = cutoff_local;
                cutoff_local = find_change();
            }
            while (cutoff_local > CUTOFF_SPIN); // 10^-14


            slope_subdivide_h_axis(h_jj_S_k, delta_h_k, jj_S, h_start);
        }
        // printf("\n===\n");
        // printf(  "=2=");
        // printf("\n===\n");
        return 2;
    }
    
    int binary_subdivide_phi(double h_phi, double delta_phi, int jj_S, double h_start)
    {
        static int binary_or_slope = 1;
        if (binary_or_slope)
        {
            printf("\n ====== BINARY ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
       
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        if (delta_phi <= del_phi_cutoff)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", 1);
            
            return 0;
        }

        double h_phi_k, delta_phi_k, cutoff_local;
        
        if (delta_m > del_phi)
        {
            restoring_spin(h_phi, delta_phi, jj_S, delta_m, "phi", 1);
            // ensemble_m();
            delta_phi_k = delta_phi / 2.0;
        }
        else 
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", 1);

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
            
            cutoff_local = -0.1;
            
            do
            {
                // double cutoff_local_last = cutoff_local;
                cutoff_local = find_change();
            }
            while (cutoff_local > CUTOFF_SPIN); // 10^-14
 

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
            
            cutoff_local = -0.1;
            
            do
            {
                // double cutoff_local_last = cutoff_local;
                cutoff_local = find_change();
            }
            while (cutoff_local > CUTOFF_SPIN); // 10^-14


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
        static int binary_or_slope = 1;
        if (binary_or_slope)
        {
            printf("\n ====== BINARY ====== \n");
            binary_or_slope = !binary_or_slope;
        }

        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        if (delta_h <= del_h_cutoff*h_start)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 1);
            
            return 0;
        }

        double h_jj_S_k, delta_h_k, cutoff_local;
        
        if (delta_m > del_h)
        {
            restoring_spin(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 1);
            // ensemble_m();
            delta_h_k = delta_h / 2.0;
        }
        else 
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 1);

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
            
            cutoff_local = -0.1;

            #ifdef CHECK_AVALANCHE
            if (delta_h >= del_h)
            {
                check_avalanche_max_delta();
            }
            else 
            {
                if (delta_h <= del_h_cutoff)
                {
                    check_avalanche();
                }
                else
                {
                    continue_avalanche();
                    
                }
            }
            #else
            do
            {
                // double cutoff_local_last = cutoff_local;
                cutoff_local = find_change();
            }
            while (cutoff_local > CUTOFF_SPIN); // 10^-14
            #endif

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
            
            cutoff_local = -0.1;

            do
            {
                // double cutoff_local_last = cutoff_local;
                cutoff_local = find_change();
            }
            while (cutoff_local > CUTOFF_SPIN); // 10^-14


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
        static int binary_or_slope = 1;
        static long int counter = 1;
        static int last_phi_restored = 0;
        if (binary_or_slope)
        {
            printf("\n ====== DYNAMIC BINARY DIVISION RATE ====== \n");
            binary_or_slope = !binary_or_slope;
        }
        // double h_phi_k, delta_phi_k;
        // h_phi_k = *h_phi;
        // delta_phi_k = *delta_phi;
        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        double ratio_delta_m = del_phi/del_phi_cutoff;
        if (delta_m > del_phi_cutoff)
        {
            ratio_delta_m = del_phi/delta_m;
        }

        if (delta_phi[0] <= del_phi_cutoff)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
            
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

                    save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);

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
                        save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
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
                    save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
                    return 5;
                }
            }
        }
        return 6;
    }

    int dynamic_binary_subdivide_h_axis(double *h_jj_S, double *delta_h, int jj_S, double h_start)
    {
        static int binary_or_slope = 1;
        static long int counter = 1;
        static int last_h_restored = 0;
        if (binary_or_slope)
        {
            printf("\n ====== DYNAMIC BINARY DIVISION RATE ====== \n");
            binary_or_slope = !binary_or_slope;
        }
        // double h_phi_k, delta_phi_k;
        // h_phi_k = *h_phi;
        // delta_phi_k = *delta_phi;
        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        double ratio_delta_m = del_h/del_h_cutoff;
        if (delta_m > del_h_cutoff)
        {
            ratio_delta_m = del_h/delta_m;
        }

        if (delta_h[0] <= del_h_cutoff*h_start)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
            
            if (ratio_delta_m > 2 && last_h_restored == 0)
            {
                delta_h[0] = delta_h[0] * 2;
                if (delta_h[0] >= del_h*h_start)
                {
                    delta_h[0] = del_h*h_start;
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
            if (delta_h[0] < del_h*h_start)
            {
                if (ratio_delta_m > 2 && last_h_restored == 0)
                {
                    backing_up_spin();

                    save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);

                    delta_h[0] = delta_h[0] * 2;
                    if (delta_h[0] >= del_h*h_start)
                    {
                        delta_h[0] = del_h*h_start;
                    }
                    return 1;
                }
                else
                {
                    if (ratio_delta_m <= 1)
                    {
                        restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                        last_h_restored = 1;
                        h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                        
                        delta_h[0] = delta_h[0] / 2;
                        return 2;
                    }
                    else
                    {
                        backing_up_spin();
                        last_h_restored = 0;
                        save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                        return 3;
                    }
                }
            }
            else
            {
                if (ratio_delta_m <= 1)
                {
                    restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                    last_h_restored = 1;
                    h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                    
                    delta_h[0] = delta_h[0] / 2;
                    return 4;
                }
                else
                {
                    backing_up_spin();
                    last_h_restored = 0;
                    save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                    return 5;
                }
            }
        }
        return 6;
    }

    int dynamic_binary_slope_divide_phi(double *h_phi, double *delta_phi, int jj_S, double h_start)
    {
        static int binary_or_slope = 1;
        static int last_phi_restored = 0;
        static const double reqd_ratio = 1.1;
        static long int counter = 1;
        if (binary_or_slope)
        {
            printf("\n ====== DYNAMIC BINARY DIVISION TO ADJUST TO SLOPE ====== \n");
            binary_or_slope = !binary_or_slope;
        }
        // double h_phi_k, delta_phi_k;
        // h_phi_k = *h_phi;
        // delta_phi_k = *delta_phi;
        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        double ratio_delta_m = del_phi/del_phi_cutoff;
        if (delta_m > del_phi_cutoff)
        {
            ratio_delta_m = del_phi/delta_m;
        }

        if (delta_phi[0] <= del_phi_cutoff)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
            
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

                    save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);

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

                        save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
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

                    save_to_file(h_phi[0], delta_phi[0], jj_S, delta_m, "phi", 1);
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
        static int binary_or_slope = 1;
        static int last_h_restored = 0;
        static const double reqd_ratio = 1.1;
        static long int counter = 1;
        if (binary_or_slope)
        {
            printf("\n ====== DYNAMIC BINARY DIVISION TO ADJUST TO SLOPE ====== \n");
            binary_or_slope = !binary_or_slope;
        }
        // double h_phi_k, delta_phi_k;
        // h_phi_k = *h_phi;
        // delta_phi_k = *delta_phi;
        int j_S;
        
        // double old_m[dim_S], new_m[dim_S];
        double delta_m = 0.0;
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

        double ratio_delta_m = del_h/del_h_cutoff;
        if (delta_m > del_h_cutoff)
        {
            ratio_delta_m = del_h/delta_m;
        }

        if (delta_h[0] <= del_h_cutoff*h_start)
        {
            // ensemble_E();
            backing_up_spin();

            save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
            
            if (ratio_delta_m > 2 && last_h_restored == 0)
            {
                delta_h[0] = delta_h[0] * 2;
                if (delta_h[0] >= del_h*h_start)
                {
                    delta_h[0] = del_h*h_start;
                }
            }
            else
            {
                if (ratio_delta_m > reqd_ratio && last_h_restored == 0)
                {
                    delta_h[0] = delta_h[0] * ratio_delta_m / reqd_ratio;
                    if (delta_h[0] >= del_h*h_start)
                    {
                        delta_h[0] = del_h*h_start;
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
            if (delta_h[0] < del_h*h_start)
            {
                if (ratio_delta_m > 2 && last_h_restored == 0)
                {
                    backing_up_spin();

                    save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);

                    delta_h[0] = delta_h[0] * 2;
                    if (delta_h[0] >= del_h*h_start)
                    {
                        delta_h[0] = del_h*h_start;
                    }
                    return 1;
                }
                else
                {
                    if (ratio_delta_m <= 1)
                    {
                        restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                        last_h_restored = 1;
                        h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                        
                        delta_h[0] = delta_h[0] / 2;
                        return 2;
                    }
                    else
                    {
                        backing_up_spin();

                        save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                        if (last_h_restored == 0)
                        {
                            delta_h[0] = delta_h[0] * ratio_delta_m / reqd_ratio;
                            if (delta_h[0] >= del_h*h_start)
                            {
                                delta_h[0] = del_h*h_start;
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
                    restoring_spin(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
                    last_h_restored = 1;
                    h_jj_S[0] = h_jj_S[0] + delta_h[0] * order[jj_S];
                    
                    delta_h[0] = delta_h[0] / 2;
                    return 4;
                }
                else
                {
                    backing_up_spin();

                    save_to_file(h_jj_S[0], delta_h[0], jj_S, delta_m, "h_jj_S", 1);
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

    int zero_temp_RFXY_hysteresis_axis_checkerboard(int jj_S, double order_start)
    {
        CUTOFF_M_SQ = 4.0*del_h*del_h;
        CUTOFF_M_SQ_BY_4 = del_h*del_h;
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
            
            #else
            cudaMemcpyToSymbol("dev_CUTOFF_SPIN", &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
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
            
            #endif
        #endif

        T = 0;
        int ax_ro = 0;
        int or_ho_ra = 0;
        double cutoff_local = 0.0;
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
            h[j_S] = 0;
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
            h_start = order[jj_S]*(sigma_h_trnsvrs);
            delta_h = sigma_h_trnsvrs*del_h;
            h_order = 0;
            r_order = 0;
            initialize_spin_config();
        }
        else
        {
            if (sigma_h[jj_S] == 0.0)
            {
                h_start = order[jj_S]*(h_max);
                delta_h = h_max*del_h;
                
            }
            else
            {
                // h[!jj_S] = sigma_h[jj_S]*del_h_cutoff;
                // h[!jj_S] = 0.10*del_h_cutoff;
                // h_start = order[jj_S]*(h_max);
                if (h_i_max >= -h_i_min)
                {
                    h_start = order[jj_S]*(h_max + h_i_max);
                }
                else
                {
                    h_start = order[jj_S]*(h_max - h_i_min);
                }
                delta_h = (h_i_max + h_max)*del_h;
            }
            h_order = 0;
            r_order = 1;
            initialize_spin_config();
        }

        h_end = -h_start;
        // h_order = 0;
        // r_order = 1;
        // initialize_spin_config();
        

        printf("\nztne RFXY looping along h[%d] at T=%lf.. \n", jj_S,  T);

        ensemble_m();
        ensemble_E();
        
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
                pos += sprintf(pos, "%d", lattice_size[j_L]);
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
            //     pos += sprintf(pos, "%lf", order[j_S]);
            // }
            // pos += sprintf(pos, "}");
            pos += sprintf(pos, "_h%d_r%d", h_order, r_order);
            #ifdef enable_CUDA_CODE
            pos += sprintf(pos, "_cuda");
            #endif
            pos += sprintf(pos, ".dat");

        }
        
        // column labels and parameters
        print_header_column(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        {
            fprintf(pFile_1, "h[%d]\t", jj_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "h[%d]\t", j_S);
            }
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t", j_S);
            }
            // fprintf(pFile_1, "<E>\t");
            fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
        }
        // if (update_all_or_checker == 0)
        #ifdef UPDATE_ALL_NON_EQ
        {
            printf("\nUpdating all sites simultaneously.. \n");
            
            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
            fprintf(pFile_1, "Updating all sites simultaneously.. \n");
            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");

            // spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        }
        #endif
        // else
        #ifdef UPDATE_CHKR_NON_EQ
        {
            printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
            fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
        }
        #endif
        fclose(pFile_1);

        long int site_i;
        int black_or_white = 0;
        double h_jj_S;


        pFile_1 = fopen(output_file_0, "a");
        // print statements:
        {
            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
            printf("\n%lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            fprintf(pFile_1, "\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                    fprintf(pFile_1, ",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                    fprintf(pFile_1, "%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                    fprintf(pFile_1, "%lf", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
            fprintf(pFile_1, "}, %d, %d)\n", h_order, r_order);
            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
        }
        h_jj_S = h_start;
        #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
            delta_h = del_h_cutoff*fabs(h_start);
        #endif
        // for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
        while (order[jj_S] * h_jj_S >= order[jj_S] * h_end)
        {
            h[jj_S] = h_jj_S;
            #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE) 
                backing_up_spin();
            #endif

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #else
                cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #endif
            #endif
            
            
            #ifdef CHECK_AVALANCHE
                #ifndef CONST_RATE
                if (delta_h <= del_h_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }
                #else
                continue_avalanche();
                #endif
            #else
            cutoff_local = -0.1;
            do
            {
                // double cutoff_local_last = cutoff_local;
                cutoff_local = find_change();
            }
            while (cutoff_local > CUTOFF_SPIN); // 10^-14
            #endif
            
            #ifdef enable_CUDA_CODE
            // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
            #endif
            
            
            if (h_jj_S != h_start)
            {
                // printf(  "=========================");
                // printf("\n  h_phi != 0.0 (%.15e)  \n", h_phi);
                // printf(  "=========================");
                
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
            else
            {
                int j_S;
                
                // double old_m[dim_S], new_m[dim_S];
                double delta_m = 0.0;
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

                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 0);

                printf(  "\n=========================");
                printf("\n  h[%d] = h_start (%.15e)  \n", jj_S, h_jj_S);
                printf(  "=========================\n");
                
            }
            
            
            // printf("\nblah = %lf", h[jj_S]);
            // printf("\nblam = %lf", m[jj_S]);
            // printf("\n");

            // fprintf(pFile_1, "%.12e\t", h[jj_S]);
            // for(j_S=0; j_S<dim_S; j_S++)
            // {
            //     fprintf(pFile_1, "%.12e\t", m[j_S]);
            // }
            // fprintf(pFile_1, "%.12e\t", E);

            // fprintf(pFile_1, "\n");
            #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
                // if (h_phi * order[jj_S] + delta_phi >= 1.0 && h_phi * order[jj_S] < 1.0)
                if (h_jj_S * order[jj_S] - delta_h <= h_end * order[jj_S] && h_jj_S * order[jj_S] > h_end * order[jj_S])
                {
                    // delta_phi = 1.0 - h_phi * order[jj_S];
                    delta_h = (-h_end + h_jj_S) * order[jj_S];
                    // h_phi = 1.0;
                    h_jj_S = h_end;
                }
                else
                {
                    h_jj_S = h_jj_S - order[jj_S] * delta_h;
                }
            #else
                h_jj_S = h_jj_S - order[jj_S] * delta_h;
            #endif
        }
        fclose(pFile_1);
        printf(  "\n=========================");
        printf("\n  |h[%d]| > |h_end| (%.15e)  \n", jj_S, h_jj_S);
        printf(  "=========================\n");

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
            h_start = order[jj_S]*(sigma_h_trnsvrs);
            initialize_spin_config();
        }
        else
        {
            if (sigma_h[jj_S] == 0.0)
            {
                h_start = order[jj_S]*(h_max);
            }
            else
            {
                // h_start = order[jj_S]*(h_max);
                if (h_i_max >= -h_i_min)
                {
                    h_start = order[jj_S]*(h_max + h_i_max);
                }
                else
                {
                    h_start = order[jj_S]*(h_max - h_i_min);
                }
            }
            
        }
        h_end = -h_start;
        // h_order = 0;
        // r_order = 1;
        // initialize_spin_config();
        
        ensemble_m();
        ensemble_E();

        /* pFile_1 = fopen(output_file_0, "a");
        for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
        {
            cutoff_local = -0.1;
            
            do 
            {
                
                cutoff_local = find_change();

                // printf("\nblac = %g\n", cutoff_local);

            }
            while (cutoff_local > CUTOFF_SPIN); // 10^-14

            ensemble_m();
            ensemble_E();

            // printf("\nblah = %lf", h[jj_S]);
            // printf("\nblam = %lf", m[jj_S]);
            // printf("\n");
            
            fprintf(pFile_1, "%.12e\t", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\t", E);

            fprintf(pFile_1, "\n");
        }
        fclose(pFile_1); */
        pFile_1 = fopen(output_file_0, "a");
        // print statements:
        {
            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
            printf("\n%lf", m[0]);
            for(j_S=1; j_S<dim_S; j_S++)
            {
                printf(",%lf", m[j_S]);
            }
            printf("\norder = ({");
            fprintf(pFile_1, "\norder = ({");
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S != 0)
                {
                    printf(",");
                    fprintf(pFile_1, ",");
                }
                if (j_S == jj_S)
                {
                    printf("%lf-->%lf", order[j_S], -order[j_S]);
                    fprintf(pFile_1, "%lf-->%lf", order[j_S], -order[j_S]);
                }
                else
                {
                    printf("%lf", order[j_S]);
                    fprintf(pFile_1, "%lf", order[j_S]);
                }
            }
            printf("}, %d, %d)\n", h_order, r_order);
            fprintf(pFile_1, "}, %d, %d)\n", h_order, r_order);
            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
        }
        h_jj_S = h_start;
        #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
            delta_h = del_h_cutoff*fabs(h_start);
        #endif
        // for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
        while (order[jj_S] * h[jj_S] >= order[jj_S] * h_end)
        {
            h[jj_S] = h_jj_S;
            
            #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE) 
                backing_up_spin();
            #endif

            #ifdef enable_CUDA_CODE
                #ifdef CUDA_with_managed
                cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #else
                cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                #endif
            #endif

            
            #ifdef CHECK_AVALANCHE
                #ifndef CONST_RATE
                if (delta_h <= del_h_cutoff)
                {
                    continue_avalanche();
                }
                else
                {
                    check_avalanche();
                }
                #else
                continue_avalanche();
                #endif
            #else
            cutoff_local = -0.1;
            do
            {
                // double cutoff_local_last = cutoff_local;
                cutoff_local = find_change();
            }
            while (cutoff_local > CUTOFF_SPIN); // 10^-14
            #endif

            #ifdef enable_CUDA_CODE
            // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
            #endif

            if (h_jj_S != h_start)
            {
                // printf(  "=========================");
                // printf("\n  h_phi != 0.0 (%.15e)  \n", h_phi);
                // printf(  "=========================");
                
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
            else
            {
                int j_S;
                
                // double old_m[dim_S], new_m[dim_S];
                double delta_m = 0.0;
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

                save_to_file(h_jj_S, delta_h, jj_S, delta_m, "h_jj_S", 0);

                printf(  "\n=========================");
                printf("\n  h[%d] = h_start (%.15e)  \n", jj_S, h_jj_S);
                printf(  "=========================\n");
                
            }
            
            
            // printf("\nblah = %lf", h[jj_S]);
            // printf("\nblam = %lf", m[jj_S]);
            // printf("\n");

            // fprintf(pFile_1, "%.12e\t", h[jj_S]);
            // for(j_S=0; j_S<dim_S; j_S++)
            // {
            //     fprintf(pFile_1, "%.12e\t", m[j_S]);
            // }
            // fprintf(pFile_1, "%.12e\t", E);

            // fprintf(pFile_1, "\n");
            #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
                // if (h_phi * order[jj_S] + delta_phi >= 1.0 && h_phi * order[jj_S] < 1.0)
                if (h_jj_S * order[jj_S] - delta_h <= h_end * order[jj_S] && h_jj_S * order[jj_S] > h_end * order[jj_S])
                {
                    // delta_phi = 1.0 - h_phi * order[jj_S];
                    delta_h = (-h_end + h_jj_S) * order[jj_S];
                    // h_phi = 1.0;
                    h_jj_S = h_end;
                }
                else
                {
                    h_jj_S = h_jj_S - order[jj_S] * delta_h;
                }
            #else
                h_jj_S = h_jj_S - order[jj_S] * delta_h;
            #endif
        }
        fclose(pFile_1);
        printf(  "\n=========================");
        printf("\n  |h[%d]| > |h_end| (%.15e)  \n", jj_S, h_jj_S);
        printf(  "=========================\n");

        // if (update_all_or_checker == 0)
        // {
        //     free(spin_temp);
        // }
        return 0;
    }

    int zero_temp_RFXY_hysteresis_rotate_checkerboard(int jj_S, double order_start, double h_start, char output_file_name[])
    {
        CUTOFF_M_SQ = 4.0*pie*pie*del_phi*del_phi;
        CUTOFF_M_SQ_BY_4 = pie*pie*del_phi*del_phi;
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
            
            #else
            cudaMemcpyToSymbol("dev_CUTOFF_SPIN", &CUTOFF_SPIN, sizeof(double), cudaMemcpyHostToDevice);
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
            
            #endif
        #endif

        T = 0;
        double delta_phi = del_phi;

        pFile_1 = fopen(output_file_name, "a");
        // if (update_all_or_checker == 0)
        #ifdef UPDATE_ALL_NON_EQ
        {
            printf("\nUpdating all sites simultaneously.. \n");
            
            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
            fprintf(pFile_1, "Updating all sites simultaneously..\n");
            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");

            // spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        }
        #endif
        // else
        #ifdef UPDATE_CHKR_NON_EQ
        {
            printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
            fprintf(pFile_1, "Updating all (first)black/(then)white checkerboard sites simultaneously..\n");
            fprintf(pFile_1, "----------------------------------------------------------------------------------\n");
        }
        #endif
        fclose(pFile_1);

        double cutoff_local = 0.0;
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
        // int black_or_white = 0;

        int repeat_loop = 1;
        int repeat_cond = 1;
        int restore_chkpt = 1;
        int is_complete = 0;
        long int h_counter = 0;
        
        while (repeat_cond)
        {
            pFile_1 = fopen(output_file_name, "a");
            h_phi = 0.0;
            #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
                delta_phi = del_phi;
            #endif
            // for (h_phi = 0.0; h_phi * order[jj_S] <= 1.0; h_phi = h_phi + order[jj_S] * delta_phi)
            while (h_phi * order[jj_S] <= 1.0)
            {
                #ifdef CHECKPOINT_TIME
                    if (omp_get_wtime()-start_time > CHECKPOINT_TIME)
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

                        fprintf(pFile_1, "----- Checkpointed here. -----\n");
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
                
                #if defined (BINARY_DIVISION) || defined (DIVIDE_BY_SLOPE) || defined (CONST_RATE)
                    backing_up_spin();
                #endif

                #ifdef enable_CUDA_CODE
                    #ifdef CUDA_with_managed
                    cudaMemcpy(dev_h, h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #else
                    cudaMemcpyToSymbol("dev_h", h, dim_S*sizeof(double), cudaMemcpyHostToDevice);
                    #endif
                #endif
                
                
                #ifdef CHECK_AVALANCHE
                    #ifndef CONST_RATE
                    if (delta_phi <= del_phi_cutoff)
                    {
                        continue_avalanche();
                    }
                    else
                    {
                        check_avalanche();
                    }
                    #else
                    continue_avalanche();
                    #endif
                #else
                cutoff_local = -0.1;
                do
                {
                    // double cutoff_local_last = cutoff_local;
                    cutoff_local = find_change();
                }
                while (cutoff_local > CUTOFF_SPIN); // 10^-14
                #endif
                
                #ifdef enable_CUDA_CODE
                // cudaMemcpy(spin, dev_spin, dim_S*no_of_sites*sizeof(double), cudaMemcpyDeviceToHost);
                #endif

                if (h_phi != 0.0)
                {
                    // printf(  "=========================");
                    // printf("\n  h_phi != 0.0 (%.15e)  \n", h_phi);
                    // printf(  "=========================");
                    
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
                    
                }
                else
                {
                    int j_S;
                    
                    // double old_m[dim_S], new_m[dim_S];
                    double delta_m = 0.0;
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
                    
                    save_to_file(h_phi, delta_phi, jj_S, delta_m, "phi", 0);

                    printf(  "\n=========================");
                    printf("\n  h_phi = 0.0 (%.15e)  \n", h_phi);
                    printf(  "=========================\n");
                    
                }
                
                #ifdef SAVE_SPIN_AFTER
                    if ( h_counter % SAVE_SPIN_AFTER == 0 )
                    {
                        char append_string[128];
                        char *pos = append_string;
                        pos += sprintf(pos, "_loop_%d", repeat_loop);
                        save_spin_config(append_string, "a");
                    }
                    h_counter++;
                #endif

                #if defined (DYNAMIC_BINARY_DIVISION) || defined (DYNAMIC_BINARY_DIVISION_BY_SLOPE)
                    if (h_phi * order[jj_S] + delta_phi >= 1.0 && h_phi * order[jj_S] < 1.0)
                    {
                        delta_phi = 1.0 - h_phi * order[jj_S];
                        h_phi = 1.0;
                    }
                    else
                    {
                        h_phi = h_phi + order[jj_S] * delta_phi;
                    }
                #else
                    h_phi = h_phi + order[jj_S] * delta_phi;
                #endif
            }
            printf(  "\n=========================");
            printf("\n  h_phi > 1.0 (%.15e)  \n", h_phi);
            printf(  "=========================\n");

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
            save_spin_config(append_string, "a");
            if (repeat_loop == hysteresis_repeat)
            {
                break;
            }
            repeat_loop++;
            h_counter = 0;
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

    int ordered_initialize_and_rotate_checkerboard(int jj_S, double order_start, double h_rotate_abs)
    {
        T = 0;
        ax_ro = 1;
        or_ho_ra = 0;
        double cutoff_local = 0.0;
        int j_S, j_L;
        
        
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
        double h_start = order[jj_S]*(h_rotate_abs);
        h[jj_S] = h_start;
        double delta_phi = del_phi; // double delta_phi = 1.0; // for h_start=0
        h_order = 0;
        r_order = 0;
        initialize_spin_config();

        ensemble_m();
        ensemble_E();
        
        // print statements:
        {
            printf("\n%lf", m[0]);
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
                pos += sprintf(pos, "%d", lattice_size[j_L]);
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
                    pos += sprintf(pos, "(%lf)", h_start);
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
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "_%lf}", delta_phi);
            
            pos += sprintf(pos, "_o_r.dat");
        }
        if( access( output_file_0, F_OK ) != -1 )
        {
            if (RESTORE_CHKPT_VALUE == 0)
            {
                printf("File exists! filename = %s\n", output_file_0);
                return 0; // file exists
            }
        }
        else
        {
            // column labels and parameters
            print_header_column(output_file_0);
            pFile_1 = fopen(output_file_0, "a");
            {
                fprintf(pFile_1, "phi(h[:])\t");
                fprintf(pFile_1, "h[0]\t");
                fprintf(pFile_1, "h[1]\t");
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "<m[%d]>\t", j_S);
                }
                // fprintf(pFile_1, "<E>\t");
                fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
            }
            fclose(pFile_1);            
        }
        
        T = 0;
            
        int is_chkpt = zero_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start, h_start, output_file_0);
        
        return is_chkpt;
    }

    int field_cool_and_rotate_checkerboard(int jj_S, double order_start, double h_rotate_abs)
    {
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
        double h_start = order[jj_S]*(h_rotate_abs);
        h[jj_S] = h_start;
        double delta_phi = del_phi; // double delta_phi = 1.0; // for h_start=0
        h_order = 0;
        r_order = 1;
        load_spin_config("");
                
        ensemble_m();
        ensemble_E();
        
        // print statements:
        {
            printf("\n%lf", m[0]);
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
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_{%lf,%lf}_{", Temp_max, Temp_min);
            /* for (j_L = 0 ; j_L != dim_L ; j_L++) 
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
                    pos += sprintf(pos, "(%lf)", h_start);
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
            // pos += sprintf(pos, "}");    
            // pos += sprintf(pos, "_{");    
            // for (j_S = 0 ; j_S != dim_S ; j_S++) 
            // {
            //     if (j_S)
            //     {
            //         pos += sprintf(pos, ",");
            //     }
            //     pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            // }
            pos += sprintf(pos, "}");
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "_%lf}_c", delta_phi);
        }
        char output_file_1[256];
        strcpy(output_file_1, output_file_0);
        strcat(output_file_1, ".dat");
        
        // cooling_protocol T_MAX - T_MIN=0
        // column labels and parameters
        print_header_column(output_file_1);
        pFile_1 = fopen(output_file_1, "a");
        {
            
            fprintf(pFile_1, "T\t");
            // fprintf(pFile_1, "|m|\t");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t", j_S);
            } 
            for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    fprintf(pFile_1, "<m[%d]m[%d]>\t", j_S, j_SS);
                }
            } 
            fprintf(pFile_1, "<m^2>\t");
            fprintf(pFile_1, "<m^4>\t");
            // for (j_S=0; j_S<dim_S; j_S++)
            // {
            //     fprintf(pFile_1, "<m[%d]^2>\t", j_S);
            //     fprintf(pFile_1, "<m[%d]^4>\t", j_S);
            // } 
            
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "<Y[%d,%d][%d]>\t", j_S, j_SS, j_L);
                    }
                }
            } */
            // fprintf(pFile_1, "<E>\t");
            fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
        }
        fclose(pFile_1);
        
        cooling_protocol(output_file_1);
        save_spin_config("", "a");

        // rotate field
        if ( Temp_min > 0.005 )
        {
            return 0;
        }
        else
        {
            char output_file_2[256];
            strcpy(output_file_2, output_file_0);
            strcat(output_file_2, "_r.dat");
            if( access( output_file_2, F_OK ) != -1 )
            {
                if (RESTORE_CHKPT_VALUE == 0)
                {
                    printf("File exists! filename = %s\n", output_file_2);
                    return 0; // file exists
                }
            }
            else
            {
                // column labels and parameters
                print_header_column(output_file_2);
                pFile_1 = fopen(output_file_2, "a");
                {
                    fprintf(pFile_1, "\nphi(h[:])\t ");
                    fprintf(pFile_1, "h[0]\t ");
                    fprintf(pFile_1, "h[1]\t ");
                    for (j_S=0; j_S<dim_S; j_S++)
                    {
                        fprintf(pFile_1, "<m[%d]>\t ", j_S);
                    }
                    fprintf(pFile_1, "<E>\t");
                    fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
                }
                fclose(pFile_1);
            }
            
            int is_chkpt = zero_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start, h_start, output_file_2);
            return is_chkpt;
        }
        
        return 0;
    }

    int random_initialize_and_rotate_checkerboard(int jj_S, double order_start, double h_rotate_abs)
    {
        T = 0;
        int ax_ro = 1;
        int or_ho_ra = 2;
        int j_S, j_L;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0;
        }
        order[jj_S] = order_start;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            h[j_S] = 0;
        }
        double h_start = order[jj_S]*(h_rotate_abs);
        h[jj_S] = h_start;
        double delta_phi = del_phi;
        h_order = 0;
        r_order = 1;
        initialize_spin_config();
        
        ensemble_m();
        ensemble_E();
        
        // print statements:
        {
            printf("\n%lf", m[0]);
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
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hys_rot_rand_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
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
                    pos += sprintf(pos, "(%lf)", h_start);
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
            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "_%lf}_ri.dat", delta_phi);
        }
        if( access( output_file_0, F_OK ) != -1 )
        {
            if (RESTORE_CHKPT_VALUE == 0)
            {
                printf("File exists! filename = %s\n", output_file_0);
                return 0; // file exists
            }
        }
        else
        {
            // column labels and parameters
            print_header_column(output_file_0);
            pFile_1 = fopen(output_file_0, "a");
            {
                fprintf(pFile_1, "phi(h[:])\t");
                fprintf(pFile_1, "h[0]\t");
                fprintf(pFile_1, "h[1]\t");
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "<m[%d]>\t", j_S);
                }
                fprintf(pFile_1, "<E>\t");
                fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
            }
            fclose(pFile_1);
        }
            
        int is_chkpt = zero_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start, h_start, output_file_0);
                
        return is_chkpt;
    }

//====================      RFXYZ ZTNE                       ====================//
    
    int checking_O3_spin_with_O2_RF()
    {
        T = 0.0;
        int j_S, j_L;
        double cutoff_local = -0.1;
        long int counter = 0;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            order[j_S] = 0.0;
        }
        order[0] = 1.0 ;
        r_order = 0;
        h_order = 1;
        initialize_spin_config();
        
        char output_file_1[] = "magnetization_O3_O2_h.dat";

        // char append_string[128];
        // char *pos = append_string;
        // pos += sprintf(pos, "_count_%ld", counter);
        // save_spin_config(append_string, "w");
            
        pFile_1 = fopen(output_file_1, "a"); // opens new file for writing
        ensemble_m();
        for (j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.12e\t", m[j_S]);
        }
        fprintf(pFile_1, "\n");
        fclose(pFile_1);

        do
        {
            counter++;
            
            cutoff_local = find_change();
            
            // char append_string[128];
            // char *pos = append_string;
            // pos += sprintf(pos, "_count_%ld", counter);
            // save_spin_config(append_string, "w");

            pFile_1 = fopen(output_file_1, "a"); // opens new file for writing
            ensemble_m();
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m[j_S]);
            }
            fprintf(pFile_1, "%.12e\n", cutoff_local);
            printf("\r%.12e", cutoff_local);
            fflush(stdout);
            fclose(pFile_1);
        }
        while (cutoff_local > CUTOFF_SPIN); // 10^-14
        printf("\n");
        char append_string[128];
        char *pos = append_string;
        pos += sprintf(pos, "_count_%ld_h", counter);
        save_spin_config(append_string, "w");

        T=0.1;
        evolution_at_T(100);
        
        return 0;
    }

//===============================================================================//

//===============================================================================//
//====================      Main                             ====================//
    
    int free_memory()
    {
        // printf("..spin..");
        // usleep(1000);
        if (spin_reqd == 1)
        {
            free(spin);
        }
        // printf("..N_N_I..");
        // usleep(1000);
        if (N_N_I_reqd == 1)
        {
            free(N_N_I);
        }
        // printf("..black_white_checkerboard..");
        // usleep(1000);
        if (black_white_checkerboard_reqd == 1)
        {
            free(black_white_checkerboard);
            // free(black_white_checkerboard[0]);
            // free(black_white_checkerboard[1]);
        }
        // printf("..spin_bkp..");
        // usleep(1000);
        if (spin_bkp_reqd == 1)
        {
            free(spin_bkp);
        }
        // printf("..spin_temp..");
        // usleep(1000);
        if (spin_temp_reqd == 1)
        {
            free(spin_temp);
        }
        // printf("..h_random..");
        // usleep(1000);
        if (h_random_reqd == 1)
        {
            free(h_random);
        }
        // printf("..J_random..");
        // usleep(1000);
        if (J_random_reqd == 1)
        {
            free(J_random);
        }
        // printf("..cluster..");
        // usleep(1000);
        if (cluster_reqd == 1)
        {
            free(cluster);
        }
        // printf("..sorted_h_index..");
        // usleep(1000);
        if (sorted_h_index_reqd == 1)
        {
            free(sorted_h_index);
        }
        // printf("..next_in_queue..");
        // usleep(1000);
        if (next_in_queue_reqd == 1)
        {
            free(next_in_queue);
        }
        // printf("..spin_old..");
        // usleep(1000);
        if (spin_old_reqd == 1)
        {
            free(spin_old);
        }
        // printf("..spin_new..");
        // usleep(1000);
        if (spin_new_reqd == 1)
        {
            free(spin_new);
        }
        // printf("..field_site..");
        // usleep(1000);
        if (field_site_reqd == 1)
        {
            free(field_site);
        }
        #ifdef _OPENMP
        free(random_seed);
        #endif
        
        #ifdef enable_CUDA_CODE
        cudaFree(dev_spin);
        cudaFree(dev_spin_temp);
        cudaFree(dev_spin_bkp);
        cudaFree(dev_CUTOFF_SPIN);
        cudaFree(dev_J);
        cudaFree(dev_J_random);
        cudaFree(dev_h);
        cudaFree(dev_h_random);
        cudaFree(dev_N_N_I);
        cudaFree(dev_m);
        cudaFree(dev_m_bkp);
        cudaFree(dev_spin_reduce);
        #endif

        return 0;
    }
    
    int allocate_memory()
    {
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
            spin = (double*)malloc(dim_S*no_of_sites*sizeof(double));
            spin_reqd_local = 1;
        }
        else 
        {
            if (spin_reqd_local == 1 && spin_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(spin);
                    spin = (double*)malloc(dim_S*no_of_sites*sizeof(double));
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
            N_N_I = (long int*)malloc(2*dim_L*no_of_sites*sizeof(long int));
            N_N_I_reqd_local = 1;
        }
        else 
        {
            if (N_N_I_reqd_local == 1 && N_N_I_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(N_N_I);
                    N_N_I = (long int*)malloc(2*dim_L*no_of_sites*sizeof(long int));
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
        
        static int spin_bkp_reqd_local = 0;
        if (spin_bkp_reqd_local == 0 && spin_bkp_reqd == 1)
        {
            spin_bkp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
            spin_bkp_reqd_local = 1;
        }
        else 
        {
            if (spin_bkp_reqd_local == 1 && spin_bkp_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(spin_bkp);
                    spin_bkp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
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
            spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
            spin_temp_reqd_local = 1;
        }
        else 
        {
            if (spin_temp_reqd_local == 1 && spin_temp_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(spin_temp);
                    spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));
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
            J_random = (double*)malloc(2*dim_L*no_of_sites*sizeof(double));
            J_random_reqd_local = 1;
        }
        else 
        {
            if (J_random_reqd_local == 1 && J_random_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(J_random);
                    J_random = (double*)malloc(2*dim_L*no_of_sites*sizeof(double));
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
            cluster = (int*)malloc(no_of_sites*sizeof(int));
            cluster_reqd_local = 1;
        }
        else 
        {
            if (cluster_reqd_local == 1 && cluster_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(cluster);
                    cluster = (int*)malloc(no_of_sites*sizeof(int));
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
        
        static int sorted_h_index_reqd_local = 0;
        if (sorted_h_index_reqd_local == 0 && sorted_h_index_reqd == 1)
        {
            sorted_h_index = (long int*)malloc(no_of_sites*sizeof(long int));
            sorted_h_index_reqd_local = 1;
        }
        else 
        {
            if (sorted_h_index_reqd_local == 1 && sorted_h_index_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(sorted_h_index);
                    sorted_h_index = (long int*)malloc(no_of_sites*sizeof(long int));
                }
            }
            else 
            {
                if (sorted_h_index_reqd_local == 1 && sorted_h_index_reqd == 0)
                {
                    free(sorted_h_index);
                    sorted_h_index_reqd_local = 0;
                }
            }
        }

        static int next_in_queue_reqd_local = 0;
        if (next_in_queue_reqd_local == 0 && next_in_queue_reqd == 1)
        {
            next_in_queue = (long int*)malloc((1+no_of_sites)*sizeof(long int));
            next_in_queue_reqd_local = 1;
        }
        else 
        {
            if (next_in_queue_reqd_local == 1 && next_in_queue_reqd == 1)
            {
                if (free_and_allocate == 1)
                {
                    free(next_in_queue);
                    next_in_queue = (long int*)malloc((1+no_of_sites)*sizeof(long int));
                }
            }
            else 
            {
                if (next_in_queue_reqd_local == 1 && next_in_queue_reqd == 0)
                {
                    free(next_in_queue);
                    next_in_queue_reqd_local = 0;
                }
            }
        }
        
        
        static int spin_old_reqd_local = 0;
        if (spin_old_reqd_local == 0 && spin_old_reqd == 1)
        {
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
    
        return 0;

    }

    int for_cuda_parallelization()
    {
        printf("\nCUDA Active.\n");
        long int i, j;
        num_of_threads = omp_get_max_threads();
        num_of_procs = omp_get_num_procs();
        random_seed = (unsigned int*)malloc(cache_size*num_of_threads*sizeof(unsigned int));
        
        // use CUDA_RANDOM
        init_genrand64( (unsigned long long) rand() );
        
        printf("\nNo. of THREADS = %d\n", num_of_threads);
        printf("No. of PROCESSORS = %d\n", num_of_procs);
        for (i=0; i < num_of_threads; i++)
        {
            // random_seed[i] = rand_r(&random_seed[cache_size*(i-1)]);
            random_seed[i] = genrand64_int64();
        }
        double *start_time_loop = (double*)malloc((num_of_threads)*sizeof(double)); 
        double *end_time_loop = (double*)malloc((num_of_threads)*sizeof(double)); 
        
        #ifdef enable_CUDA_CODE
        cudaMalloc((void **)&dev_CUTOFF_SPIN, sizeof(double));
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
        cudaMalloc((void **)&dev_m, dim_S*sizeof(double));
        cudaMalloc((void **)&dev_m_bkp, dim_S*sizeof(double));
        cudaMalloc((void **)&dev_spin_reduce, dim_S*no_of_sites*sizeof(double));
        #endif
        
        #ifdef enable_CUDA_CODE
        no_of_sites_max_power_2 = 1;
        while (no_of_sites_max_power_2 < no_of_sites-1)
        {
            no_of_sites_max_power_2 = no_of_sites_max_power_2*2;
        }
        no_of_sites_remaining_power_2 = no_of_sites - no_of_sites_max_power_2 ;
        #endif

        // for (i=num_of_threads; i>=1; i++)
        // {
        //     start_time_loop[i-1] = omp_get_wtime();
        //     omp_set_num_threads(i);
        //     printf("\n\nNo. of THREADS = %ld \n\n", i);
        //     // field_cool_and_rotate_checkerboard(0, 1);
        //     r_order = 1;
        //     initialize_spin_config();
        //     for (j=0; j<100000; j++)
        //     {
        //         spin[rand()%(dim_S*no_of_sites)] = (double)rand()/(double)RAND_MAX;
        //         ensemble_m();
        //         // ensemble_E();
        //     }
            
        //     // random_initialize_and_rotate_checkerboard(0, 1);
        //     end_time_loop[i-1] = omp_get_wtime();
        // }
        
        // for (i=1; i<=num_of_threads; i++)
        // {
        //     printf("No. of THREADS = %ld ,\t Time elapsed = %g\n", i, end_time_loop[i-1]-start_time_loop[i-1]);
        // }

        free(start_time_loop);
        free(end_time_loop);

        return 0;
    }
    
    int for_omp_parallelization()
    {
        printf("\nOpenMP Active.\n");
        long int i, j;
        num_of_threads = omp_get_max_threads();
        num_of_procs = omp_get_num_procs();
        random_seed = (unsigned int*)malloc(cache_size*num_of_threads*sizeof(unsigned int));
        init_genrand64( (unsigned long long) rand() );
        
        printf("\nNo. of THREADS = %d\n", num_of_threads);
        printf("No. of PROCESSORS = %d\n", num_of_procs);
        for (i=0; i < num_of_threads; i++)
        {
            // random_seed[i] = rand_r(&random_seed[cache_size*(i-1)]);
            random_seed[i] = genrand64_int64();
        }
        double *start_time_loop = (double*)malloc((num_of_threads)*sizeof(double)); 
        double *end_time_loop = (double*)malloc((num_of_threads)*sizeof(double)); 
        
        // for (i=num_of_threads; i>=1; i++)
        // {
        //     start_time_loop[i-1] = omp_get_wtime();
        //     omp_set_num_threads(i);
        //     printf("\n\nNo. of THREADS = %ld \n\n", i);
        //     // field_cool_and_rotate_checkerboard(0, 1);
        //     r_order = 1;
        //     initialize_spin_config();
        //     for (j=0; j<100000; j++)
        //     {
        //         spin[rand()%(dim_S*no_of_sites)] = (double)rand()/(double)RAND_MAX;
        //         ensemble_m();
        //         // ensemble_E();
        //     }
            
        //     // random_initialize_and_rotate_checkerboard(0, 1);
        //     end_time_loop[i-1] = omp_get_wtime();
        // }
        
        // for (i=1; i<=num_of_threads; i++)
        // {
        //     printf("No. of THREADS = %ld ,\t Time elapsed = %g\n", i, end_time_loop[i-1]-start_time_loop[i-1]);
        // }

        free(start_time_loop);
        free(end_time_loop);

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

    int main()
    {
        srand(time(NULL));
        start_time = omp_get_wtime();
        printf("\n---- BEGIN ----\n");
        allocate_memory();
        
        int j_L, j_S;
        // no_of_sites = custom_int_pow(lattice_size, dim_L);
        
        initialize_nearest_neighbor_index();
        // printf("nearest neighbor initialized. \n");
        
        #if defined (UPDATE_CHKR_NON_EQ) || defined (UPDATE_CHKR_EQ_MC)
            initialize_checkerboard_sites();
        #endif
        
        #ifdef RANDOM_BOND
        load_J_config("");
        // printf("J loaded. \n");
        #endif
        
        #ifdef RANDOM_FIELD
        load_h_config("");
        // printf("h loaded. \n");
        #endif        
        
        long int i, j;
        
        // thermal_i = thermal_i*lattice_size[0];
        // average_j = average_j*lattice_size[0];
        
        printf("L = %d, dim_L = %d, dim_S = %d\n", lattice_size[0], dim_L, dim_S); 
        
        printf("hysteresis_MCS_multiplier = %ld, hysteresis_MCS_max = %ld\n", hysteresis_MCS_multiplier, hysteresis_MCS_max); 
        
        // srand(time(NULL));
        
        printf("RAND_MAX = %lf,\n sizeof(int) = %ld,\n sizeof(long) = %ld,\n sizeof(double) = %ld,\n sizeof(long int) = %ld,\n sizeof(short int) = %ld,\n sizeof(unsigned int) = %ld,\n sizeof(RAND_MAX) = %ld\n", (double)RAND_MAX, sizeof(int), sizeof(long), sizeof(double), sizeof(long int), sizeof(short int), sizeof(unsigned int), sizeof(RAND_MAX));
        
        #ifdef enable_CUDA_CODE
        for_cuda_parallelization();
        #else
            #ifdef _OPENMP
            for_omp_parallelization();
            #endif
        #endif
        
        // save_h_config("_test");
        // load_h_config("_test");
        // for (i=0; i<10; i++)
        // {
        //     printf("%x     ", h_random[2*i]);
        //     printf("%X     ", h_random[2*i]);
        //     printf("%.17e      ", h_random[2*i]);
        //     printf("%.17g\n", h_random[2*i]);
        // }
        
        
        int is_chkpt = 0;

        // double h_field_vals[] = { 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15 };
        // double h_field_vals[] = { 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.048, 0.050, 0.052, 0.054, 0.056, 0.058, 0.060, 0.064, 0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150 };
        // double h_field_vals[] = { 0.100, 0.500, 1.000, 0.800, 0.300, 2.000 };
        double h_field_vals[] = { 1.000 };
        int len_h_field_vals = sizeof(h_field_vals) / sizeof(h_field_vals[0]);
        for (i=0; i<len_h_field_vals; i++)
        {
            double start_time_loop[2];
            double end_time_loop[2];
            start_time_loop[0] = omp_get_wtime();
            // field_cool_and_rotate_checkerboard(0, 1);
            
            // is_chkpt = ordered_initialize_and_rotate_checkerboard(1, 1, h_field_vals[i]);
            
            sigma_h[0] = h_field_vals[i];
            sigma_h[1] = h_field_vals[i];
            load_h_config("");
            zero_temp_RFXY_hysteresis_axis_checkerboard(0, -1);
            // zero_temp_RFXY_hysteresis_axis_checkerboard(1, -1);
            // random_initialize_and_rotate_checkerboard(0, 1);
            end_time_loop[0] = omp_get_wtime();
            
            if (is_chkpt == -1)
            {
                printf("is_chkpt = %d\n", is_chkpt);
                printf("\nIncomplete rotating hysteresis starting from y ( |h|=%lf ) = %lf \n", h[1], end_time_loop[0] - start_time_loop[0] );
                break;
            }
            
            start_time_loop[1] = omp_get_wtime();
            // zero_temp_RFXY_hysteresis_axis_checkerboard(1, 1);
            // evolution_at_T(100);
            end_time_loop[1] = omp_get_wtime();
            
            // printf("\nCooling protocol time (from T=%lf to T=%lf) = %lf \n", Temp_max, Temp_min, end_time_loop[0] - start_time_loop[0] );
            // printf("\nRotating hysteresis starting from y ( |h|=%lf ) = %lf \n", h[1], end_time_loop[0] - start_time_loop[0] );
            printf("\nHysteresis along x ( Max(|sigma_h|)=%lf ) = %lf \n", sigma_h[0], end_time_loop[0] - start_time_loop[0] );
            // printf("\nHysteresis along y ( Max(|h|)=%lf ) = %lf \n", h_max+h_i_max, end_time_loop[1] - start_time_loop[1] );
            // printf("\nEvolution time (at T=%lf) = %lf \n", T, end_time_loop[1] - start_time_loop[1] );
        }
        
        // zero_temp_RFIM_hysteresis();
        // zero_temp_RFIM_ringdown();
        // zero_temp_RFIM_return_point_memory();
        
        /* 
        Gl_Me_Wo = 1;
        Ch_Ra_Li = 0;
        h_order = 0;
        r_order = 0;
        evo_diff_ini_config_temp();
        */
        
        // h[0] = 0.1;
        // Gl_Me_Wo = 2;
        // Ch_Ra_Li = 0;
        // h_order = 0;
        // r_order = 1;
        // zfc_zfh_or_both(2);
        // evolution_at_T(1);
        // evolution_at_T_h();
        // evo_diff_ini_config_temp();
        
        
        /* 
        Gl_Me_Wo = 1;
        Ch_Ra_Li = 0;
        h_order = 0;
        r_order = 1;
        zfc_zfh_or_both(2);
        */
        
        /* 
        Gl_Me_Wo = 0;
        Ch_Ra_Li = 1;
        h_order = 0;
        r_order = 0;
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
        */
        /* 
        Gl_Me_Wo = 0;
        Ch_Ra_Li = 0;
        h_order = 0;
        r_order = 0;
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
        } */
        
        
        
        free_memory();
        double end_time = omp_get_wtime();
        printf("\nCPU Time elapsed total = %lf \n", end_time-start_time);
        printf("\n----- END -----\n");
        // is_chkpt = -1;
        return -is_chkpt;
    }

//===============================================================================//

