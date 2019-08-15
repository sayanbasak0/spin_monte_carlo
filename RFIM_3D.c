
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


#define MARSAGLIA 1 // uncomment only one
// #define REJECTION 1 // uncomment only one
// #define BOX_MULLER 1 // uncomment only one

// Comment next line for newer compilers // gcc-6.3.0 or newer // intel-18.0.1.163 or newer
#define OLD_COMPILER 1 
// Uncomment ^it if OpenMP reduction has compile errors 

#define RANDOM_FIELD 1 // for random field disorder

// #define RANDOM_BOND 1 // for random bond disorder

#define dim_L 3 // 3D
#define dim_S 1 // Ising Model

#define UPDATE_CHKR_EQ_MC 1 // for checkerboard updates - parallelizable

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
    

//===============================================================================//
//====================      Lattice size                     ====================//
    int lattice_size[dim_L] = { 16, 16, 16 }; // lattice_size[dim_L]
    long int no_of_sites;
    long int no_of_black_sites;
    long int no_of_white_sites;
    long int no_of_black_white_sites[2];

//====================      Checkerboard variables           ====================//
    long int *black_white_checkerboard; 
    #if defined (UPDATE_CHKR_EQ_MC)
        int black_white_checkerboard_reqd = 1;
    #else
        int black_white_checkerboard_reqd = 0;
    #endif
    int site_to_dir_index[dim_L];


//====================      Near neighbor /Boundary Cond     ====================//
    long int *N_N_I; int N_N_I_reqd = 1;
    double BC[dim_L] = { 1, 1, 1 }; // 1 -> Periodic | 0 -> Open | -1 -> Anti-periodic -- Boundary Condition

//====================      Spin variable                    ====================//
    double *spin; int spin_reqd = 1;

//====================      Initialization type              ====================//
    double order[dim_S] = { 1.0 }; //  Initialize all spins along this direction
    int h_order = 0; // 0/1 Initialize spin along RF direction
    int r_order = 1; // 0/1 Random initialize spin

//====================      MC-update type                   ====================//
    int Gl_Me = 1; // 0 -> Glauber , 1 -> Metropolis
    char G_M[] = "GM"; 
    int Ch_Ra_Li = 0; // 0 -> Checkerboard updates , 1 -> Random updates , 2 -> Linear updates
    char C_R_L[] = "CRL"; 

//====================      NN-interaction (J)               ====================//
    double J[dim_L] = { 1.0, 1.0, 1.0 }; 
    double sigma_J[dim_L] = { 0.0, 0.0, 0.0 };
    double *J_random;
    #ifdef RANDOM_BOND
        int J_random_reqd = 1;
    #else
        int J_random_reqd = 0;
    #endif
    double J_i_max = 0.0; double J_i_min = 0.0;
    double J_dev_net[dim_L];
    double J_dev_avg[dim_L];

//====================      on-site field (h)                ====================//
    double h[dim_S] = { 0.0 }; // uniform field 
    double sigma_h[dim_S] = { 2.00 }; // random field strength
    double *h_random;
    #ifdef RANDOM_FIELD
        int h_random_reqd = 1;
    #else
        int h_random_reqd = 0;
    #endif
    double h_max = 2.0*dim_L+0.5; double h_min = -2.0*dim_L-0.5;
    double h_i_max = 0.0; double h_i_min = 0.0;
    double del_h = 0.01; 
    double Delta_h = 0.5; 
    double h_dev_net[dim_S];
    double h_dev_avg[dim_S];

//====================      Temperature (T)                  ====================//
    double T = 1.00;
    double Temp_min = 0.01;
    double Temp_max = 2.01;
    double delta_T = 0.1;

//====================      Magnetisation <M>                ====================//
    double m[dim_S];
    
    double m_sum[dim_S];
    double m_avg[dim_S];

    double m_abs_sum = 0;
    double m_abs_avg = 0;
    double m_2_sum = 0;
    double m_2_avg = 0;

    double m_4_sum = 0;
    double m_4_avg = 0;

//====================      Energy <E>                       ====================//
    double E = 0;
    double E_sum = 0;
    double E_avg = 0;
    double E_2_sum = 0;
    double E_2_avg = 0;

//====================      Specific heat (Cv)               ====================//
    double Cv = 0;

//====================      Susceptibility (X)               ====================//
    double X = 0;

//====================      Binder Parameter (B)             ====================//
    double B = 0;

//====================      MC-update iterations             ====================//
    long int thermal_i = 128*100; // thermalizing MCS
    long int average_j = 128*10; // no. of measurements
    int sampling_inter = 16; // random no. of MCS before taking each measurement 

//====================      Hysteresis                       ====================//
    long int hysteresis_MCS = 1; // no. of averaging MCS for hysteresis protocol with field
    int hysteresis_repeat = 2; // No. of times hysteresis loop is repeated

//===============================================================================//

//===============================================================================//
//====================      Functions                        ====================//

    long int custom_int_pow(long int base, int power);

    double custom_double_pow(double base, int power);

    long int nearest_neighbor(long int xyzi, int j_L, int k_L);

    int direction_index(long int xyzi);

    double generate_gaussian();

    int initialize_h_zero();

    int initialize_h_random_gaussian();

    int initialize_J_zero();

    int initialize_J_random_gaussian();

    int initialize_nearest_neighbor_index();

    int initialize_checkerboard_sites();

    int initialize_ordered_spin_config();

    int initialize_h_ordered_spin_config();

    int initialize_random_spin_config();

    int initialize_spin_config();

    int save_spin_config(char append_string[], char write_mode[]);

    int save_h_config(char append_string[]);

    int save_J_config(char append_string[]);

    int load_spin_config(char append_string[]);

    int load_h_config(char append_string[]);

    int load_J_config(char append_string[]);

    int print_header_column(char output_file_name[]);

    int print_to_file();

    int set_sum_of_moment_m_0();

    int ensemble_m();

    int sum_of_moment_m();

    int average_of_moment_m(double MCS_counter);

    int set_sum_of_moment_m_abs_0();

    int sum_of_moment_m_abs();

    int average_of_moment_m_abs(double MCS_counter);

    int set_sum_of_moment_m_2_0();

    int sum_of_moment_m_2();

    int average_of_moment_m_2(double MCS_counter);

    int set_sum_of_moment_m_4_0();

    int sum_of_moment_m_4();

    int average_of_moment_m_4(double MCS_counter);

    int set_sum_of_moment_m_higher_0();

    int sum_of_moment_m_higher();

    int average_of_moment_m_higher(double MCS_counter);

    int set_sum_of_moment_m_all_0();

    int sum_of_moment_m_all();

    int average_of_moment_m_all(double MCS_counter);

    int set_sum_of_moment_B_0();

    int ensemble_B();

    int sum_of_moment_B();

    int average_of_moment_B(double MCS_counter);

    int set_sum_of_moment_X_0();

    int ensemble_X();

    int sum_of_moment_X();

    int average_of_moment_X(double MCS_counter);

    int set_sum_of_moment_E_0();

    int ensemble_E();

    int sum_of_moment_E();

    int average_of_moment_E(double MCS_counter);

    int set_sum_of_moment_E_2_0();

    int sum_of_moment_E_2();

    int average_of_moment_E_2(double MCS_counter);

    int set_sum_of_moment_Cv_0();

    int ensemble_Cv();

    int sum_of_moment_Cv();

    int average_of_moment_Cv(double MCS_counter);

    int set_sum_of_moment_all_0();

    int ensemble_all();

    int sum_of_moment_all();

    int average_of_moment_all(double MCS_counter);

    int update_spin_single(long int xyzi, double* __restrict__ spin_local);

    double Energy_old(long int xyzi, double* __restrict__ spin_local, double* __restrict__ field_local);

    double Energy_new(long int xyzi, double* __restrict__ spin_local, double* __restrict__ field_local);

    double update_probability_Metropolis(long int xyzi);

    int linear_Metropolis_sweep(long int iter);

    int random_Metropolis_sweep(long int iter);

    int checkerboard_Metropolis_sweep(long int iter);

    double update_probability_Glauber(long int xyzi);

    int linear_Glauber_sweep(long int iter);
    
    int random_Glauber_sweep(long int iter);

    int checkerboard_Glauber_sweep(long int iter);

    int Monte_Carlo_Sweep(long int sweeps);

    int thermalizing_iteration(long int thermal_iter);

    int averaging_iteration(long int average_iter, int inter);

    int evolution_at_T_h(int repeat_at_same_T_h);

    int evo_diff_ini_spin_at_T_h(int jj_S);
    
    int cooling_protocol(char output_file_name[]);
    
    int heating_protocol(char output_file_name[]);
    
    int fc_fh_or_both(int c_h_ch_hc);
    
    int hysteresis_protocol(int jj_S, double order_start);

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
        #endif

        return 0;
    }

    int initialize_J_zero()
    {
        long int i;
        for(i=0; i<2*dim_L*no_of_sites; i=i+1)
        {
            J_random[i] = 0; 
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
        
        #pragma omp parallel for private(j_L,k_L) 
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
        
        int black_white[2] = { 0, 1 };

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
                black_white_checkerboard[0+black_white_index[0]] = i;
                black_white_index[0]++;
            }
            else //if (dir_index_sum % 2 == black_white[1])
            {
                black_white_checkerboard[no_of_black_white_sites[0]+black_white_index[1]] = i;
                black_white_index[1]++;
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
                    #ifdef _OPENMP
                    spin[dim_S*i+j_S] = (-1.0 + 2.0 * (double)rand_r(&random_seed[cache_size*omp_get_thread_num()])/(double)(RAND_MAX));
                    #else
                    spin[dim_S*i+j_S] = (-1.0 + 2.0 * (double)rand()/(double)(RAND_MAX));
                    #endif
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
        
        h_order = 0;
        r_order = 1;
        return 0;
    }

    int initialize_spin_config()
    {
        int j_S;
        printf("Initialize Spin-> ");
        fflush(stdout);
        if (r_order == 1)
        {
            initialize_random_spin_config();
            printf("Random. ");

        }
        else
        {
            if (h_order == 1)
            {
                initialize_h_ordered_spin_config();
                printf("Along h_i. ");
            }
            else
            {

                initialize_ordered_spin_config();
                printf("To ");
                printf("{%.1f", order[0]);
                for(j_S=1; j_S<dim_S; j_S++)
                {
                    printf(",%.1f", order[j_S]);
                }
                printf("}.");
            }
        }
        ensemble_all();
        printf(" M=(%.3f", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%.3f", m[j_S]);
        }
        printf("), E=%.3f. ", E);
        // printf("\n");
        fflush(stdout);
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

        //-------------------------------------Check--------------------------------------------------//
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

        //-------------------------------------Check--------------------------------------------------//
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

        //-------------------------------------Check--------------------------------------------------//
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

        //-------------------------------------Check--------------------------------------------------//
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

    int load_h_config(char append_string[])
    {
        
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
            initialize_h_random_gaussian();

            save_h_config(append_string); // creates file for later
            printf("Initialized h_random config. Output file name: %s\n", input_file_1);
            
        }
        else
        {
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
        printf("h_i_min=%le, ", h_i_min);
        printf("h_i_max=%le, ", h_i_max);
        printf("h={%le,", h[0]);
        for (j_S=1; j_S<dim_S; j_S++)
        {
            printf("%le", h[j_S]);
        }
        printf("}, ");
        printf("sigma_h={%le,", sigma_h[0]);
        for (j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%le", sigma_h[j_S]);
        }
        printf("}, ");
        printf("h_dev_avg[j_S]={%le", h_dev_avg[0]);
        for (j_S=1; j_S<dim_S; j_S++)
        {
            printf("\nh_dev_avg[j_S]=%le", h_dev_avg[j_S]);
        }
        printf("}. \n");
        //-------------------------------------Check--------------------------------------------------//
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
            initialize_J_random_gaussian();

            save_J_config(append_string); // creates file for later
            printf("Initialized J_random config. Output file name: %s\n", input_file_1);
        }
        else
        {
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
        printf("J_i_min=%le, ", J_i_min);
        printf("J_i_max=%le, ", J_i_max);
        printf("J={%le,", J[0]);
        for (j_L=1; j_L<dim_L; j_L++)
        {
            printf("%le", J[j_L]);
        }
        printf("}, ");
        printf("sigma_J={%le,", sigma_J[0]);
        for (j_L=1; j_L<dim_L; j_L++)
        {
            printf(",%le", sigma_J[j_L]);
        }
        printf("}, ");
        printf("J_dev_avg[j_S]={%le", J_dev_avg[0]);
        for (j_L=1; j_L<dim_L; j_L++)
        {
            printf("\nJ_dev_avg[j_S]=%le", J_dev_avg[j_L]);
        }
        printf("}. \n");
        //-------------------------------------Check--------------------------------------------------//
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

//====================      Print Output                     ====================//

    int print_header_column(char output_file_name[])
    {
        int j_L, j_S;
        printf("\nOutput file name: %s\n", output_file_name);
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

        fprintf(pFile_output, "==================================================================================\n");

        
        fclose(pFile_output);
        return 0;
    }

    int print_to_file()
    {


        return 0;
    }

//====================      Magnetisation                    ====================//

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

    int set_sum_of_moment_m_higher_0()
    {
        int j_S, j_SS;

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
            m_2_persite += m[j_S] * m[j_S];
        }

        m_2_sum += m_2_persite;
        m_4_sum += m_2_persite * m_2_persite;

        return 0;
    }

    int average_of_moment_m_higher(double MCS_counter)
    {
        int j_S, j_SS, j_L;
        
        m_2_avg = m_2_sum / MCS_counter;
        m_4_avg = m_4_sum / MCS_counter;
        
        return 0;
    }

    int set_sum_of_moment_m_all_0()
    {
        int j_S, j_SS;
        
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

        m_abs_avg = m_abs_sum / MCS_counter;
        m_2_avg = m_2_sum / MCS_counter;
        m_4_avg = m_4_sum / MCS_counter;
        
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

//====================      All moments                      ====================//

    int set_sum_of_moment_all_0()
    {
        set_sum_of_moment_m_0();
        set_sum_of_moment_m_abs_0();
        set_sum_of_moment_m_2_0();
        set_sum_of_moment_m_4_0();
        set_sum_of_moment_E_0();
        set_sum_of_moment_E_2_0();

        return 0;
    }

    int ensemble_all()
    {
        ensemble_m();
        ensemble_E();

        return 0;
    }

    int sum_of_moment_all()
    {
        sum_of_moment_m();
        sum_of_moment_m_abs();
        sum_of_moment_m_2();
        sum_of_moment_m_4();
        sum_of_moment_E();
        sum_of_moment_E_2();
        
        return 0;
    }

    int average_of_moment_all(double MCS_counter)
    {
        average_of_moment_m(MCS_counter);
        average_of_moment_m_abs(MCS_counter);
        average_of_moment_m_2(MCS_counter);
        average_of_moment_m_4(MCS_counter);
        average_of_moment_E(MCS_counter);
        average_of_moment_E_2(MCS_counter);
        
        Cv = (E_2_avg - (E_avg * E_avg)) / (T * T);

        X = (m_2_avg - (m_abs_avg * m_abs_avg)) / T;

        B = (1.0 / 2.0) * ( 3.0 - ( m_4_avg / (m_2_avg * m_2_avg) ) );  
        
        return 0;
    }

//====================      MonteCarlo-tools                 ====================//
    
    int update_spin_single(long int xyzi, double* __restrict__ spin_local)
    {
        int j_S;
        for (j_S=0; j_S<dim_S; j_S++)
        {
            spin[dim_S*xyzi + j_S] = spin_local[j_S];
        }

        return 0;
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
                #ifdef _OPENMP
                spin_local[j_S] = (1.0 - 2.0 * (double)rand_r(&random_seed[cache_size*omp_get_thread_num()])/(double)(RAND_MAX));
                #else
                spin_local[j_S] = (1.0 - 2.0 * (double)rand()/(double)(RAND_MAX));
                #endif
                s_mod = s_mod + spin_local[j_S] * spin_local[j_S];
            }
        }
        while(s_mod >= 1 || s_mod <= limit);
        s_mod = sqrt(s_mod);
        
        
        for(j_S=0; j_S<dim_S; j_S++)
        {
            spin_local[j_S] = spin_local[j_S] / s_mod;
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
        #ifdef _OPENMP
        double r = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()])/ (double) RAND_MAX;
        #else
        double r = (double) rand()/ (double) RAND_MAX;
        #endif
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
            #ifdef _OPENMP
            xyzi = rand_r(&random_seed[cache_size*omp_get_thread_num()])%no_of_sites;
            #else
            xyzi = rand()%no_of_sites;
            #endif
            double update_prob = update_probability_Metropolis(xyzi);

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
        #ifdef _OPENMP
        double r = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()])/ (double) RAND_MAX;
        #else
        double r = (double) rand()/ (double) RAND_MAX;
        #endif
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
            #ifdef _OPENMP
            xyzi = rand_r(&random_seed[cache_size*omp_get_thread_num()])%no_of_sites;
            #else
            xyzi = rand()%no_of_sites;
            #endif
            double update_prob = update_probability_Glauber(xyzi);

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

                double update_prob = update_probability_Glauber(site_index);
            }

            black_or_white = !black_or_white;
            iter--;
        } 
        
        return 0;
    }

//====================      MonteCarlo-Sweep                 ====================//

    int Monte_Carlo_Sweep(long int sweeps)
    {
        if (Gl_Me == 0)
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
            if (Gl_Me == 1)
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
        }
        return 0;
    }

//====================      T!=0                             ====================//

    int thermalizing_iteration(long int thermal_iter)
    {
        int j_S;
        printf("Thermalizing.. ");
        
        fflush(stdout);
        Monte_Carlo_Sweep(thermal_iter);
        printf("Done.");
        fflush(stdout);
        ensemble_all();
        // ensemble_E();
        // ensemble_m();
        printf(" M=(%.3f", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%.3f", m[j_S]);
        }
        printf("), E=%.3f. ", E);
        fflush(stdout);
        return 0;
    }

    int averaging_iteration(long int average_iter, int inter)
    {
        double MCS_counter = 0;
        int j_S, j_SS, j_L;
        
        set_sum_of_moment_all_0();
        // set_sum_of_moment_m_0();
        // set_sum_of_moment_E_0();

        printf("Averaging.. ");
        fflush(stdout);
        while(average_iter)
        {
            Monte_Carlo_Sweep(inter-genrand64_int64()%inter);

            ensemble_all();
            // ensemble_m();
            // ensemble_E();
            sum_of_moment_all();
            // sum_of_moment_m();
            // sum_of_moment_E();

            MCS_counter = MCS_counter + 1;
            
            average_iter = average_iter - 1;
        }
        printf("Done.");
        fflush(stdout);
        average_of_moment_all(MCS_counter);
        // average_of_moment_m(MCS_counter);
        // average_of_moment_E(MCS_counter);

        printf(" <M>=(%.3f", m_avg[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%.3f", m_avg[j_S]);
        }
        printf("), <E>=%.3f. ", E_avg);
        
        fflush(stdout);
        return 0;
    }

    int evolution_at_T_h(int repeat_at_same_T_h)
    {
        printf("\n__________________________________________________________\n");
        int j_S, j_SS, j_L;
        printf("\nEvolve system with (thermalizing steps+averaging steps)*n=(%ld+%ld)*%d at T=%lf, h={%lf", thermal_i, average_j, repeat_at_same_T_h, T, h[0]);
        for (j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", h[j_S]);
        }
        printf("}..\n");
        

        // create file name and pointer. 
        {

            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_evo(t)_at_T=(%.3f)_%c_%c_", dim_S, dim_L, T, G_M[Gl_Me], C_R_L[Ch_Ra_Li]);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }

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
        printf("\n");
        initialize_spin_config();
        printf("\n");
        int i;
        for (i=0; i<repeat_at_same_T_h; i++)
        {
            printf("\rStep-%d : ", i+1);
            thermalizing_iteration(thermal_i);
            averaging_iteration(average_j, sampling_inter);
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
        
        printf("\n__________________________________________________________\n");
        return 0;
    }


    int evo_diff_ini_spin_at_T_h(int jj_S)
    {
        printf("\n__________________________________________________________\n");
        printf("\nInitialize and Evolve system with (thermalizing steps+averaging steps)=(%ld+%ld) at every T and h set by T_min,T_max,delta_T,h_min,h_max,Delta_h", thermal_i, average_j);
        int j_S, j_L, j_SS;

        printf("\n");

        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_%c_%c_", dim_S, dim_L, G_M[Gl_Me], C_R_L[Ch_Ra_Li]);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_(%.3f)", T);

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
        if( access( output_file_0, F_OK ) == -1 )
        {
            print_header_column(output_file_0);
            pFile_1 = fopen(output_file_0, "a");
            {
                fprintf(pFile_1, "T\t");
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "<h[%d]>\t", j_S);
                }
                fprintf(pFile_1, "<|m|>\t");
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "<m[%d]>\t", j_S);
                }

                fprintf(pFile_1, "<E>\t");
                fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
            }
            fclose(pFile_1);
        }
        else
        {
            printf("\nAppending to file: %s \n", output_file_0);
        }
        printf("\n");
        for (T=Temp_min; T<=Temp_max; T=T+delta_T)
        {
            for (h[jj_S]=h_min; h[jj_S]<=h_max; h[jj_S]+=Delta_h)
            {
                printf("\r");
                printf("T=%.2f,h={%.2f", T, h[0]);
                for (j_S=1; j_S<dim_S; j_S++)
                {
                    printf(",%.2f", h[j_S]);
                }
                printf("} : ");

                initialize_spin_config();

                thermalizing_iteration(thermal_i);
                averaging_iteration(average_j, sampling_inter);


                pFile_1 = fopen(output_file_0, "a");
                fprintf(pFile_1, "%.12e\t", T);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", h[j_S]);
                }
                fprintf(pFile_1, "%.12e\t", m_abs_avg);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%.12e\t", m_avg[j_S]);
                }

                fprintf(pFile_1, "%.12e\t", E_avg);

                fprintf(pFile_1, "\n");
                fclose(pFile_1);
            }
        }

        printf("\n__________________________________________________________\n");
        return 0;
    }

    int cooling_protocol(char output_file_name[])
    {
        #ifdef _OPENMP
            omp_set_num_threads(num_of_threads);
        #endif

        
        int j_S, j_SS, j_L;

        ensemble_all();
        // ensemble_E();
        // ensemble_m();

        printf("\n-------------Cooling-------------\n");
        printf("Initial Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);

        pFile_1 = fopen(output_file_name, "a");
        fprintf(pFile_1, "Cooling... \n");
        fclose(pFile_1);

        for (T=Temp_max; T>Temp_min; T=T-delta_T)
        {
            printf("\r");
            printf("T=%.2f", T);
            printf(" : ");

            thermalizing_iteration(thermal_i);
            averaging_iteration(average_j, sampling_inter);

            pFile_1 = fopen(output_file_name, "a");

            fprintf(pFile_1, "%.12e\t", T);

            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m_avg[j_S]);
            }

            fprintf(pFile_1, "%.12e\t", m_2_avg);
            fprintf(pFile_1, "%.12e\t", m_4_avg);

            fprintf(pFile_1, "%.12e\t", E_avg);    
            fprintf(pFile_1, "\n");
            fclose(pFile_1);
        }

        ensemble_all();
        // ensemble_E();
        // ensemble_m();

        printf("\nFinal Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf. \n", E);
       
        
        return 0;
    }

    int heating_protocol(char output_file_name[])
    {
        #ifdef _OPENMP
            omp_set_num_threads(num_of_threads);
        #endif
        

        int j_S, j_SS, j_L;

        ensemble_all();
        // ensemble_E();
        // ensemble_m();

        printf("\n-------------Heating-------------\n");
        printf("Initial Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);

        pFile_1 = fopen(output_file_name, "a");
        fprintf(pFile_1, "Heating... \n");
        fclose(pFile_1);
        
        for (T=Temp_min; T<=Temp_max; T=T+delta_T)
        {
            printf("\r");
            printf("T=%.2f", T);
            printf(" : ");
            thermalizing_iteration(thermal_i);
            averaging_iteration(average_j, sampling_inter);

            pFile_1 = fopen(output_file_name, "a");

            fprintf(pFile_1, "%.12e\t", T);
            fprintf(pFile_1, "%.12e\t", m_abs_avg);

            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%.12e\t", m_avg[j_S]);
            }

            fprintf(pFile_1, "%.12e\t", E_avg);
            fprintf(pFile_1, "\n");
            fclose(pFile_1);
        }

        ensemble_all();
        // ensemble_E();
        // ensemble_m();

        printf("\nFinal Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf. \n", E);
        
        return 0;
    }

    int fc_fh_or_both(int c_h_ch_hc)
    {
        printf("\n__________________________________________________________\n");
        int j_S, j_SS, j_L;
        printf("\nField Cooling/Heating with (thermalizing steps+averaging steps)/delta_T=(%ld+%ld)/%lf at h={%lf", thermal_i, average_j, delta_T, h[0]);
        for (j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", h[j_S]);
        }
        printf("}..\n");


        if (c_h_ch_hc == 0 || c_h_ch_hc == 2)
        {
            T = Temp_max;
        }
        if (c_h_ch_hc == 1 || c_h_ch_hc == 3)
        {
            T = Temp_min;
        }
        

        // create file name and pointer. 
        {

            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_FC-FH_%c_%c_", dim_S, dim_L, G_M[Gl_Me], C_R_L[Ch_Ra_Li]);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_(%.3f<->%.3f)-[%.3f]", Temp_min, Temp_max, delta_T);

            pos += sprintf(pos, "_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%.3f", h[j_S]);
            }
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
        print_header_column(output_file_0);
        pFile_1 = fopen(output_file_0, "a");
        {
            fprintf(pFile_1, "T\t|m|\t");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t", j_S);
            } 

            fprintf(pFile_1, "<E>\t");
            fprintf(pFile_1, "\n----------------------------------------------------------------------------------\n");
        }
        fclose(pFile_1);
        
        printf("\n");
        initialize_spin_config();
        printf("\n");
        int i;
        for (i=0; i<hysteresis_repeat; i++)
        {
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
        }

        printf("\n__________________________________________________________\n");
        return 0;
    }

    int hysteresis_protocol(int jj_S, double order_start)
    {
        printf("\n__________________________________________________________\n");
        printf("\nHysteresis looping %d-times at T=%lf.. with MCS/delta_field = %ld/%lf..\n", hysteresis_repeat, T, hysteresis_MCS, del_h);

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
        
        // create file name and pointer. 
        {
            // char output_file_0[256];
            char *pos = output_file_0;
            pos += sprintf(pos, "O(%d)_%dD_hysteresis_%c_%c_", dim_S, dim_L, G_M[Gl_Me], C_R_L[Ch_Ra_Li]);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_(%.3f)", T);

            pos += sprintf(pos, "_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                if (j_S==jj_S)
                {
                    pos += sprintf(pos, "(%.3f<->%.3f)-[%.3f]", h_start, h_end, delta_h);
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
        printf("\n");
        initialize_spin_config();
        printf("\n");
        for (i=0; i<hysteresis_repeat; i=i+1)
        {
            printf("\n-------------loop %d-------------", i+1);
            printf("\nh = %lf --> %lf \n", h_start, h_end);
            for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
            {
                printf("\r");
                printf("h={%.2f", h[0]);
                for (j_S=1; j_S<dim_S; j_S++)
                {
                    printf(",%.2f", h[j_S]);
                }
                printf("} : ");
                averaging_iteration(hysteresis_MCS, 1);

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
            printf("\n");
            printf("\nh = %lf <-- %lf \n", h_start, h_end);
            for (h[jj_S] = h_end; order[jj_S] * h[jj_S] <= order[jj_S] * h_start; h[jj_S] = h[jj_S] + order[jj_S] * delta_h)
            {
                printf("\r");
                printf("h={%.2f", h[0]);
                for (j_S=1; j_S<dim_S; j_S++)
                {
                    printf(",%.2f", h[j_S]);
                }
                printf("} : ");
                averaging_iteration(hysteresis_MCS, 1);

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
            printf("\n");
            h[jj_S] = 0;
        }

        printf("\n__________________________________________________________\n");
        return 0;
    }

//===============================================================================//

//===============================================================================//
//====================      Main                             ====================//
    
    int free_memory()
    {

        if (spin_reqd == 1)
        {
            free(spin);
        }

        if (N_N_I_reqd == 1)
        {
            free(N_N_I);
        }

        if (black_white_checkerboard_reqd == 1)
        {
            free(black_white_checkerboard);
        }

        if (h_random_reqd == 1)
        {
            free(h_random);
        }

        if (J_random_reqd == 1)
        {
            free(J_random);
        }

        #ifdef _OPENMP
        free(random_seed);
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
                    
                    free(black_white_checkerboard);
                    black_white_checkerboard = (long int*)malloc(2*no_of_black_white_sites[0]*sizeof(long int));
                }
            }
            else 
            {
                if (black_white_checkerboard_reqd_local == 1 && black_white_checkerboard_reqd == 0)
                {
                    free(black_white_checkerboard);
                    black_white_checkerboard_reqd_local = 0;
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
    
        return 0;

    }
    
    #ifdef _OPENMP
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
            random_seed[i] = genrand64_int64();
        }

        return 0;
    }
    #endif

    int main()
    {
        srand(time(NULL));
        #ifdef _OPENMP
        start_time = omp_get_wtime();
        #else
        time_t time_now;
        time(&time_now);
        start_time = (double)time_now;
        #endif
        printf("\n---- BEGIN ----\n");
                
        printf("RAND_MAX = %lf,\n sizeof(int) = %ld,\n sizeof(long) = %ld,\n sizeof(double) = %ld,\n sizeof(long int) = %ld,\n sizeof(short int) = %ld,\n sizeof(unsigned int) = %ld,\n sizeof(RAND_MAX) = %ld\n", (double)RAND_MAX, sizeof(int), sizeof(long), sizeof(double), sizeof(long int), sizeof(short int), sizeof(unsigned int), sizeof(RAND_MAX));
        
        allocate_memory();
        
        int j_L, j_S;
        
        initialize_nearest_neighbor_index();
        
        #if defined (UPDATE_CHKR_EQ_MC)
            initialize_checkerboard_sites();
        #endif
        
        #ifdef RANDOM_BOND
        load_J_config("");
        #endif
        
        #ifdef RANDOM_FIELD
        load_h_config("");
        #endif        
        
        long int i, j;
        
        printf("L = %d, dim_L = %d, dim_S = %d\n", lattice_size[0], dim_L, dim_S); 
        
        #ifdef _OPENMP
        for_omp_parallelization();
        #endif
        
        
        // repeat thermalization and averaging n-times
        evolution_at_T_h(/*n=*/2000); // argument : n(=1)-> repeat n-times
        
        // argument 'n'th direction of field
        // evo_diff_ini_spin_at_T_h(/*n=*/0); 
        
        // arg=0-> cooling: Temp_max -> Temp_min or Temp_min -> Temp_max 
        // arg=1-> heating: Temp_min -> Temp_max or Temp_min -> Temp_max 
        // arg=2-> cooling then heating: Temp_max -> Temp_min -> Temp_max 
        // arg=3-> heating then cooling: Temp_min -> Temp_max -> Temp_min 
        // fc_fh_or_both(/*arg=*/2);  

        // hysteresis along 'n'th spin direction 
        // initialized from ordered spin configuration: 'start' = +1 or -1
        // hysteresis_protocol(/*n=*/0, /*start=*/1.0);

        
        
        free_memory();

        double end_time;
        #ifdef _OPENMP
        end_time = omp_get_wtime();
        #else
        time(&time_now);
        end_time = (double)time_now ;
        #endif

        printf("\nCPU Time elapsed total = %lf seconds \n", (end_time-start_time) );

        printf("\n----- END -----\n");

        return 0;
    }

//===============================================================================//

