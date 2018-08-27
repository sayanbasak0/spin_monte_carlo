// using bitbucket
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define dim_L 2
#define dim_S 2

FILE *pFile_1;
char output_file_1[256];

const double pie = 3.141592625359;
double k_B = 1;

unsigned int *random_seed;
int num_of_threads;
int num_of_procs;
int cache_size=512;
double CUTOFF = 0.0000000001;
// long int CHUNK_SIZE = 256; 

//===============================================================================//
//====================      Variables                        ====================//
//===============================================================================//
//====================      Lattice size                     ====================//
    int lattice_size[dim_L] = { 128, 128 }; // lattice_size[dim_L]
    long int no_of_sites;
    long int no_of_black_sites;
    long int no_of_white_sites;
    long int no_of_black_white_sites[2];

//====================      Checkerboard variables           ====================//
    long int *black_white_checkerboard[2];
    long int *black_checkerboard;
    long int *white_checkerboard;
    int site_to_dir_index[dim_L];

//====================      Ising hysteresis sorted list     ====================//
    long int *sorted_h_index;
    long int *next_in_queue;
    long int remaining_sites;

//====================      Wolff/Cluster variables          ====================//
    double reflection_plane[dim_S];
    int *cluster;

//====================      Near neighbor /Boundary Cond     ====================//
    long int *N_N_I;
    double BC[dim_L] = { 1, 1 }; // 1 -> Periodic | 0 -> Open | -1 -> Anti-periodic -- Boundary Condition

//====================      Spin variable                    ====================//
    double *spin;
    double *spin_temp;
    double spin_0[dim_S];
    double *spin_old;
    double *spin_new;
    // int *spin_sum;

//====================      Initialization type              ====================//
    double order[dim_S] = { 1.0, 0.0 }; // order[dim_S]
    int h_order = 0; // 0/1
    int r_order = 0; // 0/1

//====================      MC-update type                   ====================//
    int Gl_Me_Wo = 1;
    char G_M_W[] = "GMW";
    int Ch_Ra_Li = 0;
    char C_R_L[] = "CRL";

//====================      NN-interaction (J)               ====================//
    double J[dim_L] = { 1.0, 1.0 }; 
    double sigma_J[dim_L] = { 0.0, 0.0 };
    double *J_random;
    double J_max = 0.0;
    double J_min = 0.0;
    double delta_J = 0.01, J_i_max = 0.0, J_i_min = 0.0; // for hysteresis
    double J_dev_net[dim_L];
    double J_dev_avg[dim_L];

//====================      on-site field (h)                ====================//
    double h[dim_S] = { 0.0, 0.0 }; // h[0] = 0.1; // h[dim_S]
    double sigma_h[dim_S] = { 0.50, 0.00 }; 
    double *h_random;
    double h_max = 4.01;
    double h_min = -4.01;
    double delta_h = 0.01, h_i_max = 0.0, h_i_min = 0.0; // for hysteresis
    double h_dev_net[dim_S];
    double h_dev_avg[dim_S];
    double *field_site; // field experienced by spin due to nearest neighbors and on-site field

//====================      Temperature                      ====================//
    double T = 3.0;
    double Temp_min = 0.0;
    double Temp_max = 2.0;
    double delta_T = 0.1;

//====================      Magnetisation <M>                ====================//
    double m[dim_S];
    double abs_m[dim_S];
    double m_sum[dim_S];
    double m_avg[dim_S];
    double m_abs_sum = 0;
    double m_abs_avg = 0;
    double m_2_sum = 0;
    double m_2_avg = 0;
    double m_4_sum = 0;
    double m_4_avg = 0;
    double m_ab[dim_S*dim_S] = { 0 };
    double m_ab_sum[dim_S*dim_S] = { 0 };
    double m_ab_avg[dim_S*dim_S] = { 0 };

//====================      Energy <E>                       ====================//
    double E = 0;
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

//====================      MC-update iterations             ====================//
    long int thermal_i = 1*10; // *=lattice_size
    long int average_j = 1; // *=lattice_size

//====================      Hysteresis T!=0                  ====================//
    long int hysteresis_MCS = 1; 
    long int hysteresis_MCS_min = 1; 
    long int hysteresis_MCS_max = 100;
    int hysteresis_repeat = 1;
    long int hysteresis_MCS_multiplier = 10;


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
        double U1, U2, W, mult;
        static double X1, X2;
        static int call = 0;
        double sigma = 1.0;

        if (call == 1)
        {
            call = !call;
            return (sigma * (double) X2);
        }

        do
        {
            U1 = -1 + ((double) rand () / RAND_MAX) * 2;
            U2 = -1 + ((double) rand () / RAND_MAX) * 2;
            W = custom_double_pow (U1, 2) + custom_double_pow (U2, 2);
        }
        while (W >= 1 || W == 0);
        
        mult = sqrt ((-2 * log (W)) / W);
        X1 = U1 * mult;
        X2 = U2 * mult;
        
        call = !call;

        return (sigma * (double) X1);
    }

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
                else 
                {
                    if (h_random[dim_S*r_i + j_S]<h_i_min)
                    {
                        h_i_min = h_random[dim_S*r_i + j_S];
                    }
                }
            }
            
            h_dev_avg[j_S] = h_dev_net[j_S] / no_of_sites;
        }
        if (fabs(h_i_max) < fabs(h_i_min))
        {
            h_i_max = fabs(h_i_min);
        }
        else
        {
            h_i_max = fabs(h_i_max);
        }
        h_i_min = -h_i_max;
        return 0;
    }

    int initialize_J_zero()
    {
        long int i;
        for(i=0; i<dim_L*no_of_sites; i=i+1)
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
            }
            J_dev_avg[j_L] = J_dev_net[j_L] / no_of_sites;
        }
        
        for(j_L=0; j_L<dim_L; j_L=j_L+1)
        {
            J_dev_net[j_L] = 0;
            for(i=0; i<no_of_sites; i=i+1)
            {
                
                J_random[2*dim_L*N_N_I[2*dim_L*i + 2*j_L] + 2*j_L + 1] = J_random[2*dim_L*i + 2*j_L];
                
            }
        }
        
        return 0;
    }

    int initialize_nearest_neighbor_index()
    {
        long int i; 
        int j_L, k_L;
        no_of_sites = 1;
        for (j_L=0; j_L<dim_L; j_L++)
        {
            no_of_sites = no_of_sites*lattice_size[j_L];
        }
        N_N_I = (long int*)malloc(2*dim_L*no_of_sites*sizeof(long int));    
        
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
        
        no_of_sites = 1;


        for (j_L=0; j_L<dim_L; j_L++)
        {
            no_of_sites = no_of_sites*lattice_size[j_L];
        }

        if (no_of_sites % 2 == 1)
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
        // no_of_black_white_sites[0] = no_of_black_sites;
        // no_of_black_white_sites[1] = no_of_white_sites;
        
        // black_checkerboard = (long int*)malloc(no_of_black_sites*sizeof(long int));
        black_white_checkerboard[0] = (long int*)malloc(no_of_black_white_sites[0]*sizeof(long int));
        // white_checkerboard = (long int*)malloc(no_of_white_sites*sizeof(long int));
        black_white_checkerboard[1] = (long int*)malloc(no_of_black_white_sites[1]*sizeof(long int));
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
                black_white_checkerboard[0][black_white_index[0]] = i;
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
                black_white_checkerboard[1][black_white_index[1]] = i;
                // if ( white_checkerboard[black_white_index[1]] - black_white_checkerboard[1][black_white_index[1]] != 0 )
                // {
                //     printf("white_checkerboard[i] = %ld, black_white_checkerboard[1][i] = %ld; %ld\n", white_checkerboard[black_white_index[1]], black_white_checkerboard[1][black_white_index[1]], white_checkerboard[black_white_index[1]] - black_white_checkerboard[1][black_white_index[1]] );
                // }
                black_white_index[1]++;
                // white_index++;
            }
            
        }
        return 0;
    }

    int initialize_ordered_spin_config()
    {
        long int i;
        int j_S;

        for(j_S=0; j_S<dim_S; j_S=j_S+1)
        {
            #pragma omp parallel for
            for(i=0; i<no_of_sites; i=i+1)
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
                h_mod = h_mod + custom_double_pow(h_random[dim_S*i+j_S], 2);
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
                    h_dev_mod = h_dev_mod + custom_double_pow(h_dev_avg[j_S], 2);
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
                    s_mod = s_mod + custom_double_pow(spin[dim_S*i+j_S], 2);
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

//====================      Magnetisation                    ====================//

    int ensemble_abs_m()
    {
        long int i; 
        int j_S;
        for (j_S=0; j_S<dim_S; j_S=j_S+1)
        {
            abs_m[j_S] = 0;
        }
        #pragma omp parallel for private(j_S) reduction(+:abs_m[:dim_S])
        for(i=0; i<no_of_sites; i=i+1)
        {
            for (j_S=0; j_S<dim_S; j_S=j_S+1)
            {
                abs_m[j_S] += fabs( spin[dim_S*i + j_S]);
            }
        }
        for (j_S=0; j_S<dim_S; j_S=j_S+1)
        {
            abs_m[j_S] = abs_m[j_S] / no_of_sites;
        }
        
        // for (j_S=0; j_S<dim_S; j_S=j_S+1)
        // {
        //     double abs_m_j_S = 0;
        //     #pragma omp parallel for reduction(+:abs_m_j_S)
        //     for(i=0; i<no_of_sites; i=i+1)
        //     {
        //         abs_m_j_S += fabs( spin[dim_S*i + j_S]);
        //     }
        //     abs_m[j_S] = abs_m_j_S / no_of_sites;
        // }
        
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

    int ensemble_m()
    {
        long int i; 
        int j_S;
        for (j_S=0; j_S<dim_S; j_S=j_S+1)
        {
            m[j_S] = 0;
        }
        #pragma omp parallel for private(j_S) reduction(+:m[:dim_S])
        for(i=0; i<no_of_sites; i=i+1)
        {
            for (j_S=0; j_S<dim_S; j_S=j_S+1)
            {
                m[j_S] += spin[dim_S*i + j_S];
            }
        }
        for (j_S=0; j_S<dim_S; j_S=j_S+1)
        {
            m[j_S] = m[j_S] / no_of_sites;
        }
        // for (j_S=0; j_S<dim_S; j_S=j_S+1)
        // {
            // m[j_S] = 0;
            // #pragma omp parallel for reduction(+:m[j_S])
            // for(i=0; i<no_of_sites; i=i+1)
            // {
                // m[j_S] += spin[dim_S*i + j_S];
            // }
            // m[j_S] = m[j_S] / no_of_sites;
        // }
        
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
            m_2_persite += custom_double_pow(m[j_S], 2);
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
            m_2_persite += custom_double_pow(m[j_S], 2);
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
            m_2_persite += custom_double_pow(m[j_S], 2);
        }

        m_4_sum += custom_double_pow(m_2_persite, 2);

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
        m_abs_sum = 0;
        m_abs_avg = 1;
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
            m_2_persite += custom_double_pow(m[j_S], 2);
        }

        m_abs_sum += sqrt(m_2_persite);
        m_2_sum += m_2_persite;
        m_4_sum += custom_double_pow(m_2_persite, 2);

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
        m_abs_avg = m_abs_sum / MCS_counter;
        m_2_avg = m_2_sum / MCS_counter;
        m_4_avg = m_4_sum / MCS_counter;
        
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
                    E += - (J[j_L] + J_random[2*dim_L*i + 2*j_L])  * spin[dim_S * N_N_I[i*2*dim_L + 2*j_L] + j_S] * (spin[dim_S*i + j_S]);
                }
                E += - (h[j_S] + h_random[dim_S*i + j_S]) * (spin[dim_S*i + j_S]);
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
        E_2_sum += custom_double_pow(E, 2);
        
        return 0;
    }

    int average_of_moment_E_2(double MCS_counter)
    {
        E_2_avg = E_2_sum / MCS_counter;
        
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
        #pragma omp parallel for private(j_L, j_S, j_SS) reduction(+:Y_1[:dim_S*dim_S*dim_L], Y_2[:dim_S*dim_S*dim_L])
        for(i=0; i<no_of_sites; i++)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                        Y_1[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += - ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_S] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_SS];
                        Y_2[dim_S*dim_S*j_L + dim_S*j_S + j_SS] += ( J[j_L] + J_random[2*dim_L*i + 2*j_L] ) * spin[dim_S*i + j_SS] * spin[dim_S*N_N_I[i*2*dim_L + 2*j_L] + j_S];
                    }
                }
            }
        }
        for (j_L_j_S_j_SS=0; j_L_j_S_j_SS < dim_S*dim_S*dim_L; j_L_j_S_j_SS++)
        {
            Y_1[j_L_j_S_j_SS] = Y_1[j_L_j_S_j_SS] / no_of_sites;
            Y_2[j_L_j_S_j_SS] = custom_double_pow(Y_2[j_L_j_S_j_SS], 2) / no_of_sites;
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
        
        Cv = (E_2_avg - custom_double_pow(E_avg, 2)) / custom_double_pow(T, 2);
        
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
        
        B = (1.0 / 2.0) * (3.0 - (m_4_avg / custom_double_pow(m_2_avg, 2)));
        
        
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
        
        X = (m_2_avg - custom_double_pow(m_abs_avg, 2)) / T;
        
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
        
        Cv = (E_2_avg - custom_double_pow(E_avg, 2)) / custom_double_pow(T, 2);

        X = (m_2_avg - custom_double_pow(m_abs_avg, 2)) / T;

        B = (1.0 / 2.0) * (3.0 - (m_4_avg / custom_double_pow(m_2_avg, 2)));
        
        
        return 0;
    }

//====================      MonteCarlo-tools                 ====================//

    int update_spin_sum(long int xyzi, double *spin_local)
    {
        int j_S;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            spin[dim_S*xyzi + j_S] = spin_local[j_S];
        }

        return 0;
    }

    double Energy_minimum(long int xyzi, double *spin_local, double *field_local)
    {
        int j_S, j_L, k_L;
        double Energy_min=0.0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            field_local[j_S] = 0.0;
            for (j_L=0; j_L<dim_L; j_L++)
            {
                for (k_L=0; k_L<2; k_L++)
                {
                    field_local[j_S] = field_local[j_S] - (J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                }
            }
            field_local[j_S] = field_local[j_S] - (h[j_S] + h_random[dim_S*xyzi + j_S]);
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

    double Energy_old(long int xyzi, double *spin_local, double *field_local)
    {
        int j_S, j_L, k_L;
        double Energy_ol=0.0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            field_local[j_S] = 0.0;
            for (j_L=0; j_L<dim_L; j_L++)
            {
                for (k_L=0; k_L<2; k_L++)
                {
                    // field_site[dim_S*xyzi + j_S] = field_site[dim_S*xyzi + j_S] - (J[j_L] ) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                    field_local[j_S] = field_local[j_S] - (J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                }
            }
            // field_site[dim_S*xyzi + j_S] = field_site[dim_S*xyzi + j_S] - (h[j_S]);
            field_local[j_S] = field_local[j_S] - (h[j_S] + h_random[dim_S*xyzi + j_S]);
            // field_site[dim_S*xyzi + j_S] = field_local[j_S];
            // spin_old[dim_S*xyzi + j_S] = spin[dim_S*xyzi + j_S];
            Energy_ol = Energy_ol + field_local[j_S] * spin[dim_S*xyzi + j_S];
        }
        
        return Energy_ol;
    }

    double Energy_new(long int xyzi, double *spin_local, double *field_local)
    {
        int j_S;
        double Energy_nu=0.0, s_mod=0.0;
        double limit = 0.01 * dim_S;
        
        do
        {
            s_mod=0.0;
            for(j_S=0; j_S<dim_S; j_S=j_S+1)
            {
                spin_local[j_S] = (-1.0 + 2.0 * (double)rand_r(&random_seed[cache_size*omp_get_thread_num()])/(double)(RAND_MAX));
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
            update_spin_sum(xyzi, spin_local);
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
                //     update_spin_sum(site_i, 1);
                // }
                // else
                // {
                //     update_spin_sum(site_i, 0);
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
            //     update_spin_sum(xyzi, 1);
            // }
            // else
            // {
            //     update_spin_sum(xyzi, 0);
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
                long int site_index = black_white_checkerboard[black_or_white][i];

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
            update_spin_sum(xyzi, spin_local);
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
                //     update_spin_sum(site_i, 1);
                // }
                // else
                // {
                //     update_spin_sum(site_i, 0);
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
            //     update_spin_sum(xyzi, 1);
            // }
            // else
            // {
            //     update_spin_sum(xyzi, 0);
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
                long int site_index = black_white_checkerboard[black_or_white][i];

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
                s_mod = s_mod + custom_double_pow(reflection_plane[j_S], 2);
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

    int transform_spin(long int xyzi)
    {
        double Si_dot_ref = 0;
        int j_S;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            Si_dot_ref += spin[dim_S*xyzi + j_S] * reflection_plane[j_S];
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            spin_new[dim_S*xyzi + j_S] = spin[dim_S*xyzi + j_S] - 2 * Si_dot_ref * reflection_plane[j_S];
        }
        return 0;
    }

    double E_site_old(long int xyzi)
    {
        double energy_site = 0;
        int j_S;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            energy_site = -(h[j_S] + h_random[xyzi*dim_S + j_S]) * spin[dim_S*xyzi + j_S];
        }

        return energy_site;
    }

    double E_site_new(long int xyzi)
    {
        double energy_site = 0;

        int j_S;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            energy_site = -(h[j_S] + h_random[xyzi*dim_S + j_S]) * spin_new[dim_S*xyzi + j_S];
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
            energy_site = -(J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*xyzi + j_S] * spin[dim_S*xyzi_nn + j_S];
        }

        return energy_site;
    }

    double E_bond_new(long int xyzi, int j_L, int k_L)
    {
        double energy_site = 0;
        double Si_dot_ref = 0;
        int j_S;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            energy_site = -(J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*xyzi + j_S] * spin_new[dim_S*xyzi + j_S];
        }

        return energy_site;
    }

    int nucleate_from_site(long int xyzi)
    {
        
        update_spin_sum(xyzi, spin_new + dim_S*xyzi);
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
                    transform_spin(xyzi_nn);
                    double delta_E_bond = -E_bond_old(xyzi, j_L, k_L, xyzi_nn);
                    delta_E_bond += E_bond_new(xyzi, j_L, k_L);
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
                            delta_E_site += E_site_new(xyzi_nn);
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
                transform_spin(xyzi);
                delta_E_site += E_site_new(xyzi);
                if (delta_E_site <= 0)
                {
                    nucleate_from_site(xyzi);
                }
                else
                {
                    double r = (double) rand_r(&random_seed[cache_size*omp_get_thread_num()]) / (double) RAND_MAX;
                    if (r<exp(-delta_E_site/T))
                    {
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

//====================      Save J, h                        ====================//

    int save_h_config()
    {
        long int i;
        int j_S, j_L;
        h_random = (double*)malloc(dim_S*no_of_sites*sizeof(double));

        initialize_h_random_gaussian();

        char output_file_1[128];
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
        pos += sprintf(pos, ".dat");
            
        pFile_1 = fopen(output_file_1, "w"); // opens new file for writing
        
        fprintf(pFile_1, "%lf ", h_i_min);
        printf( "\nh_i_min=%lf ", h_i_min);
        // fprintf(pFile_1, "\n");
        fprintf(pFile_1, "%lf ", h_i_max);
        printf( "h_i_max=%lf \n", h_i_max);
        fprintf(pFile_1, "\n");

        for (j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%lf ", h[j_S]);
            fprintf(pFile_1, "%lf ", sigma_h[j_S]);
            printf( "sigma_h[%d]=%lf \n", j_S, sigma_h[j_S]);
            fprintf(pFile_1, "%lf ", h_dev_avg[j_S]);
            printf( "h_dev_avg[%d]=%lf \n", j_S, h_dev_avg[j_S]);
            fprintf(pFile_1, "\n");
        }
        fprintf(pFile_1, "\n");

        for (i = 0; i < no_of_sites; i++)
        {
            for (j_S = 0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf ", h_random[dim_S*i + j_S]);
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

    int save_J_config()
    {
        long int i;
        int j_L, k_L;
        J_random = (double*)malloc(2*dim_L*no_of_sites*sizeof(double));

        initialize_J_random_gaussian();
        
        char output_file_1[128];
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
        pos += sprintf(pos, ".dat");
        
        pFile_1 = fopen(output_file_1, "w"); // opens new file for writing
        
        fprintf(pFile_1, "%lf ", J_i_min);
        // fprintf(pFile_1, "\n");
        fprintf(pFile_1, "%lf ", J_i_max);
        fprintf(pFile_1, "\n");

        for (j_L=0; j_L<dim_L; j_L++)
        {
            fprintf(pFile_1, "%lf ", J[j_L]);
            fprintf(pFile_1, "%lf ", sigma_J[j_L]);
            fprintf(pFile_1, "%lf ", J_dev_avg[j_L]);
            fprintf(pFile_1, "\n");
        }
        fprintf(pFile_1, "\n");

        for (i = 0; i < no_of_sites; i++)
        {
            for (j_L = 0; j_L<dim_L; j_L++)
            {
                for (k_L = 0; k_L<2; k_L++)
                {
                    fprintf(pFile_1, "%lf ", J_random[2*dim_L*i + 2*j_L + k_L]);
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

//====================      Load J, h                        ====================//

    int load_h_config()
    {
        //---------------------------------------------------------------------------------------//
        long int i;
        int j_S, j_L;
        char input_file_1[128];
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
        pos += sprintf(pos, ".dat");
        
        pFile_1 = fopen(input_file_1, "r"); // opens file for reading

        if (pFile_1 == NULL)
        {
            save_h_config(); // creates file for later
        }
        else
        {
            h_random = (double*)malloc(dim_S*no_of_sites*sizeof(double));
            fscanf(pFile_1, "%lf", &h_i_min);
            fscanf(pFile_1, "%lf", &h_i_max);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fscanf(pFile_1, "%lf", &h[j_S]);
                fscanf(pFile_1, "%lf", &sigma_h[j_S]);
                fscanf(pFile_1, "%lf", &h_dev_avg[j_S]);
            }
            
            for (i = 0; i < no_of_sites; i++)
            {
                for (j_S = 0; j_S<dim_S; j_S++)
                {
                    fscanf(pFile_1, "%lf", &h_random[dim_S*i + j_S]);
                }
            }
            fclose(pFile_1);
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

    int load_J_config()
    {
        //---------------------------------------------------------------------------------------//
        long int i;
        int j_L, k_L;
        char input_file_1[128];
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
        pos += sprintf(pos, ".dat");
        
        pFile_1 = fopen(input_file_1, "r"); // opens file for reading

        if (pFile_1 == NULL)
        {
            save_J_config(); // creates file for later
        }
        else
        {
            J_random = (double*)malloc(2*dim_L*no_of_sites*sizeof(double));
            fscanf(pFile_1, "%lf", &J_i_min);
            fscanf(pFile_1, "%lf", &J_i_max);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                fscanf(pFile_1, "%lf", &J[j_L]);
                fscanf(pFile_1, "%lf", &sigma_J[j_L]);
                fscanf(pFile_1, "%lf", &J_dev_avg[j_L]);
            }
            
            for (i = 0; i < no_of_sites; i++)
            {
                for (j_L = 0; j_L<dim_L; j_L++)
                {
                    for (k_L = 0; k_L<2; k_L++)
                    {
                        fscanf(pFile_1, "%lf", &J_random[2*dim_L*i + 2*j_L + k_L]);
                    }
                }
            }
            fclose(pFile_1);
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
        set_sum_of_moment_m_abs_0();
        set_sum_of_moment_E_0();

        printf("Averaging iterations... h=%lf", h[0]);
        for (j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", h[j_S]);
        }
        printf("\n");

        while(average_iter)
        {
            Monte_Carlo_Sweep(1);
            // random_Wolff_sweep(1);
            ensemble_m();
            ensemble_E();
            // ensemble_Y_ab_mu();
            sum_of_moment_m();
            sum_of_moment_m_abs();
            sum_of_moment_E();
            // sum_of_moment_Y_ab_mu();
            MCS_counter = MCS_counter + 1;
            
            average_iter = average_iter - 1;
        }
        printf("Done.\n");

        average_of_moment_m(MCS_counter);
        average_of_moment_m_abs(MCS_counter);
        average_of_moment_E(MCS_counter);
        // average_of_moment_Y_ab_mu(MCS_counter);

        printf("Final: Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);

        printf("<M> = (%lf", m_avg[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m_avg[j_S]);
        }
        printf("), <E> = %lf \n", E_avg);
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
            // char output_file_1[256];
            char *pos = output_file_1;
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
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                    pos += sprintf(pos, "%lf", h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld_%ld.dat", h_order, r_order, thermal_i, average_j);
            pFile_1 = fopen(output_file_1, "a");
            
            fprintf(pFile_1, "step\t ");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t ");
            fprintf(pFile_1, "T=%lf\t dim_{Lat}=%d\t L=", T, dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_{Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                    fprintf(pFile_1, "%lf", h[j_S]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t thermalizing-MCS=%ld\t averaging-MCS=%ld\t \n", h_order, r_order, thermal_i, average_j);

            fclose(pFile_1);
        }

        int i;
        for (i=0; i<repeat_for_same_T; i++)
        {
            thermalizing_iteration(thermal_i);
            averaging_iteration(average_j);
            pFile_1 = fopen(output_file_1, "a");
            fprintf(pFile_1, "%d\t ", i);
            
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m_avg[j_S]);
            }
            
            fprintf(pFile_1, "%lf\t ", E_avg);

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

        // create file name and pointer. 
        {
            // char output_file_1[256];
            char *pos = output_file_1;
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
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                    pos += sprintf(pos, "%lf", h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld_%ld.dat", h_order, r_order, thermal_i, average_j);
            pFile_1 = fopen(output_file_1, "a");
            
            fprintf(pFile_1, "|m|\t ");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t ");
            fprintf(pFile_1, "T\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_{Spin}=%d\t h", dim_S);
            
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t thermalizing-MCS=%ld\t averaging-MCS=%ld\t \n", h_order, r_order, thermal_i, average_j);

            fclose(pFile_1);
        }

        thermalizing_iteration(thermal_i);
        averaging_iteration(average_j);
        pFile_1 = fopen(output_file_1, "a");
        fprintf(pFile_1, "%lf\t ", m_abs_avg);
        
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%lf\t ", m_avg[j_S]);
        }
        
        fprintf(pFile_1, "%lf\t ", E_avg);
        fprintf(pFile_1, "%lf\t ", T);

        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S)
            {
                fprintf(pFile_1, ",");
            }
            fprintf(pFile_1, "%lf", h[j_S]);
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
            // char output_file_1[256];
            char *pos = output_file_1;
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
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                    pos += sprintf(pos, "%lf", h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld_%ld.dat", h_order, r_order, thermal_i, average_j);
            pFile_1 = fopen(output_file_1, "a");
            
            fprintf(pFile_1, "T\t ");
            fprintf(pFile_1, "<|m|>\t ");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "<Y[%d,%d][%d]>\t ", j_S, j_SS, j_L);
                    }
                }
            } */
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_{Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                    fprintf(pFile_1, "%lf", h[j_S]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t thermalizing-MCS=%ld\t averaging-MCS=%ld\t \n", h_order, r_order, thermal_i, average_j);
        }
        
        for (T=Temp_min; T<=Temp_max; T=T+delta_T)
        {
            printf("\nT=%lf\t ", T);
            initialize_spin_and_evolve_at_T(); 
            
            fprintf(pFile_1, "%lf\t ", T);
            fprintf(pFile_1, "%lf\t ", m_abs_avg);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m_avg[j_S]);
            }
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "%lf\t ", Y_ab_mu[dim_S*dim_S*j_L + dim_S*j_S + j_SS]);
                    }
                }
            } */
            fprintf(pFile_1, "%lf\t ", E_avg);

            fprintf(pFile_1, "\n");
        }
        
        fclose(pFile_1);
        
        return 0;
    }

    int cooling_protocol()
    {
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
        fprintf(pFile_1, "Cooling... \n");
        for (T=Temp_max; T>Temp_min; T=T-delta_T)
        {
            printf("\nT=%lf\t ", T);
            
            thermalizing_iteration(thermal_i);
            averaging_iteration(average_j);

            // pFile_1 = fopen(output_file_1, "a");

            fprintf(pFile_1, "%lf\t %lf\t ", T, m_abs_avg);

            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m_avg[j_S]);
            } 
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "%lf\t ", Y_ab_mu[dim_S*dim_S*j_L + dim_S*j_S + j_SS]);
                    }
                }
            } */

            fprintf(pFile_1, "%lf\t ", E_avg);    
            fprintf(pFile_1, "\n");
            // fclose(pFile_1);
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

    int heating_protocol()
    {
        int j_S, j_SS, j_L;

        ensemble_all();
        // ensemble_E();
        // ensemble_m();

        printf("Initial Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);
        printf("Heating... ");
        fprintf(pFile_1, "Heating... \n");
        for (T=Temp_min; T<=Temp_max; T=T+delta_T)
        {
            printf("\nT=%lf\t ", T);
            
            thermalizing_iteration(thermal_i);
            averaging_iteration(average_j);

            // pFile_1 = fopen(output_file_1, "a");

            fprintf(pFile_1, "%lf\t %lf\t ", T, m_abs_avg);

            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m_avg[j_S]);
            }
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "%lf\t ", Y_ab_mu[dim_S*dim_S*j_L + dim_S*j_S + j_SS]);
                    }
                }
            } */
            fprintf(pFile_1, "%lf\t ", E_avg);
            fprintf(pFile_1, "\n");
            // fclose(pFile_1);
        }

        ensemble_all();
        // ensemble_E();
        // ensemble_m();

        printf("Final Magnetisation = (%lf", m[0]);
        for(j_S=1; j_S<dim_S; j_S++)
        {
            printf(",%lf", m[j_S]);
        }
        printf("), Energy = %lf \n", E);
        printf("------------------------\n");
        
        return 0;
    }

    int zfc_zfh_or_both(int c_h_b)
    {
        int j_S, j_SS, j_L;
        
        initialize_spin_config();
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

        // create file name and pointer. 
        {
            // char output_file_1[256];
            char *pos = output_file_1;
            pos += sprintf(pos, "O(%d)_%dD_ZFC-ZFH_%c_%c_", dim_S, dim_L, G_M_W[Gl_Me_Wo], C_R_L[Ch_Ra_Li]);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_(%lf,%lf)-[%lf]_{", Temp_min, Temp_max, delta_T);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                    pos += sprintf(pos, "%lf", h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld_%ld.dat", h_order, r_order, thermal_i, average_j);
            pFile_1 = fopen(output_file_1, "a");
            
            fprintf(pFile_1, "T\t |m|\t ");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            } 
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "<Y[%d,%d][%d]>\t ", j_S, j_SS, j_L);
                    }
                }
            } */
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_{Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                    fprintf(pFile_1, "%lf", h[j_S]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t (thermalizing-MCS,averaging-MCS)/{/Symbol D}T=(%ld,%ld)/%lf\t \n", h_order, r_order, thermal_i, average_j, delta_T);
        }

        if (c_h_b == 0 || c_h_b == 2)
        {
            cooling_protocol();
        }
        if (c_h_b == 1 || c_h_b == 2)
        {
            heating_protocol();
        }

        fclose(pFile_1);
        
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

    int hysteresis_protocol(int jj_S)
    {
        int j_S, j_L;
        double h_start = order[jj_S]*(h_max+h_i_max);
        double h_end = -h_start;

        // delta_h = (2*order[jj_S]-1)*delta_h;

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
            // char output_file_1[256];
            char *pos = output_file_1;
            pos += sprintf(pos, "O(%d)_%dD_hysteresis_%c_%c_", dim_S, dim_L, G_M_W[Gl_Me_Wo], C_R_L[Ch_Ra_Li]);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%lf_{", T);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{");
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
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}_%d_%d_%ld.dat", h_order, r_order, hysteresis_MCS);
            pFile_1 = fopen(output_file_1, "a");
            
            fprintf(pFile_1, "h[%d]\t ", jj_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_(Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                if (j_S==jj_S)
                {
                    fprintf(pFile_1, "-");
                }
                else
                {
                    fprintf(pFile_1, "%lf", h[j_S]);
                }
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t MCS/{/Symbol d}h=%ld/%lf\t \n", h_order, r_order, hysteresis_MCS, delta_h);
        }

        int i;
        for (i=0; i<hysteresis_repeat; i=i+1)
        {
            printf("h = %lf --> %lf\t ", h_start, h_end);
            for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
            {
                hysteresis_average(hysteresis_MCS);

                fprintf(pFile_1, "%lf\t ", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m_avg[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E_avg);

                fprintf(pFile_1, "\n");
            }
            printf("..(%d) \nh = %lf <-- %lf\t ", i+1, h_start, h_end);
            for (h[jj_S] = h_end; order[jj_S] * h[jj_S] <= order[jj_S] * h_start; h[jj_S] = h[jj_S] + order[jj_S] * delta_h)
            {
                hysteresis_average(hysteresis_MCS);

                fprintf(pFile_1, "%lf\t ", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m_avg[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E_avg);

                fprintf(pFile_1, "\n");
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
        h[0] = h_i_max;
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
            // char output_file_1[256];
            char *pos = output_file_1;
            pos += sprintf(pos, "O(%d)_%dD_hysteresis_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%lf_{", T);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{");
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
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}.dat");
            pFile_1 = fopen(output_file_1, "a");
            
            fprintf(pFile_1, "h[%d]\t ", jj_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_(Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                if (j_S==jj_S)
                {
                    fprintf(pFile_1, "-");
                }
                else
                {
                    fprintf(pFile_1, "%lf", h[j_S]);
                }
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t MCS/{/Symbol d}h=%ld/%lf\t \n", h_order, r_order, hysteresis_MCS, delta_h);
        }
        
        long int nucleation_site;

        long int remaining_sites = no_of_sites;
        ensemble_m();
        ensemble_E();
        fprintf(pFile_1, "%lf\t ", h[jj_S]);
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%lf\t ", m[j_S]);
        }
        fprintf(pFile_1, "%lf\t ", E);

        fprintf(pFile_1, "\n");
        
        while (remaining_sites)
        {
            nucleation_site = find_extreme(order[0], remaining_sites);

            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);

            fprintf(pFile_1, "\n");
            
            remaining_sites = flip_unstable(nucleation_site, remaining_sites);
            printf("h=%lf, ", h[0]);
            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);

            fprintf(pFile_1, "\n");
        }

        h[0] = h_i_min;
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
        fprintf(pFile_1, "%lf\t ", h[jj_S]);
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%lf\t ", m[j_S]);
        }
        fprintf(pFile_1, "%lf\t ", E);

        fprintf(pFile_1, "\n");

        while (remaining_sites)
        {
            nucleation_site = find_extreme(order[0], remaining_sites);

            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);

            fprintf(pFile_1, "\n");
            
            remaining_sites = flip_unstable(nucleation_site, remaining_sites);
            printf("h=%lf, ", h[0]);
            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);

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
        h[0] = h_i_max;
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
            // char output_file_1[256];
            char *pos = output_file_1;
            pos += sprintf(pos, "O(%d)_%dD_ringdown_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%lf_{", T);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{");
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
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}.dat");
            pFile_1 = fopen(output_file_1, "a");
            
            fprintf(pFile_1, "h[%d]\t ", jj_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_(Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                if (j_S==jj_S)
                {
                    fprintf(pFile_1, "-");
                }
                else
                {
                    fprintf(pFile_1, "%lf", h[j_S]);
                }
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t MCS/{/Symbol d}h=%ld/%lf\t \n", h_order, r_order, hysteresis_MCS, delta_h);
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
            
            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);
            
            fprintf(pFile_1, "\n");

            double old_h=0.0, new_h=0.0;
            

            while (old_h == new_h || M_compare > -m[0])
            {
                nucleation_site = find_extreme(order[0], remaining_sites);

                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%lf\t ", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E);

                fprintf(pFile_1, "\n");
                
                remaining_sites = flip_unstable(nucleation_site, remaining_sites);
                printf("h=%lf, ", h[0]);
                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%lf\t ", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E);
                old_h = h[0];
                find_extreme(order[0], remaining_sites);
                new_h = h[0];

                fprintf(pFile_1, "\n");
            }

            h[0] = h_i_min;
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
            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);

            fprintf(pFile_1, "\n");

            old_h = 0.0, new_h = 0.0;

            while (old_h == new_h || M_compare > m[0])
            {
                nucleation_site = find_extreme(order[0], remaining_sites);

                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%lf\t ", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E);

                fprintf(pFile_1, "\n");
                
                remaining_sites = flip_unstable(nucleation_site, remaining_sites);
                printf("h=%lf, ", h[0]);
                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%lf\t ", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E);
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
            // char output_file_1[256];
            char *pos = output_file_1;
            pos += sprintf(pos, "O(%d)_%dD_rpm_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%lf_{", T);
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{");
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
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "}.dat");
            pFile_1 = fopen(output_file_1, "a");
        }
        // column labels and parameters
        {
            fprintf(pFile_1, "h[%d]\t ", jj_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t RPM_error");
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_(Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                if (j_S==jj_S)
                {
                    fprintf(pFile_1, "-");
                }
                else
                {
                    fprintf(pFile_1, "%lf", h[j_S]);
                }
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t MCS/{/Symbol d}h=%ld/%lf\t \n", h_order, r_order, hysteresis_MCS, delta_h);
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
        
        fprintf(pFile_1, "%lf\t ", h[jj_S]);
        for(j_S=0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%lf\t ", m[j_S]);
        }
        fprintf(pFile_1, "%lf\t ", E);
        
        fprintf(pFile_1, "\n");

        while (old_h == new_h || m_start < m[jj_S])
        {
            nucleation_site = find_extreme(order[0], remaining_sites);

            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);

            fprintf(pFile_1, "\n");
            
            remaining_sites = flip_unstable(nucleation_site, remaining_sites);
            // printf("h=%lf, ", h[0]);
            ensemble_m();
            ensemble_E();
            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);
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
            
            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);
            
            fprintf(pFile_1, "\n");
            while ( old_h == new_h || delta_m[i] > fabs( mag_rpm[i] - m[jj_S] ) )
            {
                nucleation_site = find_extreme(order[0], remaining_sites);

                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%lf\t ", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E);

                fprintf(pFile_1, "\n");
                
                remaining_sites = flip_unstable(nucleation_site, remaining_sites);
                // printf("h=%lf, ", h[0]);
                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%lf\t ", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E);
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
            
            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);
            
            fprintf(pFile_1, "\n");

            while ( old_h == new_h || !( ( h_ext[i] >= old_h && h_ext[i] < new_h ) || ( h_ext[i] <= old_h && h_ext[i] > new_h ) ) )
            {
                nucleation_site = find_extreme(order[0], remaining_sites);

                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%lf\t ", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E);

                fprintf(pFile_1, "\n");
                
                remaining_sites = flip_unstable(nucleation_site, remaining_sites);
                // printf("h=%lf, ", h[0]);
                ensemble_m();
                ensemble_E();
                fprintf(pFile_1, "%lf\t ", h[jj_S]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E);
                old_h = h[0];
                find_extreme(order[0], remaining_sites);
                new_h = h[0];

                fprintf(pFile_1, "\n");
            }

            
            {
                fprintf(pFile_1, "%lf\t ", old_h);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E);

                fprintf(pFile_1, "%lf\t ", (mag_rpm[i] - m[jj_S]) );

                fprintf(pFile_1, "\n");

                printf("m[0]=%lf, h_ext[%d]=%lf \n", m[jj_S], i, h_ext[i]);
            }
        }
        fclose(pFile_1);
        return 0;
    }

//====================      RFXY ZTNE                        ====================//

    int update_spin_XY(long int xyzi, int new_or_min)
    {
        int j_S;
        if(new_or_min==1)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin[dim_S*xyzi + j_S] = spin_new[dim_S*xyzi + j_S];
            }
        }
        else
        {
            if (new_or_min == -1)
            {
                for (j_S=0; j_S<dim_S; j_S++)
                {
                    spin[dim_S*xyzi + j_S] = spin_old[dim_S*xyzi + j_S];
                }
            }
        }

        return 0;
    }

    double Energy_minimum_old_XY(long int xyzi, double *spin_local)
    {
        int j_S, j_L, k_L;
        double Energy_min=0.0;
        double field_local[dim_S];
        
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            field_local[j_S] = 0.0;
            for (j_L=0; j_L<dim_L; j_L++)
            {
                for (k_L=0; k_L<2; k_L++)
                {
                    field_local[j_S] = field_local[j_S] - (J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                }
            }
            field_local[j_S] = field_local[j_S] - (h[j_S] + h_random[dim_S*xyzi + j_S]);
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

    double Energy_minimum_new_XY(long int xyzi, double *spin_local)
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
                    field_local[j_S] = field_local[j_S] - (J[j_L] + J_random[2*dim_L*xyzi + 2*j_L + k_L]) * spin[dim_S*N_N_I[2*dim_L*xyzi + 2*j_L + k_L] + j_S];
                }
            }
            field_local[j_S] = field_local[j_S] - (h[j_S] + h_random[dim_S*xyzi + j_S]);
            Energy_min = Energy_min + field_local[j_S] * field_local[j_S];
        }
        if(Energy_min==0)
        {
            for (j_S=0; j_S<dim_S; j_S++)
            {
                spin_local[j_S] = -order[j_S];
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

    double update_to_minimum_checkerboard(long int xyzi, double *spin_local)
    {
        int j_S;
        
        Energy_minimum_old_XY(xyzi, spin_local);

        double spin_diff_abs = 0.0;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            spin_diff_abs += fabs(spin[dim_S*xyzi + j_S] - spin_local[j_S]);
        }
        update_spin_sum(xyzi, spin_local);
        return spin_diff_abs;
    }

    int zero_temp_RFXY_hysteresis_axis(int jj_S, double order_start)
    {
        double cutoff_local = 0.0;
        int j_S, j_L;
        T = 0;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[j_S] = order_start;
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
        double h_end = -h_start;
        h_order = 0;
        r_order = 0;
        initialize_spin_config();
        spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));

        printf("\nztne RFXY looping  at T=%lf.. \n",  T);

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
            // char output_file_1[256];
            char *pos = output_file_1;
            pos += sprintf(pos, "O(%d)_%dD_hysteresis_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%lf_{", T);
            /* for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{"); */
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
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "_%lf}.dat", delta_h);
            pFile_1 = fopen(output_file_1, "a");
            
            fprintf(pFile_1, "h[%d]\t ", jj_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_(Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                if (j_S==jj_S)
                {
                    fprintf(pFile_1, "-");
                }
                else
                {
                    fprintf(pFile_1, "%lf", h[j_S]);
                }
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t MCS/{/Symbol d}h=%ld/%lf\t \n", h_order, r_order, hysteresis_MCS, delta_h);
        }
        long int site_i;
        
        for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
        {
            cutoff_local = -0.1;
            
            do 
            {
                double cutoff_local_last = cutoff_local;
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
                        cutoff_local += fabs(spin[site_i] - spin_temp[site_i]);
                        spin[site_i] = spin_temp[site_i];
                    }
                }
                // printf("\nblac = %g\n", cutoff_local);

                if (cutoff_local == cutoff_local_last)
                {
                    break;
                }
            }
            while (cutoff_local > CUTOFF); // 10^-10

            ensemble_m();
            ensemble_E();
            
            // printf("\nblah = %lf", h[jj_S]);
            // printf("\nblam = %lf", m[jj_S]);
            // printf("\n");

            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);

            fprintf(pFile_1, "\n");
        }

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
        h_start = order[jj_S]*(h_max+h_i_max);
        h_end = -h_start;

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

        for (h[jj_S] = h_start; order[jj_S] * h[jj_S] >= order[jj_S] * h_end; h[jj_S] = h[jj_S] - order[jj_S] * delta_h)
        {
            cutoff_local = -0.1;
            
            do 
            {
                double cutoff_local_last = cutoff_local;
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
                        cutoff_local += fabs(spin[site_i] - spin_temp[site_i]);
                        spin[site_i] = spin_temp[site_i];
                    }
                }
                // printf("\nblac = %g\n", cutoff_local);

                if (cutoff_local == cutoff_local_last)
                {
                    break;
                }
            }
            while (cutoff_local > CUTOFF); // 10^-10

            ensemble_m();
            ensemble_E();

            // printf("\nblah = %lf", h[jj_S]);
            // printf("\nblam = %lf", m[jj_S]);
            // printf("\n");
            
            fprintf(pFile_1, "%lf\t ", h[jj_S]);
            for(j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "%lf\t ", m[j_S]);
            }
            fprintf(pFile_1, "%lf\t ", E);

            fprintf(pFile_1, "\n");
        }

        fclose(pFile_1);
        free(spin_temp);
        return 0;
    }

    int zero_temp_RFXY_hysteresis_rotate_checkerboard(int jj_S, double order_start)
    {
        T = 0;

        printf("\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");
        fprintf(pFile_1, "\nUpdating all (first)black/(then)white checkerboard sites simultaneously.. \n");

        double cutoff_local = 0.0;
        int j_S, j_L;
        double *m_last = (double*)malloc(dim_S*sizeof(double));
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_last[j_S] = 2;
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[j_S] = order_start;
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
        double h_start = order[jj_S]*(sigma_h[0]/4.0);
        double h_theta = 0;
        printf("\nztne RFXY h rotating with |h|=%lf at T=%lf..", h_start, T);

        long int site_i;
        int black_or_white = 0;

        int repeat_loop = 1;
        int repeat_cond = 1;
        while (repeat_cond)
        {

            for (h_theta = 0.0; h_theta * order[jj_S] <= 1.0; h_theta = h_theta + order[jj_S] * delta_h)
            {
                cutoff_local = -0.1;
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*h_theta);
                    h[1] = h_start * sin(2*pie*h_theta);
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*h_theta);
                    h[1] = h_start * cos(2*pie*h_theta);
                }

                do
                {
                    // double cutoff_local_last = cutoff_local;
                    cutoff_local = 0.0;

                    // #pragma omp parallel 
                    // {
                    //     #pragma omp for reduction(+:cutoff_local)
                    //     for (site_i=0; site_i<no_of_black_sites; site_i++)
                    //     {
                    //         long int site_index = black_white_checkerboard[0][site_i];
                    //         double spin_local[dim_S];

                    //         cutoff_local += update_to_minimum_checkerboard(site_index, spin_local);
                    //     }
                    // }

                    // ensemble_m();
                    // printf("blam = %lf,", m[jj_S]);
                    // printf("blac=%.17g\n", cutoff_local);
                    // if (cutoff_local == cutoff_local_last)
                    // {
                    //     break;
                    // }
                    // else
                    // {
                    //     cutoff_local_last = cutoff_local;
                    // }
                    

                    #pragma omp parallel 
                    {
                        #pragma omp for reduction(+:cutoff_local)
                        for (site_i=0; site_i<no_of_black_white_sites[black_or_white]; site_i++)
                        {
                            long int site_index = black_white_checkerboard[black_or_white][site_i];
                            double spin_local[dim_S];

                            cutoff_local += update_to_minimum_checkerboard(site_index, spin_local);
                        }

                        #pragma omp for reduction(+:cutoff_local)
                        for (site_i=0; site_i<no_of_black_white_sites[!black_or_white]; site_i++)
                        {
                            long int site_index = black_white_checkerboard[!black_or_white][site_i];
                            double spin_local[dim_S];

                            cutoff_local += update_to_minimum_checkerboard(site_index, spin_local);
                        }
                        
                    }
                    // ensemble_m();
                    // printf("blam = %lf,", m[jj_S]);
                    // printf("blac=%.17g\n", cutoff_local);
                    
                    // black_or_white = !black_or_white;
                    
                    // if (cutoff_local == cutoff_local_last)
                    // {
                    //     break;
                    // }
                    
                }
                while (cutoff_local > CUTOFF); // 10^-10

                ensemble_m();
                ensemble_E();
                
                printf("\nblah = %lf", h[jj_S]);
                printf("\nblam = %lf", m[jj_S]);
                printf("\n");

                fprintf(pFile_1, "%lf\t ", h_theta);
                fprintf(pFile_1, "%lf\t %lf\t ", h[0], h[1]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E);

                fprintf(pFile_1, "\n");
                
                // ----------------------------------------------//
                // if (h_theta * order[jj_S] + delta_h > 1.0)
                // {
                //     for(j_S=0; j_S<dim_S; j_S++)
                //     {
                //         if (fabs(m_last[j_S] - m[j_S]) > CUTOFF )
                //         {
                //             h_theta = -delta_h* order[jj_S];
                //         }
                //         m_last[j_S] = m[j_S];
                //     }
                //     fprintf(pFile_1, "loop %d\n", repeat_loop);
                //     printf("\nloop %d\n", repeat_loop);
                //     repeat_loop++;
                // }
            }

            repeat_cond = 0;
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (fabs(m_last[j_S] - m[j_S]) > CUTOFF )
                {
                    repeat_cond = 1;
                }
                m_last[j_S] = m[j_S];
            }
            fprintf(pFile_1, "loop %d\n", repeat_loop);
            printf("\nloop %d\n", repeat_loop);
            repeat_loop++;
        }
            
        return 0;
    }

    int zero_temp_RFXY_hysteresis_rotate(int jj_S, double order_start)
    {
        T = 0;
        
        printf("\nUpdating all sites simultaneously.. \n");
        fprintf(pFile_1, "\nUpdating all sites simultaneously.. \n");
        
        spin_temp = (double*)malloc(dim_S*no_of_sites*sizeof(double));

        double cutoff_local = 0.0;
        int j_S, j_L;
        double *m_last = (double*)malloc(dim_S*sizeof(double));
        for (j_S=0; j_S<dim_S; j_S++)
        {
            m_last[j_S] = 2;
        }
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[j_S] = order_start;
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
        double h_start = order[jj_S]*(sigma_h[0]/4.0);
        double h_theta = 0;
        printf("\nztne RFXY h rotating with |h|=%lf at T=%lf..", h_start, T);

        long int site_i;
        int black_or_white = 0;

        int repeat_loop = 1;
        int repeat_cond = 1;
        while (repeat_cond)
        {

            for (h_theta = 0.0; h_theta * order[jj_S] <= 1.0; h_theta = h_theta + order[jj_S] * delta_h)
            {
                if (jj_S == 0)
                {
                    h[0] = h_start * cos(2*pie*h_theta);
                    h[1] = h_start * sin(2*pie*h_theta);
                }
                else
                {
                    h[0] = -h_start * sin(2*pie*h_theta);
                    h[1] = h_start * cos(2*pie*h_theta);
                }
                
                cutoff_local = -0.1;
                do
                {
                    double cutoff_local_last = cutoff_local;
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
                            cutoff_local += fabs(spin[site_i] - spin_temp[site_i]);
                            spin[site_i] = spin_temp[site_i];
                        }
                    }
                    // ensemble_m();
                    // printf("blam = %lf,", m[jj_S]);
                    // printf("blac=%.17g\n", cutoff_local);
                    
                    if (cutoff_local == cutoff_local_last)
                    {
                        break;
                    }
                }
                while (cutoff_local > CUTOFF); // 10^-10

                ensemble_m();
                ensemble_E();
                
                printf("\nblah = %lf", h[jj_S]);
                printf("\nblam = %lf", m[jj_S]);
                printf("\n");

                fprintf(pFile_1, "%lf\t ", h_theta);
                fprintf(pFile_1, "%lf\t %lf\t ", h[0], h[1]);
                for(j_S=0; j_S<dim_S; j_S++)
                {
                    fprintf(pFile_1, "%lf\t ", m[j_S]);
                }
                fprintf(pFile_1, "%lf\t ", E);

                fprintf(pFile_1, "\n");
                
                // ----------------------------------------------//
                // if (h_theta * order[jj_S] + delta_h > 1.0)
                // {
                //     for(j_S=0; j_S<dim_S; j_S++)
                //     {
                //         if (fabs(m_last[j_S] - m[j_S]) > CUTOFF )
                //         {
                //             h_theta = -delta_h* order[jj_S];
                //         }
                //         m_last[j_S] = m[j_S];
                //     }
                //     fprintf(pFile_1, "loop %d\n", repeat_loop);
                //     printf("\nloop %d\n", repeat_loop);
                //     repeat_loop++;
                // }
            }
            
            repeat_cond = 0;
            for(j_S=0; j_S<dim_S; j_S++)
            {
                if (fabs(m_last[j_S] - m[j_S]) > CUTOFF )
                {
                    repeat_cond = 1;
                }
                m_last[j_S] = m[j_S];
            }
            fprintf(pFile_1, "loop %d\n", repeat_loop);
            printf("\nloop %d\n", repeat_loop);
            repeat_loop++;
        }
        free(spin_temp);
        return 0;
    }

    int ordered_initialize_and_rotate(int jj_S, double order_start)
    {
        T = 0;
        double cutoff_local = 0.0;
        int j_S, j_L;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[j_S] = order_start;
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
        double h_start = order[jj_S]*(h_i_max/32);
        double h_theta = 0;
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
            // char output_file_1[256];
            char *pos = output_file_1;
            pos += sprintf(pos, "O(%d)_%dD_hys_rot_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%lf_{", T);
            /* for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{"); */
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
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "_%lf}.dat", delta_h);
            
        }
        pFile_1 = fopen(output_file_1, "a");

        // print column heaser
        {
            fprintf(pFile_1, "theta(h[:])\t ");
            fprintf(pFile_1, "h[0]\t ");
            fprintf(pFile_1, "h[1]\t ");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_(Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                if (j_S==jj_S)
                {
                    fprintf(pFile_1, "-");
                }
                else
                {
                    fprintf(pFile_1, "%lf", h[j_S]);
                }
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t MCS/{/Symbol d}h=%ld/%lf\t \n", h_order, r_order, hysteresis_MCS, delta_h);
        }
        long int site_i;
        T = 0;
        printf("\nztne RFXY, h rotating with |h|=%lf at T=%lf.. \n", h_start, T);
        
        zero_temp_RFXY_hysteresis_rotate(jj_S, order_start);

        fclose(pFile_1);
        
        return 0;
    }

    int field_cool_and_rotate(int jj_S, double order_start)
    {
        // random initialization
        int j_S, j_L;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[j_S] = order_start;
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
        // start from h[0] or h[1] != 0
        double h_start = order[jj_S]*(sigma_h[0]/4.0);
        h[jj_S] = h_start;
        double h_theta = 0;
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
            // char output_file_1[256];
            char *pos = output_file_1;
            pos += sprintf(pos, "O(%d)_%dD_hys_rot_fcool_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%lf,%lf_{", Temp_max, Temp_min);
            /* for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{"); */
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
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "_%lf}.dat", delta_h);
        }
        pFile_1 = fopen(output_file_1, "a");

        // cooling_protocol T_MAX - T_MIN=0
        // print column heaser
        {
            
            fprintf(pFile_1, "T\t |m|\t ");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            } 
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "<Y[%d,%d][%d]>\t ", j_S, j_SS, j_L);
                    }
                }
            } */
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_{Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                    fprintf(pFile_1, "%lf", h[j_S]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t (thermalizing-MCS,averaging-MCS)/{/Symbol D}T=(%ld,%ld)/%lf\t \n", h_order, r_order, thermal_i, average_j, delta_T);
        }
        cooling_protocol();
        fclose(pFile_1);

        pFile_1 = fopen(output_file_1, "a");
        
        // rotate field
        // print column heaser
        {
            fprintf(pFile_1, "\ntheta(h[:])\t ");
            fprintf(pFile_1, "h[0]\t ");
            fprintf(pFile_1, "h[1]\t ");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_(Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                if (j_S==jj_S)
                {
                    fprintf(pFile_1, "-");
                }
                else
                {
                    fprintf(pFile_1, "%lf", h[j_S]);
                }
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t MCS/{/Symbol d}h=%ld/%lf\t \n", h_order, r_order, hysteresis_MCS, delta_h);
        }
        zero_temp_RFXY_hysteresis_rotate(jj_S, order_start);
        fclose(pFile_1);
        
        return 0;
    }

    int field_cool_and_rotate_checkerboard(int jj_S, double order_start)
    {
        // random initialization
        int j_S, j_L;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[j_S] = order_start;
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
        // start from h[0] or h[1] != 0
        double h_start = order[jj_S]*(sigma_h[0]/4.0);
        h[jj_S] = h_start;
        double h_theta = 0.0;
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
            // char output_file_1[256];
            char *pos = output_file_1;
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
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{"); */
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
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "_%lf}.dat", delta_h);
        }
        pFile_1 = fopen(output_file_1, "a");
        
        // cooling_protocol T_MAX - T_MIN=0
        // print column heaser
        {
            
            fprintf(pFile_1, "T\t |m|\t ");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            } 
            /* for (j_S=0; j_S<dim_S; j_S++)
            {
                for (j_SS=0; j_SS<dim_S; j_SS++)
                {
                    for (j_L=0; j_L<dim_L; j_L++)
                    {
                        fprintf(pFile_1, "<Y[%d,%d][%d]>\t ", j_S, j_SS, j_L);
                    }
                }
            } */
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_{Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                    fprintf(pFile_1, "%lf", h[j_S]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t (thermalizing-MCS,averaging-MCS)/{/Symbol D}T=(%ld,%ld)/%lf\t \n", h_order, r_order, thermal_i, average_j, delta_T);
        }
        cooling_protocol();
        fclose(pFile_1);
        
        pFile_1 = fopen(output_file_1, "a");
        // rotate field
        // print column heaser
        {
            fprintf(pFile_1, "\ntheta(h[:])\t ");
            fprintf(pFile_1, "h[0]\t ");
            fprintf(pFile_1, "h[1]\t ");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_(Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                if (j_S==jj_S)
                {
                    fprintf(pFile_1, "-");
                }
                else
                {
                    fprintf(pFile_1, "%lf", h[j_S]);
                }
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t MCS/{/Symbol d}h=%ld/%lf\t \n", h_order, r_order, hysteresis_MCS, delta_h);
        }
        zero_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start);
        fclose(pFile_1);
        
        return 0;
    }

    int random_initialize_and_rotate(int jj_S, double order_start)
    {
        T = 0;
        int j_S, j_L;

        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[j_S] = order_start;
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
        double h_start = order[jj_S]*(sigma_h[0]/4.0);
        double h_theta = 0;
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
            // char output_file_1[256];
            char *pos = output_file_1;
            pos += sprintf(pos, "O(%d)_%dD_hys_rot_rand_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%lf_{", T);
            /* for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{"); */
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
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "_%lf}.dat", delta_h);
        }
        pFile_1 = fopen(output_file_1, "a");

        // print column heaser
        {
            fprintf(pFile_1, "theta(h[:])\t ");
            fprintf(pFile_1, "h[0]\t ");
            fprintf(pFile_1, "h[1]\t ");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_(Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                if (j_S==jj_S)
                {
                    fprintf(pFile_1, "-");
                }
                else
                {
                    fprintf(pFile_1, "%lf", h[j_S]);
                }
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t MCS/{/Symbol d}h=%ld/%lf\t \n", h_order, r_order, hysteresis_MCS, delta_h);
        }
        zero_temp_RFXY_hysteresis_rotate(jj_S, order_start);
        fclose(pFile_1);

        return 0;
    }

    int random_initialize_and_rotate_checkerboard(int jj_S, double order_start)
    {
        T = 0;

        int j_S, j_L;
        
        for (j_S=0; j_S<dim_S; j_S++)
        {
            if (j_S == jj_S)
            {
                order[j_S] = order_start;
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
        double h_start = order[jj_S]*(sigma_h[0]/4.0);
        double h_theta = 0;
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
            // char output_file_1[256];
            char *pos = output_file_1;
            pos += sprintf(pos, "O(%d)_%dD_hys_rot_rand_", dim_S, dim_L);

            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L) 
                {
                    pos += sprintf(pos, "x");
                }
                pos += sprintf(pos, "%d", lattice_size[j_L]);
            }
            pos += sprintf(pos, "_%lf_{", T);
            /* for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_J[j_L]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_L = 0 ; j_L != dim_L ; j_L++) 
            {
                if (j_L)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", J_dev_avg[j_L]);
            }
            pos += sprintf(pos, "}_{"); */
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
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", sigma_h[j_S]);
            }
            pos += sprintf(pos, "}_{");    
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", h_dev_avg[j_S]);
            }
            pos += sprintf(pos, "}_{");
            for (j_S = 0 ; j_S != dim_S ; j_S++) 
            {
                if (j_S)
                {
                    pos += sprintf(pos, ",");
                }
                pos += sprintf(pos, "%lf", order[j_S]);
            }
            pos += sprintf(pos, "_%lf}.dat", delta_h);
        }
        pFile_1 = fopen(output_file_1, "a");

        // print column heaser
        { 
            fprintf(pFile_1, "theta(h[:])\t ");
            fprintf(pFile_1, "h[0]\t ");
            fprintf(pFile_1, "h[1]\t ");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                fprintf(pFile_1, "<m[%d]>\t ", j_S);
            }
            fprintf(pFile_1, "<E>\t dim_{Lat}=%d\t L=", dim_L);
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%d", lattice_size[j_L]);
            }
            fprintf(pFile_1, "\t J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J[j_L]);
            }
            fprintf(pFile_1, "\t {/Symbol s}_J=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_J[j_L]);
            }
            fprintf(pFile_1, "\t <J_{ij}>=");
            for (j_L=0; j_L<dim_L; j_L++)
            {
                if (j_L)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", J_dev_avg[j_L]);
            }
            fprintf(pFile_1, "\t dim_(Spin}=%d\t h=", dim_S);
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                if (j_S==jj_S)
                {
                    fprintf(pFile_1, "-");
                }
                else
                {
                    fprintf(pFile_1, "%lf", h[j_S]);
                }
            }
            fprintf(pFile_1, "\t {/Symbol s}_h=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", sigma_h[j_S]);
            }
            fprintf(pFile_1, "\t <h_i>=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", h_dev_avg[j_S]);
            }
            fprintf(pFile_1, "\t order=");
            for (j_S=0; j_S<dim_S; j_S++)
            {
                if (j_S)
                {
                    fprintf(pFile_1, ",");
                }
                fprintf(pFile_1, "%lf", order[j_S]);
            }
            fprintf(pFile_1, "\t order_h=%d\t order_r=%d\t MCS/{/Symbol d}h=%ld/%lf\t \n", h_order, r_order, hysteresis_MCS, delta_h);
        }
        zero_temp_RFXY_hysteresis_rotate_checkerboard(jj_S, order_start);
        fclose(pFile_1);
        
        return 0;
    }

//===============================================================================//

//===============================================================================//
//====================      Main                             ====================//

    int free_memory()
    {

        free(N_N_I);
        
        free(black_white_checkerboard[0]);
        
        free(black_white_checkerboard[1]);
        
        free(h_random);
        
        free(J_random);
        
        free(spin);    
        
        free(cluster);

        free(sorted_h_index);

        free(next_in_queue);

        free(spin_old);

        free(spin_new);

        free(field_site);

        return 0;
    }

    int main()
    {
        
        double start_time = omp_get_wtime();
        int j_L, j_S;
        // no_of_sites = custom_int_pow(lattice_size, dim_L);
        initialize_checkerboard_sites();
        
        initialize_nearest_neighbor_index();
        printf("nearest neighbor initialized. \n");
        
        load_h_config();
        printf("h loaded. \n");

        load_J_config();
        printf("J loaded. \n");
        
        spin = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        cluster = (int*)malloc(no_of_sites*sizeof(int));
        sorted_h_index = (long int*)malloc(no_of_sites*sizeof(long int));
        next_in_queue = (long int*)malloc((1+no_of_sites)*sizeof(long int));
        spin_old = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        spin_new = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        field_site = (double*)malloc(dim_S*no_of_sites*sizeof(double));
        long int i, j;

        thermal_i = thermal_i*lattice_size[0];
        average_j = average_j*lattice_size[0];
        
        printf("L = %d, dim_L = %d, dim_S = %d\n", lattice_size[0], dim_L, dim_S); 
        
        printf("hysteresis_MCS_multiplier = %ld, hysteresis_MCS_max = %ld\n", hysteresis_MCS_multiplier, hysteresis_MCS_max); 

        srand(time(NULL));

        printf("RAND_MAX = %lf,\n sizeof(int) = %ld,\n sizeof(long) = %ld,\n sizeof(double) = %ld,\n sizeof(long int) = %ld,\n sizeof(short int) = %ld,\n sizeof(unsigned int) = %ld,\n sizeof(RAND_MAX) = %ld\n", (double)RAND_MAX, sizeof(int), sizeof(long), sizeof(double), sizeof(long int), sizeof(short int), sizeof(unsigned int), sizeof(RAND_MAX));
        
        num_of_threads = omp_get_max_threads();
        num_of_procs = omp_get_num_procs();
        random_seed = (unsigned int*)malloc(cache_size*num_of_threads*sizeof(unsigned int));
        random_seed[0] = rand();
        printf("No. of THREADS = %d\n", num_of_threads);
        printf("No. of PROCESSORS = %d\n", num_of_procs);
        for (i=1; i < num_of_threads; i++)
        {
            random_seed[i] = rand_r(&random_seed[cache_size*(i-1)]);
        }
        double *start_time_loop = (double*)malloc((num_of_threads)*sizeof(double)); 
        double *end_time_loop = (double*)malloc((num_of_threads)*sizeof(double)); 
        
        for (i=2; i<=num_of_threads; i+=2)
        {
            start_time_loop[i] = omp_get_wtime();
            omp_set_num_threads(i);
            printf("\n\nNo. of THREADS = %ld \n\n", i);
            field_cool_and_rotate_checkerboard(0, 1);
            // random_initialize_and_rotate_checkerboard(0, 1);
            end_time_loop[i] = omp_get_wtime();
        }
        
        for (i=2; i<=num_of_threads; i+=2)
        {
            printf("No. of THREADS = %ld ,\t Time elapsed = %g\n", i, end_time_loop[i]-start_time_loop[i]);
        }
        
        // random_initialize_and_rotate(0, 1);
        // field_cool_and_rotate(0, 1);
        // zero_temp_RFXY_hysteresis_axis(0, -1);
        // zero_temp_RFXY_hysteresis_axis(1, 1);
        // zero_temp_RFXY_hysteresis_rotate(0, 1);
        
        
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
                        hysteresis_protocol(j_S);
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
                hysteresis_protocol(0);
            }
        } */

        free_memory();
        double end_time = omp_get_wtime();
        printf("\nCPU  Start Time = %lf \n", start_time);
        printf("\nCPU  End Time = %lf \n", end_time);
        
        printf("\nCPU Time elapsed total = %lf \n", end_time-start_time);
        return 0;
    }

//===============================================================================//

