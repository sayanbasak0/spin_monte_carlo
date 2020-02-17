#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "mt19937-64.h"
#include <unistd.h> // chdir 
#include <errno.h> // strerror
#include <sys/types.h>
#include <sys/stat.h>


#define MARSAGLIA 1 // uncomment only one
// #define REJECTION 1 // uncomment only one
// #define BOX_MULLER 1 // uncomment only one

#define dim_L 2
#define dim_S 2

FILE *pFile_1;

const double pie = 3.14159265358979323846;

// int dim_L = 2;
// int dim_S = 2;
int lattice_size[dim_L] = { 100, 100 };
long int no_of_sites = 1;

int no_of_disorder_configs = 1;

long int *N_N_I;

double *h_random;
double sigma_h[dim_S] = { 0.50, 0.00 }; // only parameter for distribution

double h_i_max = 0.0, h_i_min = 0.0; // for hysteresis
double h[dim_S] = { 0.0, 0.0 }; 

double h_dev_net[dim_S];
double h_dev_avg[dim_S];

double *J_random;
double sigma_J[dim_L] = { 0.0, 0.0 }; // only parameter for distribution

double J_i_max = 0.0, J_i_min = 0.0; // for hysteresis
double J[dim_L] = { 1.0, 1.0 }; 

double J_dev_net[dim_L];
double J_dev_avg[dim_L];

long int custom_int_pow(long int base, int power)
{
    if (power > 0)
    {
        return base * custom_int_pow(base, (power - 1));
    }
     return 1.0;
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
    
    fprintf(pFile_1, "%le ", h_i_min);
    printf( "\nh_i_min=%lf ", h_i_min);
    // fprintf(pFile_1, "\n");
    fprintf(pFile_1, "%le ", h_i_max);
    printf( "h_i_max=%lf \n", h_i_max);
    fprintf(pFile_1, "\n");

    for (j_S=0; j_S<dim_S; j_S++)
    {
        fprintf(pFile_1, "%le ", h[j_S]);
        fprintf(pFile_1, "%le ", sigma_h[j_S]);
        printf( "sigma_h[%d]=%lf \n", j_S, sigma_h[j_S]);
        fprintf(pFile_1, "%le ", h_dev_avg[j_S]);
        printf( "h_dev_avg[%d]=%lf \n", j_S, h_dev_avg[j_S]);
        fprintf(pFile_1, "\n");
    }
    fprintf(pFile_1, "\n");

    for (i = 0; i < no_of_sites; i++)
    {
        for (j_S = 0; j_S<dim_S; j_S++)
        {
            fprintf(pFile_1, "%.17e ", h_random[dim_S*i + j_S]);
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

    h_random = (double*)malloc(dim_S*no_of_sites*sizeof(double));
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
    //---------------------------------------------------------------------------------------//
    
    for (i = 0; i < no_of_sites; i++)
    {
        for (j_S = 0; j_S<dim_S; j_S++)
        {
            printf("|%lf|", h_random[dim_S*i + j_S]);
        }
        printf("\n");
    }
    printf("\n");
    
    

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
    
    fprintf(pFile_1, "%le ", J_i_min);
    // fprintf(pFile_1, "\n");
    fprintf(pFile_1, "%le ", J_i_max);
    fprintf(pFile_1, "\n");

    for (j_L=0; j_L<dim_L; j_L++)
    {
        fprintf(pFile_1, "%le ", J[j_L]);
        fprintf(pFile_1, "%le ", sigma_J[j_L]);
        fprintf(pFile_1, "%le ", J_dev_avg[j_L]);
        fprintf(pFile_1, "\n");
    }
    fprintf(pFile_1, "\n");

    for (i = 0; i < no_of_sites; i++)
    {
        for (j_L = 0; j_L<dim_L; j_L++)
        {
            for (k_L = 0; k_L<2; k_L++)
            {
                fprintf(pFile_1, "%le ", J_random[2*dim_L*i + 2*j_L + k_L]);
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

    J_random = (double*)malloc(2*dim_L*no_of_sites*sizeof(double));
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
    //---------------------------------------------------------------------------------------//
    
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
    
    

    return 0;
}

int free_memory()
{
    if(N_N_I!=NULL)
    {
        free(N_N_I);
    }
    /* 
    if(black_checkerboard!=NULL)
    {
        free(black_checkerboard);
    }
    if(white_checkerboard!=NULL)
    {
        free(white_checkerboard);
    }
    if(spin!=NULL)
    { 
        free(spin);
    } */
    if(h_random!=NULL)
    {
        free(h_random);
    }
    if(J_random!=NULL)
    {
        free(J_random);
    }

    return 0;
}

int main()
{
    int j_L, j_S, i;
    long initial_seed ;
    long *initial_seed_ptr = &initial_seed;

    char seed_file[128];
    char *pos = seed_file;
    pos += sprintf(pos, "h_seed_");
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
    
    strcat(seed_file, ".dat");
        
    pFile_1 = fopen(seed_file, "r");
    if (pFile_1 == NULL)
    {
        pFile_1 = fopen(seed_file, "a");
        initial_seed = (long) time(initial_seed_ptr);
        
        fprintf(pFile_1, "%ld", initial_seed );
        fprintf(pFile_1, "\n" );
        fclose(pFile_1);
    }
    else
    {
        fscanf(pFile_1, "%ld", &initial_seed );
        fclose(pFile_1);
    }

    srand(initial_seed);
    printf("\n----------\n initial_seed = %ld \n first call to rand() = %d \n----------\n", *initial_seed_ptr, rand());
    srand(initial_seed);
    
    
    // unsigned long long init[4]={0x12345ULL, 0x23456ULL, 0x34567ULL, 0x45678ULL}, length=4;
    // unsigned long long init[4]={(unsigned long long)rand() * 0x12345ULL, (unsigned long long)rand() * 0x23456ULL, (unsigned long long)rand() * 0x34567ULL, (unsigned long long)rand() * 0x45678ULL}, length=4;
    init_genrand64( (unsigned long long) rand() );
    // init_by_array64(init, length);
    // printf("%lld %lld %lld %lld \n", init[0], init[1], init[2], init[3] );
    
    
    
    for (j_L=0; j_L<dim_L; j_L++)
    {
        no_of_sites = no_of_sites*lattice_size[j_L];
    }
    printf("%ld\n", no_of_sites);

    initialize_nearest_neighbor_index();

    long size;
    char *buf;
    char *ptr;

    size = pathconf(".", _PC_PATH_MAX);

    if ((buf = (char *)malloc((size_t)size)) != NULL)
        ptr = getcwd(buf, (size_t)size);

    for (i=0; i<=no_of_disorder_configs; i++)
    {
        // load_h_config();
        // save_J_config();
        // load_J_config();
        
        
        char folder_name[64];
        char *pos = folder_name;
        // pos += sprintf(pos, buf, i);
        pos += sprintf(pos, "config_%d", i);
        
        mkdir(folder_name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        int ret = chdir(folder_name);
        if (ret){ // same as ret!=0, means an error occurred and errno is set
            printf("error!\n");
        }
        else
        {
            save_h_config();
        }
        chdir(ptr);

    }

    free_memory();

    return 0;
}
