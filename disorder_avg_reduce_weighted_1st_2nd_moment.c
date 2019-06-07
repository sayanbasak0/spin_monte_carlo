#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
// #include "mt19937-64.h"
#include <omp.h>
#include <unistd.h> // chdir 
#include <errno.h> // strerror
#include <sys/types.h>
#include <sys/stat.h>

// #define enable_CUDA_CODE
#ifdef enable_CUDA_CODE
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif


FILE *pFile_input_1, *pFile_input_2, *pFile_output_1, *pFile_output_2, *pFile_output_3;

double CUTOFF = 0.000000000001;
double CUTOFF_PHI = 0.00000001;

int main(int argc, char** argv)
{
    int i, j, k, l, input, i_col;
    if (argc == 1)
    {
        printf("Argument required! ");
        printf("\nOptions: 1(L=64), 2(L=80), 3(L=100), 4(L=128), 5(L=160), <<anything else>>(Quit)\n");
        printf("\nYour Answer: ");
        scanf("%d", &input);
    }
    else if (argc > 2)
    {
        printf("Too many arguments! ");
        printf("\nOptions: 1(L=64), 2(L=80), 3(L=100), 4(L=128), 5(L=160), <<anything else>>(Quit)\n");
        printf("\nYour Answer:");
        scanf("%d", &input);
    }
    else
    {
        input = atoi(argv[1]);
    }

    if (input > 5 || input < 1)
    {
        return 0;
    }
    else
    {
        input -= 1;
    }

    double h_field_vals[] = { 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.048, 0.050, 0.052, 0.054, 0.056, 0.058, 0.060, 0.064, 0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150 };
    int len_h_field_vals = sizeof(h_field_vals) / sizeof(h_field_vals[0]);
    int hi;

    int L_vals[] = { 64, 80, 100, 128, 160 };
    // int L_vals[] = { 64 };
    int len_L_vals = sizeof(L_vals) / sizeof(L_vals[0]);
    int Li;
    
    char *append_string[] = { "Mx", "My", "M", "absM" };
    int len_append_str = sizeof(append_string) / sizeof(append_string[0]);
    int si;

    int config_vals[] = { 75, 60, 50, 40, 30 };
    // int L_vals[] = { 64 };
    int len_config_vals = sizeof(config_vals) / sizeof(config_vals[0]);
    int ci;

    long size;
    char *buf;
    char *ptr;

    size = pathconf(".", _PC_PATH_MAX);

    if ((buf = (char *)malloc((size_t)size)) != NULL)
        ptr = getcwd(buf, (size_t)size);

    // ------------------------ Input 3d array ------------------------ //
    int no_of_columns_in = 8;
    
    double ***array_tr;
    double ***array_lc;
    array_tr = (double ***)malloc(config_vals[input]*sizeof(double **));
    array_lc = (double ***)malloc(config_vals[input]*sizeof(double **));
    
    for (ci=0; ci<config_vals[input] ; ci++)
    {
        array_tr[ci] = (double **)malloc(len_h_field_vals*sizeof(double *));
        array_lc[ci] = (double **)malloc(len_h_field_vals*sizeof(double *));
        for (hi=0; hi<len_h_field_vals; hi++)
        {
            array_tr[ci][hi] = (double *)malloc(no_of_columns_in*sizeof(double));
            array_lc[ci][hi] = (double *)malloc(no_of_columns_in*sizeof(double));
        }
    }
    // ------------------------ Output 2d array ------------------------ //
    int no_of_columns_out = 14;

    double **array_tr_out;
    double **array_lc_out;
    double **array_to_out;
    double **array_tr_out_stdev;
    double **array_lc_out_stdev;
    double **array_to_out_stdev;
    array_tr_out = (double **)malloc(len_h_field_vals*sizeof(double *));
    array_lc_out = (double **)malloc(len_h_field_vals*sizeof(double *));
    array_to_out = (double **)malloc(len_h_field_vals*sizeof(double *));
    array_tr_out_stdev = (double **)malloc(len_h_field_vals*sizeof(double *));
    array_lc_out_stdev = (double **)malloc(len_h_field_vals*sizeof(double *));
    array_to_out_stdev = (double **)malloc(len_h_field_vals*sizeof(double *));
    for (hi=0; hi<len_h_field_vals; hi++)
    {
        array_tr_out[hi] = (double *)malloc(no_of_columns_out*sizeof(double));
        array_lc_out[hi] = (double *)malloc(no_of_columns_out*sizeof(double));
        array_to_out[hi] = (double *)malloc(no_of_columns_out*sizeof(double));
        array_tr_out_stdev[hi] = (double *)malloc(no_of_columns_out*sizeof(double));
        array_lc_out_stdev[hi] = (double *)malloc(no_of_columns_out*sizeof(double));
        array_to_out_stdev[hi] = (double *)malloc(no_of_columns_out*sizeof(double));
        for (i_col=0; i_col<no_of_columns_in; i_col++)
        {
            array_tr_out[hi][i_col] = 0.0 ;
            array_lc_out[hi][i_col] = 0.0 ;
            array_to_out[hi][i_col] = 0.0 ;
            array_tr_out_stdev[hi][i_col] = 0.0 ;
            array_lc_out_stdev[hi][i_col] = 0.0 ;
            array_to_out_stdev[hi][i_col] = 0.0 ;
        }
    }

    for (ci=0; ci<config_vals[input] ; ci++)
    {
        char folder_name[128];
        char *pos_0 = folder_name;
        // pos += sprintf(pos, buf, i);
        pos_0 += sprintf(pos_0, "config_%d", ci );

        printf("\rfolder = %s : ", folder_name );
        fflush(stdout);

        int ret = chdir(folder_name);
        if (ret)
        { // same as ret!=0, means an error occurred and errno is set
            printf("error!\n"); 
            return 1;
        }

        Li = input;
        printf("\rfolder = %s : L = %d :                                           ", folder_name, L_vals[input] );
        fflush(stdout);
        
        double X;
        
        // ------------------------ store reduced data in 3d array ------------------------ //
        
        char input_file_1[256];
        char *pos_in_1 = input_file_1;
        pos_in_1 += sprintf(pos_in_1, "transient_O2_2D_%d_reduced_weighted_1st_2nd.dat", L_vals[Li] );

        pFile_input_1 = fopen(input_file_1, "r");
        
        for (hi=0; hi<len_h_field_vals; hi++)
        {
            for (i_col=0; i_col<no_of_columns_in; i_col++)
            {
                fscanf(pFile_input_1, "%le\t", &X );
                array_tr[ci][hi][i_col] = X;
            }
        }

        fclose(pFile_input_1);
        printf("\rfolder = %s : L = %d : Input file: transient.                         ", folder_name, L_vals[input] );
        fflush(stdout);
        
        char input_file_2[256];
        char *pos_in_2 = input_file_2;
        pos_in_2 += sprintf(pos_in_2, "limit_cycle_O2_2D_%d_reduced_weighted_1st_2nd.dat", L_vals[Li] );

        pFile_input_2 = fopen(input_file_2, "r");

        for (hi=0; hi<len_h_field_vals; hi++)
        {
            for (i_col=0; i_col<no_of_columns_in; i_col++)
            {
                fscanf(pFile_input_2, "%le\t", &X );
                array_lc[ci][hi][i_col] = X;
            }
        }
                    
        fclose(pFile_input_2);

        printf("\rfolder = %s : L = %d : Input file: limit cycle.                    ", folder_name, L_vals[input] );
        fflush(stdout);

        chdir(ptr);
    }
    free(buf);

    // ------------------------ disorder sum 3d array to 2d array ------------------------ //
    for (ci=0; ci<config_vals[input] ; ci++)
    {
        for (hi=0; hi<len_h_field_vals; hi++)
        {
            array_tr_out[hi][0] = h_field_vals[hi];
            array_lc_out[hi][0] = h_field_vals[hi];
            array_to_out[hi][0] = h_field_vals[hi];
            
            for (i_col=1; i_col<no_of_columns_in; i_col++)
            {
                array_tr_out[hi][i_col] += array_tr[ci][hi][i_col] ;
                array_tr_out_stdev[hi][i_col] += array_tr[ci][hi][i_col] * array_tr[ci][hi][i_col] ;
                
                array_lc_out[hi][i_col] += array_lc[ci][hi][i_col] ;
                array_lc_out_stdev[hi][i_col] += array_lc[ci][hi][i_col] * array_lc[ci][hi][i_col] ;

                array_to_out[hi][i_col] += (array_tr[ci][hi][i_col] + array_lc[ci][hi][i_col]) ;
                array_to_out_stdev[hi][i_col] += (array_tr[ci][hi][i_col] + array_lc[ci][hi][i_col]) * (array_tr[ci][hi][i_col] + array_lc[ci][hi][i_col]) ;
            }
            
            
            for (i_col=2; i_col<no_of_columns_in; i_col++)
            {
                if (array_tr[ci][hi][1] > 0)
                {
                    array_tr_out[hi][i_col+6] += array_tr[ci][hi][i_col] / array_tr[ci][hi][1];
                    array_tr_out_stdev[hi][i_col+6] += (array_tr[ci][hi][i_col] / array_tr[ci][hi][1]) * (array_tr[ci][hi][i_col] / array_tr[ci][hi][1]);
                }
                array_lc_out[hi][i_col+6] += (array_lc[ci][hi][i_col] / array_lc[ci][hi][1]);
                array_lc_out_stdev[hi][i_col+6] += (array_lc[ci][hi][i_col] / array_lc[ci][hi][1]) * (array_lc[ci][hi][i_col] / array_lc[ci][hi][1]);

                array_to_out[hi][i_col+6] += (array_tr[ci][hi][i_col] + array_lc[ci][hi][i_col]) / (array_tr[ci][hi][1] + array_lc[ci][hi][1]);
                array_to_out_stdev[hi][i_col+6] += ((array_tr[ci][hi][i_col] + array_lc[ci][hi][i_col]) / (array_tr[ci][hi][1] + array_lc[ci][hi][1])) * ((array_tr[ci][hi][i_col] + array_lc[ci][hi][i_col]) / (array_tr[ci][hi][1] + array_lc[ci][hi][1]));
            }
        }
    }
    printf(" Disorder Averaging...      \n" );
    fflush(stdout);

    // ------------------------ disorder std dev 3d array to 2d array ------------------------ //
    /* for (ci=0; ci<config_vals[input] ; ci++)
    {
        for (hi=0; hi<len_h_field_vals; hi++)
        {
            array_tr_out_stdev[hi][0] = h_field_vals[hi];
            array_lc_out_stdev[hi][0] = h_field_vals[hi];
            array_to_out_stdev[hi][0] = h_field_vals[hi];
            
            for (i_col=1; i_col<no_of_columns_in-3; i_col++)
            {
                array_tr_out_stdev[hi][i_col] += ( array_tr[ci][hi][i_col] - array_tr_out[hi][i_col] / (double)config_vals[input] ) * ( array_tr[ci][hi][i_col] - array_tr_out[hi][i_col] / (double)config_vals[input] );
                array_lc_out_stdev[hi][i_col] += ( array_lc[ci][hi][i_col] - array_lc_out[hi][i_col] / (double)config_vals[input] ) * ( array_lc[ci][hi][i_col] - array_lc_out[hi][i_col] / (double)config_vals[input] );
                array_to_out_stdev[hi][i_col] += ( (array_tr[ci][hi][i_col] + array_lc[ci][hi][i_col]) - array_to_out[hi][i_col] / (double)config_vals[input] ) * ( (array_tr[ci][hi][i_col] + array_lc[ci][hi][i_col]) - array_to_out[hi][i_col] / (double)config_vals[input] );
            }
            
            for (i_col=no_of_columns_in-3; i_col<no_of_columns_in; i_col++)
            {
                array_tr_out_stdev[hi][i_col] += (array_tr[ci][hi][i_col] - array_tr_out[hi][i_col] / (double)config_vals[input]) * (array_tr[ci][hi][i_col] - array_tr_out[hi][i_col] / (double)config_vals[input]) ;
                array_lc_out_stdev[hi][i_col] += (array_lc[ci][hi][i_col] - array_lc_out[hi][i_col] / (double)config_vals[input]) * (array_lc[ci][hi][i_col] - array_lc_out[hi][i_col] / (double)config_vals[input]) ;

                if (array_tr[ci][hi][i_col] > array_lc[ci][hi][i_col])
                {
                    array_to_out_stdev[hi][i_col] += (array_tr[ci][hi][i_col] - array_to_out[hi][i_col] / (double)config_vals[input]) * (array_tr[ci][hi][i_col] - array_to_out[hi][i_col] / (double)config_vals[input]);
                }
                else
                {
                    array_to_out_stdev[hi][i_col] += (array_lc[ci][hi][i_col] - array_to_out[hi][i_col] / (double)config_vals[input]) * (array_lc[ci][hi][i_col] - array_to_out[hi][i_col] / (double)config_vals[input]) ;
                }
            }
            
            for (i_col=2; i_col<no_of_columns_in-3; i_col++)
            {
                if (array_tr[ci][hi][1] > 0)
                {
                    array_tr_out[hi][i_col+6] += array_tr[ci][hi][i_col] / array_tr[ci][hi][1];
                }
                array_lc_out[hi][i_col+6] += array_lc[ci][hi][i_col] / array_lc[ci][hi][1];
                array_to_out[hi][i_col+6] += (array_tr[ci][hi][i_col] + array_lc[ci][hi][i_col]) / (array_tr[ci][hi][1] + array_lc[ci][hi][1]);
            }
            for (i_col=no_of_columns_in-3; i_col<no_of_columns_in; i_col++)
            {
                if ( array_tr[ci][hi][i_col] > array_tr_out[hi][i_col+6] )
                {
                    array_tr_out[hi][i_col+6] = array_tr[ci][hi][i_col] ;
                }
                if ( array_lc[ci][hi][i_col] > array_lc_out[hi][i_col+6] )
                {
                    array_lc_out[hi][i_col+6] = array_lc[ci][hi][i_col] ;
                }
                if ( array_tr[ci][hi][i_col] > array_lc[ci][hi][i_col] )
                {
                    if ( array_tr[ci][hi][i_col] > array_to_out[hi][i_col+6] )
                    {
                        array_to_out[hi][i_col+6] = (array_tr[ci][hi][i_col] ) ;
                    }
                }
                else
                {
                    if ( array_lc[ci][hi][i_col] > array_to_out[hi][i_col+6] )
                    {
                        array_to_out[hi][i_col+6] = (array_lc[ci][hi][i_col] ) ;
                    }

                }
            }
        }
    } */
    printf(" Disorder Std. deviation...      \n" );
    fflush(stdout);
    // ------------------------ disorder average 2d array and output ------------------------ //

    char output_file_1[256];
    char *pos_out_1 = output_file_1;
    pos_out_1 += sprintf(pos_out_1, "transient_O2_2D_%d_dis_avg_weighted_1st_2nd.dat", L_vals[Li] );

    pFile_output_1 = fopen(output_file_1, "a");
    
    char output_file_2[256];
    char *pos_out_2 = output_file_2;
    pos_out_2 += sprintf(pos_out_2, "limit_cycle_O2_2D_%d_dis_avg_weighted_1st_2nd.dat", L_vals[Li] );

    pFile_output_2 = fopen(output_file_2, "a");
    
    char output_file_3[256];
    char *pos_out_3 = output_file_3;
    pos_out_3 += sprintf(pos_out_3, "total_O2_2D_%d_dis_avg_weighted_1st_2nd.dat", L_vals[Li] );

    pFile_output_3 = fopen(output_file_3, "a");
    
    fprintf(pFile_output_1, "-L-\t" );
    fprintf(pFile_output_1, "---------H----------\t" );
    fprintf(pFile_output_1, "-------<dphi>-------\t" );
    fprintf(pFile_output_1, "------s<dphi>-------\t" );
    fprintf(pFile_output_1, "-----<dmx^2/dfi>----\t" );
    fprintf(pFile_output_1, "----s<dmx^2/dfi>----\t" );
    fprintf(pFile_output_1, "-----<dmy^2/dfi>----\t" );
    fprintf(pFile_output_1, "----s<dmy^2/dfi>----\t" );
    fprintf(pFile_output_1, "-----<dm^2/dfi>-----\t" );
    fprintf(pFile_output_1, "----s<dm^2/dfi>-----\t" );
    fprintf(pFile_output_1, "------<max(dmx)>----\t" );
    fprintf(pFile_output_1, "-----s<max(dmx)>----\t" );
    fprintf(pFile_output_1, "------<max(dmy)>----\t" );
    fprintf(pFile_output_1, "-----s<max(dmy)>----\t" );
    fprintf(pFile_output_1, "------<max(dm)>-----\t" );
    fprintf(pFile_output_1, "-----s<max(dm)>-----\t" );
    fprintf(pFile_output_1, "-<(dmx^2/dfi)/delfi>\t" );
    fprintf(pFile_output_1, "s<(dmx^2/dfi)/delfi>\t" );
    fprintf(pFile_output_1, "-<(dmy^2/dfi)/delfi>\t" );
    fprintf(pFile_output_1, "s<(dmy^2/dfi)/delfi>\t" );
    fprintf(pFile_output_1, "-<(dm^2/dfi)/delfi>-\t" );
    fprintf(pFile_output_1, "s<(dm^2/dfi)/delfi>-\t" );
    fprintf(pFile_output_1, "----max[max(dmx)]---\t" );
    fprintf(pFile_output_1, "----max[max(dmy)]---\t" );
    fprintf(pFile_output_1, "----max[max(dm)]----\t" );
    fprintf(pFile_output_1, "\n" );
    
    fprintf(pFile_output_2, "-L-\t" );
    fprintf(pFile_output_2, "---------H----------\t" );
    fprintf(pFile_output_2, "-------<dphi>-------\t" );
    fprintf(pFile_output_2, "------s<dphi>-------\t" );
    fprintf(pFile_output_2, "-----<dmx^2/dfi>----\t" );
    fprintf(pFile_output_2, "----s<dmx^2/dfi>----\t" );
    fprintf(pFile_output_2, "-----<dmy^2/dfi>----\t" );
    fprintf(pFile_output_2, "----s<dmy^2/dfi>----\t" );
    fprintf(pFile_output_2, "-----<dm^2/dfi>-----\t" );
    fprintf(pFile_output_2, "----s<dm^2/dfi>-----\t" );
    fprintf(pFile_output_2, "------<max(dmx)>----\t" );
    fprintf(pFile_output_2, "-----s<max(dmx)>----\t" );
    fprintf(pFile_output_2, "------<max(dmy)>----\t" );
    fprintf(pFile_output_2, "-----s<max(dmy)>----\t" );
    fprintf(pFile_output_2, "------<max(dm)>-----\t" );
    fprintf(pFile_output_2, "-----s<max(dm)>-----\t" );
    fprintf(pFile_output_2, "-<(dmx^2/dfi)/delfi>\t" );
    fprintf(pFile_output_2, "s<(dmx^2/dfi)/delfi>\t" );
    fprintf(pFile_output_2, "-<(dmy^2/dfi)/delfi>\t" );
    fprintf(pFile_output_2, "s<(dmy^2/dfi)/delfi>\t" );
    fprintf(pFile_output_2, "-<(dm^2/dfi)/delfi>-\t" );
    fprintf(pFile_output_2, "s<(dm^2/dfi)/delfi>-\t" );
    fprintf(pFile_output_2, "----max[max(dmx)]---\t" );
    fprintf(pFile_output_2, "----max[max(dmy)]---\t" );
    fprintf(pFile_output_2, "----max[max(dm)]----\t" );
    fprintf(pFile_output_2, "\n" );
    
    fprintf(pFile_output_3, "-L-\t" );
    fprintf(pFile_output_3, "---------H----------\t" );
    fprintf(pFile_output_3, "-------<dphi>-------\t" );
    fprintf(pFile_output_3, "------s<dphi>-------\t" );
    fprintf(pFile_output_3, "-----<dmx^2/dfi>----\t" );
    fprintf(pFile_output_3, "----s<dmx^2/dfi>----\t" );
    fprintf(pFile_output_3, "-----<dmy^2/dfi>----\t" );
    fprintf(pFile_output_3, "----s<dmy^2/dfi>----\t" );
    fprintf(pFile_output_3, "-----<dm^2/dfi>-----\t" );
    fprintf(pFile_output_3, "----s<dm^2/dfi>-----\t" );
    fprintf(pFile_output_3, "------<max(dmx)>----\t" );
    fprintf(pFile_output_3, "-----s<max(dmx)>----\t" );
    fprintf(pFile_output_3, "------<max(dmy)>----\t" );
    fprintf(pFile_output_3, "-----s<max(dmy)>----\t" );
    fprintf(pFile_output_3, "------<max(dm)>-----\t" );
    fprintf(pFile_output_3, "-----s<max(dm)>-----\t" );
    fprintf(pFile_output_3, "-<(dmx^2/dfi)/delfi>\t" );
    fprintf(pFile_output_3, "s<(dmx^2/dfi)/delfi>\t" );
    fprintf(pFile_output_3, "-<(dmy^2/dfi)/delfi>\t" );
    fprintf(pFile_output_3, "s<(dmy^2/dfi)/delfi>\t" );
    fprintf(pFile_output_3, "-<(dm^2/dfi)/delfi>-\t" );
    fprintf(pFile_output_3, "s<(dm^2/dfi)/delfi>-\t" );
    fprintf(pFile_output_3, "----max[max(dmx)]---\t" );
    fprintf(pFile_output_3, "----max[max(dmy)]---\t" );
    fprintf(pFile_output_3, "----max[max(dm)]----\t" );
    fprintf(pFile_output_3, "\n" );

    for (hi=0; hi<len_h_field_vals; hi++)
    {
        
        fprintf(pFile_output_1, "%d\t", L_vals[Li] );
        fprintf(pFile_output_1, "%.14e\t", h_field_vals[hi] );

        fprintf(pFile_output_2, "%d\t", L_vals[Li] );
        fprintf(pFile_output_2, "%.14e\t", h_field_vals[hi] );

        fprintf(pFile_output_3, "%d\t", L_vals[Li] );
        fprintf(pFile_output_3, "%.14e\t", h_field_vals[hi] );

        for (i_col=1; i_col<no_of_columns_out; i_col++)
        {
            fprintf(pFile_output_1, "%.14e\t", array_tr_out[hi][i_col] / (double)config_vals[input] );
            fprintf(pFile_output_1, "%.14e\t", sqrt(array_tr_out_stdev[hi][i_col] / (double)config_vals[input] - (array_tr_out[hi][i_col] / (double)config_vals[input])*(array_tr_out[hi][i_col] / (double)config_vals[input]) ) );
            fprintf(pFile_output_2, "%.14e\t", array_lc_out[hi][i_col] / (double)config_vals[input] );
            fprintf(pFile_output_2, "%.14e\t", sqrt(array_lc_out_stdev[hi][i_col] / (double)config_vals[input] - (array_lc_out[hi][i_col] / (double)config_vals[input])*(array_lc_out[hi][i_col] / (double)config_vals[input]) ) );
            fprintf(pFile_output_3, "%.14e\t", array_to_out[hi][i_col] / (double)config_vals[input] );
            fprintf(pFile_output_3, "%.14e\t", sqrt(array_to_out_stdev[hi][i_col] / (double)config_vals[input] - (array_to_out[hi][i_col] / (double)config_vals[input])*(array_to_out[hi][i_col] / (double)config_vals[input]) ) );
        }

        // for (i_col=no_of_columns_out-3; i_col<no_of_columns_out; i_col++)
        // {
        //     fprintf(pFile_output_1, "%.14e\t", array_tr_out[hi][i_col] );
        //     fprintf(pFile_output_2, "%.14e\t", array_lc_out[hi][i_col] );
        //     fprintf(pFile_output_3, "%.14e\t", array_to_out[hi][i_col] );
        // }
        fprintf(pFile_output_1, "\n" );
        fprintf(pFile_output_2, "\n" );
        fprintf(pFile_output_3, "\n" );

        printf("\r Output to file. H=%f...                                               ", h_field_vals[hi] );
        fflush(stdout);
    }
    fclose(pFile_output_1);
    fclose(pFile_output_2);
    fclose(pFile_output_3);

    // ------------------------ Free pointers ------------------------ //

    for (ci=0; ci<config_vals[input] ; ci++)
    {
        for (hi=0; hi<len_h_field_vals; hi++)
        {
            free( array_tr[ci][hi] );
            free( array_lc[ci][hi] );
        }
        free( array_tr[ci] );
        free( array_lc[ci] );
    }
    free( array_tr );
    free( array_lc );

    for (hi=0; hi<len_h_field_vals; hi++)
    {
        free( array_tr_out[hi] );
        free( array_lc_out[hi] );
        free( array_to_out[hi] );
        free( array_tr_out_stdev[hi] );
        free( array_lc_out_stdev[hi] );
        free( array_to_out_stdev[hi] );
    }
    free( array_tr_out );
    free( array_lc_out );
    free( array_to_out );
    free( array_tr_out_stdev );
    free( array_lc_out_stdev );
    free( array_to_out_stdev );

    printf("\n------------------------------------- Done.-------------------------------------\n");
    fflush(stdout);

    // free(ptr);
    return 0;
}