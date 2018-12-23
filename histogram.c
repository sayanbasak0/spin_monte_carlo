#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
// #include "mt19937-64.h"
#include <unistd.h> // chdir 
#include <errno.h> // strerror
#include <sys/types.h>
#include <sys/stat.h>

FILE *pFile_input, *pFile_output;

int main()
{
    srand(time(NULL));
    double h_field_vals[] = { 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.048, 0.050, 0.052, 0.054, 0.056, 0.058, 0.060, 0.064, 0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150 };
    // double h_field_vals[] = { 0.010 };
    int len_h_field_vals = sizeof(h_field_vals) / sizeof(h_field_vals[0]);
    int hi;

    int L_vals[] = { 64, 80, 100, 128, 160 };
    // int L_vals[] = { 64 };
    int len_L_vals = sizeof(L_vals) / sizeof(L_vals[0]);
    int Li;

    char *append_string[] = { "Mx", "My", "E", "M" };
    int len_append_str = sizeof(append_string) / sizeof(append_string[0]);
    int si;
    
    for (hi=0; hi<len_h_field_vals; hi++)
    {
        printf("\nsigma_h = %lf :\n", h_field_vals[hi] );
        for (Li=0; Li<len_L_vals; Li++)
        {
            printf("L = %d\n", L_vals[Li] );
            for (si=0; si<len_append_str; si++)
            {
                int i;
                double X;
                double log_max = 0;
                double hist_max = 0;
                double log_min = 10;
                double hist_min = 10;

                int no_of_bins = 101;
                long int *bin = (long int*)malloc((no_of_bins+1)*sizeof(long int));
                double *log_bin = (double*)malloc((no_of_bins+1)*sizeof(double));
                
                char input_file_1[256];
                char *pos_1 = input_file_1;
                pos_1 += sprintf(pos_1, "o_r_O2_2D_%d_%lf_all_%s.dat", L_vals[Li], h_field_vals[hi], append_string[si] );

                pFile_input = fopen(input_file_1, "r");
                
                while(fscanf(pFile_input, "%le", &X)==1)
                {
                    if (X > 0)
                    {
                        if(log(X) < log_min)
                        {
                            log_min = log(X);
                        }
                        if (log(X) > log_max)
                        {
                            log_max = log(X);
                        }
                    }
                }
                fclose(pFile_input);

                double bin_width = (log_max - log_min)/(no_of_bins-1);
                hist_min = log_min - bin_width/2;
                hist_max = log_max + bin_width/2;
                bin_width = (hist_max - hist_min)/(no_of_bins);

                for (i=0; i<no_of_bins; i=i+1)
                {
                    bin[i] = 0;      
                }
                for (i=0; i<no_of_bins; i=i+1)
                {
                    log_bin[i] = -1.0;      
                }

                pFile_input = fopen(input_file_1, "r");
                while(fscanf(pFile_input, "%le", &X)==1)
                {
                    if (X > 0)
                    {
                        int P = (log(X) - hist_min)/bin_width;
                        bin[P+1] = bin[P+1] + 1;
                    }
                    else
                    {
                        bin[0] = bin[0] + 1;
                    }
                }
                fclose(pFile_input);

                for (i=0; i<no_of_bins+1; i=i+1)
                {
                    if (bin[i]>0)
                    {
                        log_bin[i] = log((double) bin[i]);
                    }
                }

                char output_file_2[256];
                char *pos_2 = output_file_2;
                pos_2 += sprintf(pos_2, "hist_O2_2D_%d_%lf_all_%s.dat", L_vals[Li], h_field_vals[hi], append_string[si] );

                pFile_output = fopen(output_file_2, "a");
                fprintf(pFile_output,"%le\t%le\n", hist_min + bin_width*((double) -4.0), -1.0);
                fprintf(pFile_output,"%le\t%le\n", hist_min + bin_width*((double) -4.0), log_bin[0]);
                fprintf(pFile_output,"%le\t%le\n", hist_min + bin_width*((double) 0.0), log_bin[0]);
                fprintf(pFile_output,"%le\t%le\n", hist_min + bin_width*((double) 0.0), -1.0);
                for (i=1; i<no_of_bins+1; i=i+1)
                {
                    fprintf(pFile_output,"%le\t%le\n", hist_min + bin_width*((double) (i-1)), log_bin[i]);
                    fprintf(pFile_output,"%le\t%le\n", hist_min + bin_width*((double) i), log_bin[i]);
                    fprintf(pFile_output,"%le\t%le\n", hist_min + bin_width*((double) i), -1.0);
                }
                fclose(pFile_output);
            }
        }
    }
    return 0;
}