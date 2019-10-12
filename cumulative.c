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
    // double h_field_vals[] = { 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.048, 0.050, 0.052, 0.054, 0.056, 0.058, 0.060, 0.064, 0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150 };
    double h_field_vals[] = { 0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150 };
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
                
                
                char input_file_1[256];
                char *pos_1 = input_file_1;
                pos_1 += sprintf(pos_1, "o_r_O2_2D_%d_%lf_all_%s.dat", L_vals[Li], h_field_vals[hi], append_string[si] );

                pFile_input = fopen(input_file_1, "r");
                long int total_counter = 0;
                while(fscanf(pFile_input, "%le", &X)==1)
                {
                    total_counter += 1;
                }
                fclose(pFile_input);

                double X_old = 0;
                long int cumul_counter = 0;
                
                pFile_input = fopen(input_file_1, "r");

                char output_file_2[256];
                char *pos_2 = output_file_2;
                pos_2 += sprintf(pos_2, "cumul_O2_2D_%d_%lf_all_%s.dat", L_vals[Li], h_field_vals[hi], append_string[si] );

                pFile_output = fopen(output_file_2, "a");

                while(fscanf(pFile_input, "%le", &X)==1)
                {
                    if (X > X_old )
                    {
                        fprintf(pFile_output, "%le\t%le\n", X_old, ((double) cumul_counter)/((double) total_counter) );
                    }
                    
                    cumul_counter += 1;
                    X_old = X;
                }
                                
                fprintf(pFile_output, "%le\t%le\n", X_old, ((double) cumul_counter)/((double) total_counter) );
                
                fclose(pFile_output);
                fclose(pFile_input);
            }
        }
    }
    return 0;
}