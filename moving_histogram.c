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

    long size;
    char *buf;
    char *ptr;

    size = pathconf(".", _PC_PATH_MAX);

    if ((buf = (char *)malloc((size_t)size)) != NULL)
        ptr = getcwd(buf, (size_t)size);
    
    for (si=0; si<len_append_str; si++)
    {
        char folder_name[64];
        char *pos_0 = folder_name;
        // pos += sprintf(pos, buf, i);
        pos_0 += sprintf(pos_0, "combined_%s", append_string[si] );

        printf("\nfolder = %s : ", folder_name );

        int ret = chdir(folder_name);
        if (ret)
        { // same as ret!=0, means an error occurred and errno is set
            printf("error!\n"); 
            return 1;
        }

        for (Li=0; Li<len_L_vals; Li++)
        {
            printf("\n L = %d :\n", L_vals[Li] );

            for (hi=0; hi<len_h_field_vals; hi++)
            {
                printf("sigma_h = %lf, ", h_field_vals[hi] );
                int i,j;
                double X;
                
                double hist_max = 0;
                
                double hist_min = 10;

                int no_of_bins = 100;
                int no_of_sub_bins = 100;
                double *bin = (double*)malloc((no_of_sub_bins*(no_of_bins+2))*sizeof(double));
                
                char input_file_1[256];
                char *pos_1 = input_file_1;
                pos_1 += sprintf(pos_1, "o_r_O2_2D_%d_%lf_all_%s.dat", L_vals[Li], h_field_vals[hi], append_string[si] );

                pFile_input = fopen(input_file_1, "r");
                
                while(fscanf(pFile_input, "%le", &X)==1)
                {
                    if (X > 0)
                    {
                        if(X < hist_min)
                        {
                            hist_min = X;
                        }
                        if (X > hist_max)
                        {
                            hist_max = X;
                        }
                    }
                }
                fclose(pFile_input);

                double bin_width = (hist_max - hist_min)/((double) no_of_bins);
                hist_min = hist_min - bin_width;
                hist_max = hist_max + bin_width;
                double sub_bin_width = bin_width/((double) no_of_sub_bins);

                for (i=0; i<no_of_sub_bins*(no_of_bins+2); i=i+1)
                {
                    bin[i] = 0;      
                }

                pFile_input = fopen(input_file_1, "r");
                while(fscanf(pFile_input, "%le", &X)==1)
                {
                    double P = (X - hist_min)/sub_bin_width;
                    if( (P - (double) ((int) P)) != 0.0)
                    {
                        bin[(int) P] = bin[(int) P] + 1;
                    }
                }
                fclose(pFile_input);

                char output_file_2[256];
                char *pos_2 = output_file_2;
                pos_2 += sprintf(pos_2, "hist_O2_2D_%d_%lf_all_%s.dat", L_vals[Li], h_field_vals[hi], append_string[si] );

                pFile_output = fopen(output_file_2, "a");
                
                for (i=0; i<no_of_sub_bins*(no_of_bins+1)+1; i=i+1)
                {
                    for (j=1; j<no_of_sub_bins;j=j+1)
                    {
                        bin[i] += bin[i+j];
                    }
                    bin[i] = bin[i]/no_of_sub_bins;
                    fprintf(pFile_output,"%le\t%le\n", hist_min + (bin_width/2.0) + sub_bin_width*((double) (i)), bin[i]);
                }

                fclose(pFile_output);
            }
        }
        chdir(ptr);
    }
    return 0;
}