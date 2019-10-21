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


FILE *pFile_input_1, *pFile_input_2, *pFile_output_1, *pFile_output_2, *pFile_output_3, *pFile_output[3];

double CUTOFF = 0.000000000001;
double CUTOFF_PHI = 0.00000001;
double delta_phi_max = 0.0001;
double delta_phi_min;
char folder_name[128];
int hi;
int Li;
int si;
int ci;


double h_field_vals[] = { 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.048, 0.050, 0.052, 0.054, 0.056, 0.058, 0.060, 0.064, 0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150 };
int L_vals[] = { 64, 80, 100, 128, 160 };
char *append_string[] = { "Mx", "My", "M", "absM" };
int config_vals[] = { 75, 60, 50, 40, 30 };
int input;
double phi_loop[2];
double phi_loop_aligned[2];
long int nth_loop_start[2] = { 0, 0 };
long int nth_loop_array_size[2] = { 0, 0 } ;

double **phi ;// = (double **)malloc((loop_no+1)*sizeof(double *));
double **mx ;// = (double **)malloc((loop_no+1)*sizeof(double *));
double **my ;// = (double **)malloc((loop_no+1)*sizeof(double *));
double **del_phi ;// = (double **)malloc((loop_no+1)*sizeof(double *));
double **del_mx ;// = (double **)malloc((loop_no+1)*sizeof(double *));
double **del_my ;// = (double **)malloc((loop_no+1)*sizeof(double *));

int save_to_file(int i_loop, long int j_array, long int j_bkp)
{
    
    if (j_array > nth_loop_start[i_loop])
    {
        fprintf(pFile_output[i_loop], "%.14e\t", phi[i_loop][j_array]-phi[i_loop][j_bkp]);
        fprintf(pFile_output[i_loop], "%.14e\t", mx[i_loop][j_array]-mx[i_loop][j_bkp]);
        fprintf(pFile_output[i_loop], "%.14e\t", my[i_loop][j_array]-my[i_loop][j_bkp]);
        fprintf(pFile_output[i_loop], "\n");
    }
    
    
    fprintf(pFile_output[i_loop], "%.14e\t", phi[i_loop][j_array]);
    fprintf(pFile_output[i_loop], "%.14e\t", mx[i_loop][j_array]);
    fprintf(pFile_output[i_loop], "%.14e\t", my[i_loop][j_array]);
    // fprintf(pFile_output[i_loop], "\n");

    if (j_array == (nth_loop_start[i_loop]+nth_loop_array_size[i_loop]-1) )
    {
        fprintf(pFile_output[i_loop], "%.14e\t", phi[i_loop][j_array+1]-phi[i_loop][j_array]);
        fprintf(pFile_output[i_loop], "%.14e\t", mx[i_loop][j_array+1]-mx[i_loop][j_array]);
        fprintf(pFile_output[i_loop], "%.14e\t", my[i_loop][j_array+1]-my[i_loop][j_array]);
        fprintf(pFile_output[i_loop], "\n");
    }

    // printf("\rfolder = %s : L = %d : sigma_h = %lf... phi = %.10f                ", folder_name, L_vals[input], h_field_vals[hi], phi[i_loop][j_array]);
    // fflush(stdout);
    return 0;
}

long int subdivide_phi(double del_phi_temp, int i_loop, long int j_array)
{
    long int j_bkp = j_array;
    double dphi_temp = phi[i_loop][j_array] - phi[i_loop][j_bkp];
    // while (fabs(del_phi_temp - dphi_temp) > delta_phi_min/2)
    while (del_phi_temp - dphi_temp > delta_phi_min/2)
    {
        if (j_array == nth_loop_start[i_loop] + nth_loop_array_size[i_loop]-1)
        {
            break;
        }
        j_array++;
        dphi_temp = phi[i_loop][j_array] - phi[i_loop][j_bkp];
    }

    // if (del_phi_temp < delta_phi_min)
    if (fabs(del_phi_temp - delta_phi_min) < delta_phi_min/2)
    {
        save_to_file(i_loop, j_array, j_bkp);
        j_bkp = j_array;
        return j_bkp;
    }
    double dmx_temp = mx[i_loop][j_array]-mx[i_loop][j_bkp];
    double dmy_temp = my[i_loop][j_array]-my[i_loop][j_bkp];
    double dm_temp = sqrt(dmx_temp*dmx_temp + dmy_temp*dmy_temp);

    if (dm_temp * L_vals[input]*L_vals[input] <= 2.56)
    {
        save_to_file(i_loop, j_array, j_bkp);
        j_bkp = j_array;
        return j_bkp;
    }
    else // if (dm_temp * L_vals[input]*L_vals[input] > 2.56)
    {
        j_bkp = subdivide_phi(del_phi_temp/2, i_loop, j_bkp);
        j_bkp = subdivide_phi(del_phi_temp/2, i_loop, j_bkp);
        return j_bkp;
    }
    return 0;
}

int main(int argc, char** argv)
{
    delta_phi_max = 0.0001;
    delta_phi_min = delta_phi_max ;
    while (delta_phi_min > CUTOFF_PHI)
    {
        delta_phi_min = delta_phi_min / 2.0;
    }

    // int i, j, k, l, input;
    int i, j, k, l;
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

    // double h_field_vals[] = { 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.048, 0.050, 0.052, 0.054, 0.056, 0.058, 0.060, 0.064, 0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150 };
    int len_h_field_vals = sizeof(h_field_vals) / sizeof(h_field_vals[0]);
    // int hi;

    // int L_vals[] = { 64, 80, 100, 128, 160 };
    // int L_vals[] = { 64 };
    int len_L_vals = sizeof(L_vals) / sizeof(L_vals[0]);
    // int Li;
    
    // char *append_string[] = { "Mx", "My", "M", "absM" };
    int len_append_str = sizeof(append_string) / sizeof(append_string[0]);
    // int si;

    // int config_vals[] = { 75, 60, 50, 40, 30 };
    // int L_vals[] = { 64 };
    int len_config_vals = sizeof(config_vals) / sizeof(config_vals[0]);
    // int ci;

    long size;
    char *buf;
    char *ptr;

    size = pathconf(".", _PC_PATH_MAX);

    if ((buf = (char *)malloc((size_t)size)) != NULL)
        ptr = getcwd(buf, (size_t)size);
    
    for (ci=0; ci<config_vals[input] ; ci++)
    {
        // char folder_name[128];
        folder_name[0] = '\0';
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
        printf("\rfolder = %s : L = %d : ", folder_name,L_vals[input] );
        fflush(stdout);
            

        for (hi=0; hi<len_h_field_vals; hi++)
        {
            printf("\rfolder = %s : L = %d : sigma_h = %lf... ", folder_name, L_vals[input], h_field_vals[hi] );
            fflush(stdout);

            char input_file_1[256];
            char *pos_in_1 = input_file_1;
            pos_in_1 += sprintf(pos_in_1, "transient_O2_2D_%d_%lf.dat", L_vals[Li], h_field_vals[hi] );
            
            char input_file_2[256];
            char *pos_in_2 = input_file_2;
            pos_in_2 += sprintf(pos_in_2, "limit_cycle_O2_2D_%d_%lf.dat", L_vals[Li], h_field_vals[hi] );

            int loop_no = 1;
            // long int nth_loop_array_size[] = { 0, 0 } ;
            nth_loop_array_size[0] = 0;
            nth_loop_array_size[1] = 0;
            int transient_loops = 0;
            int limit_cycle = 1;

            double X = 0.0, X_old = -1.0;
            
            printf("\rfolder = %s : L = %d : sigma_h = %lf... Starting......                      ", folder_name,L_vals[input], h_field_vals[hi] );
            fflush(stdout);
            // ------------------------ no. of data points in each file ------------------------ //

            pFile_input_1 = fopen(input_file_1, "r");
            while(fscanf(pFile_input_1, "%le", &X)==1)
            {
                nth_loop_array_size[transient_loops] += 1;

                fscanf(pFile_input_1, "%le", &X);
                fscanf(pFile_input_1, "%le", &X);
                fscanf(pFile_input_1, "%le", &X);
                fscanf(pFile_input_1, "%le", &X);
                fscanf(pFile_input_1, "%le", &X);
            }
            fclose(pFile_input_1);

            pFile_input_2 = fopen(input_file_2, "r");
            while(fscanf(pFile_input_2, "%le", &X)==1)
            {
                nth_loop_array_size[limit_cycle] += 1;

                fscanf(pFile_input_2, "%le", &X);
                fscanf(pFile_input_2, "%le", &X);
                fscanf(pFile_input_2, "%le", &X);
                fscanf(pFile_input_2, "%le", &X);
                fscanf(pFile_input_2, "%le", &X);
            }
            fclose(pFile_input_2);

            printf("\rfolder = %s : L = %d : sigma_h = %lf... Determined data points.             ", folder_name, L_vals[input], h_field_vals[hi] );
            fflush(stdout);
            // ------------------------ collect data in arrays ------------------------ //

            phi = (double **)malloc((loop_no+1)*sizeof(double *));
            mx = (double **)malloc((loop_no+1)*sizeof(double *));
            my = (double **)malloc((loop_no+1)*sizeof(double *));
            del_phi = (double **)malloc((loop_no+1)*sizeof(double *));
            del_mx = (double **)malloc((loop_no+1)*sizeof(double *));
            del_my = (double **)malloc((loop_no+1)*sizeof(double *));

            for (i=0; i<=loop_no; i++)
            {
                phi[i] = (double *)malloc((nth_loop_array_size[i] + nth_loop_array_size[1])*sizeof(double));
                mx[i] = (double *)malloc((nth_loop_array_size[i] + nth_loop_array_size[1])*sizeof(double));
                my[i] = (double *)malloc((nth_loop_array_size[i] + nth_loop_array_size[1])*sizeof(double));
                del_phi[i] = (double *)malloc((nth_loop_array_size[i] + nth_loop_array_size[1])*sizeof(double));
                del_mx[i] = (double *)malloc((nth_loop_array_size[i] + nth_loop_array_size[1])*sizeof(double));
                del_my[i] = (double *)malloc((nth_loop_array_size[i] + nth_loop_array_size[1])*sizeof(double));
            }

            int i_loop=0;
            long int j_array=0;
            pFile_input_1 = fopen(input_file_1, "r");
            while(fscanf(pFile_input_1, "%le", &X)==1)
            {
                phi[transient_loops][j_array] = X;
                fscanf(pFile_input_1, "%le", &X);
                mx[transient_loops][j_array] = X;
                fscanf(pFile_input_1, "%le", &X);
                my[transient_loops][j_array] = X;
                
                fscanf(pFile_input_1, "%le", &X);
                // del_phi[transient_loops][j_array] = X;
                fscanf(pFile_input_1, "%le", &X);
                // del_mx[transient_loops][j_array] = X;
                fscanf(pFile_input_1, "%le", &X);
                // del_my[transient_loops][j_array] = X;
                
                j_array += 1;
            }
            fclose(pFile_input_1);

            i_loop=1;
            j_array=0;
            pFile_input_2 = fopen(input_file_2, "r");
            while(fscanf(pFile_input_2, "%le", &X)==1)
            {
                phi[limit_cycle][j_array] = X;
                fscanf(pFile_input_2, "%le", &X);
                mx[limit_cycle][j_array] = X;
                fscanf(pFile_input_2, "%le", &X);
                my[limit_cycle][j_array] = X;
                
                fscanf(pFile_input_2, "%le", &X);
                // del_phi[limit_cycle][j_array] = X;
                fscanf(pFile_input_2, "%le", &X);
                // del_mx[limit_cycle][j_array] = X;
                fscanf(pFile_input_2, "%le", &X);
                // del_my[limit_cycle][j_array] = X;
                
                j_array += 1;
            }
            fclose(pFile_input_2);

            // double phi_loop[2];
            phi_loop[limit_cycle] = phi[limit_cycle][nth_loop_array_size[limit_cycle]-1] - phi[limit_cycle][0];
            phi_loop[transient_loops] = phi[transient_loops][nth_loop_array_size[transient_loops]-1] - phi[transient_loops][0];

            for (j_array=0; j_array<nth_loop_array_size[1]-1; j_array++)
            {
                phi[limit_cycle][j_array+nth_loop_array_size[limit_cycle]] = phi_loop[limit_cycle] + phi[limit_cycle][j_array+1];
                mx[limit_cycle][j_array+nth_loop_array_size[limit_cycle]] = mx[limit_cycle][j_array+1];
                my[limit_cycle][j_array+nth_loop_array_size[limit_cycle]] = my[limit_cycle][j_array+1];
                
                phi[transient_loops][j_array+nth_loop_array_size[transient_loops]] = phi[limit_cycle][j_array+1];
                mx[transient_loops][j_array+nth_loop_array_size[transient_loops]] = mx[limit_cycle][j_array+1];
                my[transient_loops][j_array+nth_loop_array_size[transient_loops]] = my[limit_cycle][j_array+1];
            }
            printf("\rfolder = %s : L = %d : sigma_h = %lf... Collected data.                     ", folder_name, L_vals[input], h_field_vals[hi] );
            fflush(stdout);

            // double phi_loop_aligned[2];
            nth_loop_start[0] = 0;
            nth_loop_start[1] = 0;
            phi_loop_aligned[0] = delta_phi_max * ceil( phi[transient_loops][nth_loop_array_size[transient_loops]-1] / delta_phi_max ) ;
            j_array = nth_loop_array_size[transient_loops]-1;
            for (; phi_loop_aligned[0] > phi[0][j_array]; )
            {
                j_array++;
                nth_loop_start[limit_cycle] += 1;
                nth_loop_array_size[transient_loops] += 1;
            }
            

            // ------------------------ rereconstruct data in arrays ------------------------ //

            char output_file_1[256];
            char *pos_out_1 = output_file_1;
            pos_out_1 += sprintf(pos_out_1, "transient_O2_2D_%d_%lf_reconstruct.dat", L_vals[Li], h_field_vals[hi] );

            pFile_output[0] = fopen(output_file_1, "a");
            
            char output_file_2[256];
            char *pos_out_2 = output_file_2;
            pos_out_2 += sprintf(pos_out_2, "limit_cycle_O2_2D_%d_%lf_reconstruct.dat", L_vals[Li], h_field_vals[hi] );

            pFile_output[1] = fopen(output_file_2, "a");
            
            for (i_loop=0; i_loop<=loop_no; i_loop++)
            {
                double phi_temp = phi[i_loop][0];
                j_array = nth_loop_start[i_loop];
                save_to_file(i_loop, j_array, j_array);

                // while (phi_temp < phi[i_loop][nth_loop_array_size[i_loop]-1])
                while (j_array < nth_loop_start[i_loop] + nth_loop_array_size[i_loop]-1)
                {
                    double del_phi_temp = delta_phi_max;

                    j_array = subdivide_phi(del_phi_temp, i_loop, j_array);

                }

            }

            fclose(pFile_output[0]);
            fclose(pFile_output[1]);
            
            printf("\rfolder = %s : L = %d : sigma_h = %lf... Stored data in file.                ", folder_name, L_vals[input], h_field_vals[hi] );
            fflush(stdout);
            // ------------------------ Free pointers ------------------------ //

            for (i=0; i<=loop_no; i++)
            {
                free(phi[i]);
                free(mx[i]);
                free(my[i]);
                free(del_phi[i]);
                free(del_mx[i]);
                free(del_my[i]);
            }
            free(phi);
            free(mx);
            free(my);
            free(del_phi);
            free(del_mx);
            free(del_my);

        }
        printf("\rfolder = %s : L = %d : sigma_h = %lf-%lf... Done.                             \n", folder_name, L_vals[input], h_field_vals[0], h_field_vals[len_h_field_vals-1] );
        fflush(stdout);

        pFile_output_1 = fopen("header_reconstructed.dat", "a");
        
        fprintf(pFile_output_1, "-------column label------" );
        fprintf(pFile_output_1, "\n" );
        fprintf(pFile_output_1, "phi\t" );

        fprintf(pFile_output_1, "mx\t" );
        fprintf(pFile_output_1, "my\t" );
        fprintf(pFile_output_1, "dphi\t" );
        fprintf(pFile_output_1, "dmx\t" );
        fprintf(pFile_output_1, "dmy\t" );
        fprintf(pFile_output_1, "\n" );
        fprintf(pFile_output_1, "-------column label------" );
        fprintf(pFile_output_1, "\n" );

        fclose(pFile_output_1);

        // printf("\nInput Initial Answer to Continue... : ");
        // scanf("%d", &input);
        // if (input-1 != Li)
        // {
        //     return 0;
        // }
        // else
        // {
        //     input -=1;
        // }
        chdir(ptr);
    }
    free(buf);
    // free(ptr);
    return 0;
}