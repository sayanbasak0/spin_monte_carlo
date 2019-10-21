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

// #define enable_CUDA_CODE
#ifdef enable_CUDA_CODE
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif


FILE *pFile_input, *pFile_output_1, *pFile_output_2;

double CUTOFF = 0.000000000001;
double CUTOFF_PHI = 0.00000001;

int main(int argc, char** argv)
{
    int i, j, k, l, input;
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
    
    for (ci=0; ci<config_vals[input] ; ci++)
    {
        char folder_name[128];
        char *pos_0 = folder_name;
        // pos += sprintf(pos, buf, i);
        pos_0 += sprintf(pos_0, "config_%d", ci );

        printf("\nfolder = %s : ", folder_name );

        int ret = chdir(folder_name);
        if (ret)
        { // same as ret!=0, means an error occurred and errno is set
            printf("error!\n"); 
            return 1;
        }

        Li = input;
        printf("\n L = %d :\n", L_vals[input] );
        
        for (hi=0; hi<len_h_field_vals; hi++)
        {
            printf("\rsigma_h = %lf... ", h_field_vals[hi] );
            fflush(stdout);

            char input_file_1[256];
            char *pos_in_1 = input_file_1;
            pos_in_1 += sprintf(pos_in_1, "o_r_O2_2D_%d_%lf.dat", L_vals[Li], h_field_vals[hi] );

            int loop_no = 0;
            long int nth_loop_array_size[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } ;
            int transient_loops = 0;
            int limit_cycle = 1;

            pFile_input = fopen(input_file_1, "r");
            
            double X = 0.0, X_old = -1.0;

            while(fscanf(pFile_input, "%le", &X)==1)
            {
                if (X_old-X > 0.0)
                {
                    loop_no += 1;
                }
                X_old = X;
                nth_loop_array_size[loop_no] += 1;

                fscanf(pFile_input, "%le", &X);
                fscanf(pFile_input, "%le", &X);
                fscanf(pFile_input, "%le", &X);
                fscanf(pFile_input, "%le", &X);
                
            }
            fclose(pFile_input);

            pFile_input = fopen(input_file_1, "r");
            
            X = 0.0, X_old = -1.0;
            double **phi = (double **)malloc((loop_no+1)*sizeof(double *));
            double **mx = (double **)malloc((loop_no+1)*sizeof(double *));
            double **my = (double **)malloc((loop_no+1)*sizeof(double *));

            for (i=0; i<=loop_no; i++)
            {
                phi[i] = (double *)malloc((nth_loop_array_size[i])*sizeof(double));
                mx[i] = (double *)malloc((nth_loop_array_size[i])*sizeof(double));
                my[i] = (double *)malloc((nth_loop_array_size[i])*sizeof(double));
            }
            // for (i=0; i<=loop_no; i++)
            // {
            //     printf("\nloop_no = %d, nth_loop_array_size[%d] = %ld", loop_no, i, nth_loop_array_size[i]);
            // }
            // printf("\nInput Initial Answer to Continue... : ");
            // scanf("%d", &input);
            // if (input-1 != Li)
            // {
            //     return 0;
            // }

            int i_loop=0;
            long int j_array=0;
            while(fscanf(pFile_input, "%le", &X)==1)
            {
                if (X_old-X > 0.0)
                {
                    // loop_no += 1;
                    i_loop += 1;
                    j_array = 0;
                }
                X_old = X;
                phi[i_loop][j_array] = X;
                // nth_loop_array_size[loop_no] += 1;
                
                fscanf(pFile_input, "%le", &X);
                fscanf(pFile_input, "%le", &X);
                fscanf(pFile_input, "%le", &X);
                mx[i_loop][j_array] = X;
                fscanf(pFile_input, "%le", &X);
                my[i_loop][j_array] = X;
                
                j_array += 1;
            }
            fclose(pFile_input);

            // for (i=0; i<=loop_no; i++)
            // {
            //     // printf("\n");
            //     for (j_array=0; j_array<nth_loop_array_size[i]; j_array++)
            //     {
            //         printf("\rsigma_h = %lf... ", h_field_vals[hi] );
            //         printf("loop = %d - ", i );
            //         printf("phi = %le, ", phi[i][j_array] );
            //         printf("mx = %le, ", mx[i][j_array] );
            //         printf("my = %le ", my[i][j_array] );
            //         fflush(stdout);
            //         // printf("\n");
            //     }
            // }

            for (i=0; i<loop_no; i++)
            {
                if (fabs(mx[i][nth_loop_array_size[i]-1] - mx[loop_no][nth_loop_array_size[loop_no]-1]) < CUTOFF)
                {
                    if (fabs(my[i][nth_loop_array_size[i]-1] - my[loop_no][nth_loop_array_size[loop_no]-1]) < CUTOFF)
                    {
                        transient_loops = i;
                        limit_cycle = loop_no-i;
                        break;
                    }
                }
            }

            double phi_transient = phi[transient_loops][nth_loop_array_size[loop_no]-1];
            double del_m, del_m1, del_m2;
            double del_mx, del_mx1, del_mx2;
            double del_my, del_my1, del_my2;
            long int j_array_tr = nth_loop_array_size[transient_loops] - 1 ;
            long int j_array_lc = nth_loop_array_size[loop_no] - 1 ;

            while (j_array_tr>=0 && j_array_lc>=0)
            {
                // if (fabs(phi[transient_loops][j_array_tr] - phi[loop_no][j_array_lc]) < CUTOFF_PHI)
                if ( phi[transient_loops][j_array_tr] == phi[loop_no][j_array_lc] )
                {
                    if (fabs(mx[transient_loops][j_array_tr] - mx[loop_no][j_array_lc]) < CUTOFF)
                    {
                        if (fabs(my[transient_loops][j_array_tr] - my[loop_no][j_array_lc]) < CUTOFF)
                        {
                            phi_transient = phi[transient_loops][j_array_tr];
                            j_array_tr -= 1;
                            j_array_lc -= 1;
                            // printf("\rsigma_h = %lf...  phi_transient = %.14e, checkpoint-1. ", h_field_vals[hi], phi_transient);
                            fflush(stdout);
                        }
                        else
                        {
                            printf(" phi_transient = %.14e, checkpoint-1\n", phi_transient);
                            break;
                        }
                    }
                    else
                    {
                        printf(" phi_transient = %.14e, checkpoint-2\n", phi_transient);
                        break;
                    }
                }
                else if (phi[transient_loops][j_array_tr] > phi[loop_no][j_array_lc])
                {
                    j_array_tr -= 1;
                    if (phi[transient_loops][j_array_tr] < phi[loop_no][j_array_lc])
                    {
                        del_mx = mx[transient_loops][j_array_tr] - mx[transient_loops][j_array_tr+1];
                        del_my = my[transient_loops][j_array_tr] - my[transient_loops][j_array_tr+1];
                        del_m = sqrt(del_mx*del_mx + del_my*del_my) + 2*CUTOFF ;
                        del_mx1 = mx[transient_loops][j_array_tr] - mx[loop_no][j_array_lc];
                        del_my1 = my[transient_loops][j_array_tr] - my[loop_no][j_array_lc];
                        del_m1 = sqrt(del_mx1*del_mx1 + del_my1*del_my1) ;
                        del_mx2 = mx[transient_loops][j_array_tr+1] - mx[loop_no][j_array_lc];
                        del_my2 = my[transient_loops][j_array_tr+1] - my[loop_no][j_array_lc];
                        del_m2 = sqrt(del_mx2*del_mx2 + del_my2*del_my2) ;
                        if (del_m1 + del_m2 < del_m)
                        {
                            phi_transient = phi[transient_loops][j_array_tr+1];
                            printf("\rsigma_h = %lf...  phi_transient = %.14e, checkpoint-3. ", h_field_vals[hi], phi_transient);
                            fflush(stdout);
                        }
                        else
                        {
                            printf(" phi_transient = %.14e, checkpoint-3\n", phi_transient);
                            break;
                        }
                    }
                }
                else // if (phi[transient_loops][j_array_tr] < phi[loop_no][j_array_lc])
                {
                    j_array_lc -= 1;
                    if (phi[transient_loops][j_array_tr] > phi[loop_no][j_array_lc])
                    {
                        del_mx = mx[loop_no][j_array_lc] - mx[loop_no][j_array_lc+1];
                        del_my = my[loop_no][j_array_lc] - my[loop_no][j_array_lc+1];
                        del_m = sqrt(del_mx*del_mx + del_my*del_my) + 2*CUTOFF ;
                        del_mx1 = mx[transient_loops][j_array_tr] - mx[loop_no][j_array_lc];
                        del_my1 = my[transient_loops][j_array_tr] - my[loop_no][j_array_lc];
                        del_m1 = sqrt(del_mx1*del_mx1 + del_my1*del_my1) ;
                        del_mx2 = mx[transient_loops][j_array_tr] - mx[loop_no][j_array_lc+1];
                        del_my2 = my[transient_loops][j_array_tr] - my[loop_no][j_array_lc+1];
                        del_m2 = sqrt(del_mx2*del_mx2 + del_my2*del_my2) ;
                        if (del_m1 + del_m2 < del_m)
                        {
                            phi_transient = phi[loop_no][j_array_lc+1];
                            printf("\rsigma_h = %lf...  phi_transient = %.14e, checkpoint-4. ", h_field_vals[hi], phi_transient);
                            fflush(stdout);
                        }
                        else
                        {
                            printf(" phi_transient = %.14e, checkpoint-4\n", phi_transient);
                            break;
                        }
                    }
                }
            }
            
        
            // ------------------ Output ------------------ //

            // printf(" phi_transient = %.14e\n", phi_transient);
            // printf(" \n");
            double **delta_phi = (double **)malloc((loop_no+1)*sizeof(double *)) ;
            double **delta_mx = (double **)malloc((loop_no+1)*sizeof(double *)) ;
            double **delta_my = (double **)malloc((loop_no+1)*sizeof(double *)) ;
            
            for (i=0; i<=loop_no; i++)
            {
                delta_phi[i] = (double *)malloc((nth_loop_array_size[i])*sizeof(double));
                delta_mx[i] = (double *)malloc((nth_loop_array_size[i])*sizeof(double));
                delta_my[i] = (double *)malloc((nth_loop_array_size[i])*sizeof(double));
            }

            char output_file_1[256];
            char *pos_out_1 = output_file_1;
            pos_out_1 += sprintf(pos_out_1, "transient_O2_2D_%d_%lf.dat", L_vals[Li], h_field_vals[hi] );

            pFile_output_1 = fopen(output_file_1, "a");

            i_loop = 0;
            j_array = 0;
            while ( i_loop <= transient_loops )
            {
                fprintf(pFile_output_1, "%.14e\t", (double)i_loop + phi[i_loop][j_array] );
                fprintf(pFile_output_1, "%.14e\t", mx[i_loop][j_array] );
                fprintf(pFile_output_1, "%.14e\t", my[i_loop][j_array] );

                if (j_array == nth_loop_array_size[i_loop]-1)
                {
                    delta_phi[i_loop][j_array] = 1.0 + phi[i_loop+1][0] - phi[i_loop][j_array];
                    delta_mx[i_loop][j_array] = mx[i_loop+1][0] - mx[i_loop][j_array];
                    delta_my[i_loop][j_array] = my[i_loop+1][0] - my[i_loop][j_array];

                    fprintf(pFile_output_1, "%.14e\t", delta_phi[i_loop][j_array] );
                    fprintf(pFile_output_1, "%.14e\t", delta_mx[i_loop][j_array] );
                    fprintf(pFile_output_1, "%.14e\t", delta_my[i_loop][j_array] );
                    fprintf(pFile_output_1, "\n" );
                }
                else
                {
                    delta_phi[i_loop][j_array] = phi[i_loop][j_array+1] - phi[i_loop][j_array];
                    delta_mx[i_loop][j_array] = mx[i_loop][j_array+1] - mx[i_loop][j_array];
                    delta_my[i_loop][j_array] = my[i_loop][j_array+1] - my[i_loop][j_array];

                    fprintf(pFile_output_1, "%.14e\t", phi[i_loop][j_array+1] - phi[i_loop][j_array] );
                    fprintf(pFile_output_1, "%.14e\t", mx[i_loop][j_array+1] - mx[i_loop][j_array] );
                    fprintf(pFile_output_1, "%.14e\t", my[i_loop][j_array+1] - my[i_loop][j_array] );
                    fprintf(pFile_output_1, "\n" );
                }

                if (i_loop == transient_loops)
                {
                    if ( phi[i_loop][j_array] >= phi_transient )
                    {
                        break;
                    }
                }

                if (j_array == nth_loop_array_size[i_loop]-1)
                {
                    i_loop += 1;
                    j_array = 0;
                }
                else
                {
                    j_array += 1;
                }
            }

            fclose(pFile_output_1);


            char output_file_2[256];
            char *pos_out_2 = output_file_2;
            pos_out_2 += sprintf(pos_out_2, "limit_cycle_O2_2D_%d_%lf.dat", L_vals[Li], h_field_vals[hi] );

            pFile_output_2 = fopen(output_file_2, "a");

            // if ( i_loop == transient_loops )
            // {
            //     if (j_array == nth_loop_array_size[i_loop]-1)
            //     {
            //         i_loop = transient_loops+1;
            //         j_array = 0;
            //     }
            //     else
            //     {
            //         j_array += 1;
            //     }
            // }
            double phi_limit_cycle = phi[i_loop][j_array];

            
            while ( i_loop <= loop_no )
            {
                fprintf(pFile_output_2, "%.14e\t", (double)i_loop + phi[i_loop][j_array] );
                fprintf(pFile_output_2, "%.14e\t", mx[i_loop][j_array] );
                fprintf(pFile_output_2, "%.14e\t", my[i_loop][j_array] );

                if (j_array == nth_loop_array_size[i_loop]-1)
                {
                    if (i_loop == loop_no)
                    {
                        delta_phi[i_loop][j_array] = 1.0 + phi[transient_loops+1][0] - phi[i_loop][j_array];
                        delta_mx[i_loop][j_array] = mx[transient_loops+1][0] - mx[i_loop][j_array];
                        delta_my[i_loop][j_array] = my[transient_loops+1][0] - my[i_loop][j_array];

                        fprintf(pFile_output_2, "%.14e\t", delta_phi[i_loop][j_array] );
                        fprintf(pFile_output_2, "%.14e\t", delta_mx[i_loop][j_array] );
                        fprintf(pFile_output_2, "%.14e\t", delta_my[i_loop][j_array] );
                        fprintf(pFile_output_2, "\n" );
                    }
                    else
                    {
                        delta_phi[i_loop][j_array] = 1.0 + phi[i_loop+1][0] - phi[i_loop][j_array];
                        delta_mx[i_loop][j_array] = mx[i_loop+1][0] - mx[i_loop][j_array];
                        delta_my[i_loop][j_array] = my[i_loop+1][0] - my[i_loop][j_array];

                        fprintf(pFile_output_2, "%.14e\t", delta_phi[i_loop][j_array] );
                        fprintf(pFile_output_2, "%.14e\t", delta_mx[i_loop][j_array] );
                        fprintf(pFile_output_2, "%.14e\t", delta_my[i_loop][j_array] );
                        fprintf(pFile_output_2, "\n" );
                    }
                }
                else
                {
                    delta_phi[i_loop][j_array] = phi[i_loop][j_array+1] - phi[i_loop][j_array];
                    delta_mx[i_loop][j_array] = mx[i_loop][j_array+1] - mx[i_loop][j_array];
                    delta_my[i_loop][j_array] = my[i_loop][j_array+1] - my[i_loop][j_array];

                    fprintf(pFile_output_2, "%.14e\t", phi[i_loop][j_array+1] - phi[i_loop][j_array] );
                    fprintf(pFile_output_2, "%.14e\t", mx[i_loop][j_array+1] - mx[i_loop][j_array] );
                    fprintf(pFile_output_2, "%.14e\t", my[i_loop][j_array+1] - my[i_loop][j_array] );
                    fprintf(pFile_output_2, "\n" );
                }

                if (i_loop == loop_no)
                {
                    if ( phi[i_loop][j_array] >= phi_limit_cycle )
                    {
                        break;
                    }
                }

                if (j_array == nth_loop_array_size[i_loop]-1)
                {
                    i_loop += 1;
                    j_array = 0;
                }
                else
                {
                    j_array += 1;
                }
            }

            fclose(pFile_output_2);


            for (i=0; i<=loop_no; i++)
            {
                free(phi[i]);
                free(mx[i]);
                free(my[i]);
            }
            free(phi);
            free(mx);
            free(my);

            
            for (i=0; i<=loop_no; i++)
            {
                free(delta_phi[i]);
                free(delta_mx[i]);
                free(delta_my[i]);
            }
            free(delta_phi);
            free(delta_mx);
            free(delta_my);
        }
        printf("\rsigma_h = %lf - %lf. Done. \n", h_field_vals[0], h_field_vals[len_h_field_vals-1] );
        fflush(stdout);

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