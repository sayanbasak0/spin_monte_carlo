// using bitbucket
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>


int main()
{
    unsigned int random_seed[28];
    double x, y, z, r;
    int input=1;
    srand(time(NULL));
    random_seed[0] = rand();
    long long int positives = 0;
    long long int negatives = 0;
    long long int zeroes = 0;
    long long int positive_xy_positive_r = 0;
    long long int positive_xy_negative_r = 0;
    long long int positive_xy_zero_r = 0;

    long long int negative_xy_positive_r = 0;
    long long int negative_xy_negative_r = 0;
    long long int negative_xy_zero_r = 0;
    
    long long int zero_xy_positive_r = 0;
    long long int zero_xy_negative_r = 0;
    long long int zero_xy_zero_r = 0;
    int i;
    for (i=1; i<16; i++)
    {
        random_seed[i] = rand_r(&random_seed[i-1]);
    }
    while (input)
    {
        
        for (i=0; i<65536; i++)
        {
            do
            {
                x = (1.0 - 2.0 * (double)rand_r(&random_seed[i%16])/(double)(RAND_MAX));
                y = (-1.0 + 2.0 * (double)rand_r(&random_seed[i%16])/(double)(RAND_MAX));
            }
            while ( x*x+y*y >= 1 || x*x+y*y <= 0.02 );
            r = (1.0 - 2.0 * (double)rand_r(&random_seed[i%16])/(double)(RAND_MAX));
            z = x * y;
            if (z > 0.0)
            {
                positives++;
                if (r > 0.5)
                {
                    positive_xy_positive_r++;
                }
                else
                {
                    if (r < 0.5)
                    {
                        positive_xy_negative_r++;
                    }
                    else
                    {
                        positive_xy_zero_r++;
                    }
                }
            }
            else
            {
                if (z < 0.0)
                {
                    negatives++;
                    if (r > 0.5)
                    {
                        negative_xy_positive_r++;
                    }
                    else
                    {
                        if (r < 0.5)
                        {
                            negative_xy_negative_r++;
                        }
                        else
                        {
                            negative_xy_zero_r++;
                        }
                    }
                }
                else
                {
                    zeroes++;
                    if (r > 0.5)
                    {
                        zero_xy_positive_r++;
                    }
                    else
                    {
                        if (r < 0.5)
                        {
                            zero_xy_negative_r++;
                        }
                        else
                        {
                            zero_xy_zero_r++;
                        }
                    }
                }
            }

        }
        printf("\n");
        printf("\nPositives = %lld; Positive(xy) Positive(r) = %lld; Positive(xy) Negative(r) = %lld, Positive(xy) Zero(r) = %lld", positives, positive_xy_positive_r, positive_xy_negative_r, positive_xy_zero_r);
        printf("\nNegatives = %lld; Negative(xy) Positive(r) = %lld; Negative(xy) Negative(r) = %lld, Negative(xy) Zero(r) = %lld", negatives, negative_xy_positive_r, negative_xy_negative_r, negative_xy_zero_r);
        printf("\nZeroes = %lld; Zero(xy) Positive(r) = %lld; Zero(xy) Negative(r) = %lld, Zero(xy) Zero(r) = %lld", zeroes, zero_xy_positive_r, zero_xy_negative_r, zero_xy_zero_r);
        printf("\nContinue..?  ");
        // scanf("%d", &input);
        sleep(0.01);
    }
    return 0;
}