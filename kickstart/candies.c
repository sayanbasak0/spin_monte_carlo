#include <stdio.h>
#include <stdlib.h>

int main()
{
    int T;
    // printf("\ntest 0\n");
    scanf("%d", &T);
    // printf("\ntest 0.1\n");
    unsigned int *N = (unsigned int *)malloc(T*sizeof(unsigned int)); 
    unsigned int *O = (unsigned int *)malloc(T*sizeof(unsigned int)); 
    long int *D = (long int *)malloc(T*sizeof(long int));
    long int *X1 = (long int *)malloc(T*sizeof(long int));
    long int *X2 = (long int *)malloc(T*sizeof(long int));
    long int *A = (long int *)malloc(T*sizeof(long int));
    long int *B = (long int *)malloc(T*sizeof(long int));
    long int *C = (long int *)malloc(T*sizeof(long int));
    long int *M = (long int *)malloc(T*sizeof(long int));
    long int *L = (long int *)malloc(T*sizeof(long int));
    long int *Xi;
    long int *Si;
    long int *SS;
    long int *SO;
    long int *SSL = (long int *)malloc(T*sizeof(long int));
    long int *SSR = (long int *)malloc(T*sizeof(long int));
    long int *y = calloc(T, sizeof(long int));
    int j;
    
    for (j=0; j<T; j++)
    {
        // printf("\ntest 1\n");
        
        scanf("%u", &N[j]);
        // N[j] = 1;
        // printf("\ntest 1.1\n");
        scanf("%u", &O[j]);
        // O[j] = 1;
        // printf("\ntest 1.2\n");
        scanf("%ld", &D[j]);
        // D[j] = 1;
        // printf("\ntest 1.3\n");

        scanf("%ld", &X1[j]);
        // printf("\ntest 1.4\n");
        scanf("%ld", &X2[j]);
        // printf("\ntest 1.5\n");
        scanf("%ld", &A[j]);
        // printf("\ntest 1.6\n");
        scanf("%ld", &B[j]);
        // printf("\ntest 1.7\n");
        scanf("%ld", &C[j]);
        // printf("\ntest 1.8\n");
        scanf("%ld", &M[j]);
        // printf("\ntest 1.9\n");
        scanf("%ld", &L[j]);
        // printf("\ntest 1.10\n");

    }

    for (j=0; j<T; j++)
    {
        Xi = (long int *)malloc((N[j]+1)*sizeof(long int));
        Si = (long int *)malloc((N[j]+1)*sizeof(long int));
        SS = (long int *)malloc((N[j]+1)*sizeof(long int));
        SO = (long int *)malloc((N[j]+1)*sizeof(long int));
        
        Xi[0] = 0;
        Xi[1] = X1[j];
        Xi[2] = X2[j];

        Si[0] = 0;
        Si[1] = X1[j] + L[j];
        Si[2] = X2[j] + L[j];
        
        SO[0] = 0;
        SO[1] = Si[1]%2;
        SO[2] = Si[1]%2 + Si[2]%2;
        
        SS[0] = 0;
        SS[1] = Si[1];
        SS[2] = Si[1] + Si[2];
        
        SSL[j] = -1;
        SSR[j] = -1;

        long int i = 3;
        // printf("\ntest 2 \n");
        for (i=3; i<=N[j]; i++) // while (i <= N[j])
        {
            if (M[j] != 0)
            {
                Xi[i] = (A[j]*Xi[i-1] + B[j]*Xi[i-2] + C[j]) % M[j];
                // printf("%ld\n", Xi[j][i]);
            }
            else
            {
                Xi[i] = (A[j]*Xi[i-1] + B[j]*Xi[i-2] + C[j]);
                // printf("%ld\n", Xi[j][i]);
            }
            
            Si[i] = Xi[i] + L[j];
            // printf("%ld\n", Si[j][i]);
            SS[i] = SS[i-1] + Si[i];
            // printf("%ld\n", SS[j][i]);
            SO[i] = SO[i-1] + (Si[i]%2);
            // printf("%ld\n", SO[j][i]);
            // printf("%ld %ld %ld %ld \n", Xi[i], Si[i], SO[i], SS[i]);

        }
        // printf("\ntest 3\n");
        for (i=1; i<=N[j]; i++)
        {
            // printf("\ntest 4\n");
            long int ii=1;
            
            while (SO[i] - SO[i-ii] <= O[j])
            {
                if (SS[i] - SS[i-ii] <= D[j])
                {
                    if (SSR[j] == -1)
                    {
                        y[j] = SS[i] - SS[i-ii];
                        SSL[j] = i-ii+1;
                        SSR[j] = i;
                    }
                    else
                    {
                        if (y[j] < SS[i] - SS[i-ii])
                        {
                            y[j] = SS[i] - SS[i-ii];
                            SSL[j] = i-ii+1;
                            SSR[j] = i;
                        }
                        if (y[j] == D[j])
                        {
                            break;
                        }
                    }
                    
                }
                if (ii == i)
                {
                    break;
                }
                ii++;
                // printf("\ntest 5\n");
            }
            if (SSR[j] != -1)
            {
                if(y[j] == D[j])
                {
                    break;
                }
            }
        }
        if (SSR[j] == -1)
        {
            printf("Case #%d: IMPOSSIBLE\n", j+1);
        }
        else
        {
            printf("Case #%d: %ld\n", j+1, y[j]);
        }
        // printf("\ntest 6\n");
        free(Xi) ;
        free(Si) ;
        free(SS) ;
        free(SO) ;
    }

    
    /* for (j=0; j<T; j++)
    {
        if (SSR[j] == -1)
        {
            printf("Case #%d: IMPOSSIBLE\n", j+1);
        }
        else
        {
            printf("Case #%d: %ld\n", j+1, y[j]);
        }
        // free(Xi[j]);
        // free(Si[j]);
        // free(SS[j]);
        // free(SO[j]);
    } */
    free(N) ; 
    free(O) ; 
    free(D) ;
    free(X1) ;
    free(X2) ;
    free(A) ;
    free(B) ;
    free(C) ;
    free(M) ;
    free(L) ;
    // free(Xi) ;
    // free(Si) ;
    // free(SS) ;
    // free(SO) ;
    free(SSL) ;
    free(SSR) ;
    free(y) ;

    return 0;
}