#include <stdio.h>
#include <stdlib.h>

long int BinarySearch(long int L, long int R, long int find, long int in_array[])
{
    // If L and R cross each other 
    if (L > R) 
        return 0; 
  
    // If last element is smaller than find 
    if (find >= in_array[R]) 
        return R; 
  
    // Find the middle point 
    long int mid = (L+R)/2; 
  
    // If middle point is floor. 
    if (in_array[mid] == find) 
        return mid; 
  
    // If find lies between mid-1 and mid 
    if (mid > 0 && in_array[mid-1] <= find && find < in_array[mid]) 
        return mid-1; 
  
    // If find is smaller than mid, floor must be in 
    // left half. 
    if (find < in_array[mid]) 
        return BinarySearch(L, mid-1, find, in_array); 
  
    // If mid-1 is not floor and find is greater than 
    // in_array[mid], 
    return BinarySearch(mid+1, R, find, in_array); 

    /////////////////////////////////////////////////
    // printf("%ld,%ld,%ld\n",L,R,find);
    // if (L<0)
    // {
    //     return 0;
    // }
    // if (R<0)
    // {
    //     return 0;
    // }

    // long int M;
    // if (find >= in_array[R])
    // {
    //     return R;
    // }
    // else if (find <= in_array[L])
    // {
    //     return L;
    // }
    // else
    // {
    //     M = ( L + R ) / 2;
    //     if (find > in_array[M])
    //     {
    //         L = M+1;
    //         M = BinarySearch(L, R, find, in_array);
    //     }
    //     else if (find < in_array[M])
    //     {
    //         R = M;
    //         M = BinarySearch(L, R, find, in_array);
    //     }
        
    //     return M;
    // }
    
    
    return -1;
}

void CopyArray(long int from_X[], long int iBegin, long int iEnd, long int to_X[])
{
    long int k;
    for(k = iBegin; k < iEnd; k++)
        to_X[k] = from_X[k];
}

//  Left source half is A[ iBegin:iMiddle-1].
// Right source half is A[iMiddle:iEnd-1   ].
// Result is            B[ iBegin:iEnd-1   ].
void TopDownMerge(long int A[], long int B[], long int iBegin, long int iMiddle, long int iEnd, long int A_temp[], long int B_temp[])
{
    long int i = iBegin, j = iMiddle;
 
    // While there are elements in the left or right runs...
    long int k;
    for (k = iBegin; k < iEnd; k++) {
        // If left run head exists and is <= existing right run head.
        if (i < iMiddle && (j >= iEnd || A[i] <= A[j])) {
            A_temp[k] = A[i];
            B_temp[k] = B[i];
            i = i + 1;
        } else {
            A_temp[k] = A[j];
            B_temp[k] = B[j];
            j = j + 1;
        }
    }
}

// Sort the given run of array A[] using array B[] as a source.
// iBegin is inclusive; iEnd is exclusive (A[iEnd] is not in the set).
void TopDownSplitMerge(long int A_temp[], long int B_temp[], long int iBegin, long int iEnd, long int A[], long int B[])
{
    if(iEnd - iBegin < 2)                       // if run size == 1
        return;                                 //   consider it sorted
    // split the run longer than 1 item into halves
    long int iMiddle = (iEnd + iBegin) / 2;              // iMiddle = mid point
    // recursively sort both runs from array A[] into B[]
    TopDownSplitMerge(A, B, iBegin,  iMiddle, A_temp, B_temp);  // sort the left  run
    TopDownSplitMerge(A, B, iMiddle,    iEnd, A_temp, B_temp);  // sort the right run
    // merge the resulting runs from array B[] into A[]
    TopDownMerge(A_temp, B_temp, iBegin, iMiddle, iEnd, A, B);
}

// Array A[] has the items to sort; array B[] is a work array.
void TopDownMergeSort(long int A[], long int B[], long int n)
{
    long int *A_temp = (long int *)malloc(n*sizeof(long int));
    long int *B_temp = (long int *)malloc(n*sizeof(long int));
    CopyArray(A, 0, n, A_temp);           // duplicate array A[] into A_temp[]
    CopyArray(B, 0, n, B_temp);           // duplicate array B[] into B_temp[]
    TopDownSplitMerge(A_temp, B_temp, 0, n, A, B);   // sort data from A[] and B[] simultaneously w.r.t. A[]
    // for (;n>0;n--)
    // {
    //     printf("A[%ld]=%ld,B[%ld]=%ld\n", n, A[n-1], n, B[n-1]);
    // }
    free(A_temp);
    free(B_temp);
}



int main()
{
    int T;
    // printf("\ntest 0\n");
    scanf("%d", &T);
    // printf("\ntest 0.1\n");
    long int *N = (long int *)malloc(T*sizeof(long int));
    long int *K = (long int *)malloc(T*sizeof(long int));

    long int *P1 = (long int *)malloc(T*sizeof(long int));
    long int *P2 = (long int *)malloc(T*sizeof(long int));
    long int *H1 = (long int *)malloc(T*sizeof(long int));
    long int *H2 = (long int *)malloc(T*sizeof(long int));
    long int *X1 = (long int *)malloc(T*sizeof(long int));
    long int *X2 = (long int *)malloc(T*sizeof(long int));
    long int *Y1 = (long int *)malloc(T*sizeof(long int));
    long int *Y2 = (long int *)malloc(T*sizeof(long int));
    
    long int *A1 = (long int *)malloc(T*sizeof(long int));
    long int *A2 = (long int *)malloc(T*sizeof(long int));
    long int *A3 = (long int *)malloc(T*sizeof(long int));
    long int *A4 = (long int *)malloc(T*sizeof(long int));
    
    long int *B1 = (long int *)malloc(T*sizeof(long int));
    long int *B2 = (long int *)malloc(T*sizeof(long int));
    long int *B3 = (long int *)malloc(T*sizeof(long int));
    long int *B4 = (long int *)malloc(T*sizeof(long int));
    
    long int *C1 = (long int *)malloc(T*sizeof(long int));
    long int *C2 = (long int *)malloc(T*sizeof(long int));
    long int *C3 = (long int *)malloc(T*sizeof(long int));
    long int *C4 = (long int *)malloc(T*sizeof(long int));
    
    long int *M1 = (long int *)malloc(T*sizeof(long int));
    long int *M2 = (long int *)malloc(T*sizeof(long int));
    long int *M3 = (long int *)malloc(T*sizeof(long int));
    long int *M4 = (long int *)malloc(T*sizeof(long int));
    
    long int *P;
    long int *P_xsorted;
    long int *P_relev;
    long int *H;
    long int *H_xsorted;
    long int *H_relev;
    long int *X;
    long int *X_xsorted;
    long int *Y;
    long int *Y_xsorted;
    
    long int *y = calloc(T, sizeof(long int));
    int j;
    
    for (j=0; j<T; j++)
    {
        // printf("\ntest 1\n");
        
        scanf("%ld", &N[j]);
        scanf("%ld", &K[j]);
        
        scanf("%ld", &P1[j]);
        scanf("%ld", &P2[j]);
        scanf("%ld", &A1[j]);
        scanf("%ld", &B1[j]);
        scanf("%ld", &C1[j]);
        scanf("%ld", &M1[j]);
        
        scanf("%ld", &H1[j]);
        scanf("%ld", &H2[j]);
        scanf("%ld", &A2[j]);
        scanf("%ld", &B2[j]);
        scanf("%ld", &C2[j]);
        scanf("%ld", &M2[j]);
        
        scanf("%ld", &X1[j]);
        scanf("%ld", &X2[j]);
        scanf("%ld", &A3[j]);
        scanf("%ld", &B3[j]);
        scanf("%ld", &C3[j]);
        scanf("%ld", &M3[j]);

        scanf("%ld", &Y1[j]);
        scanf("%ld", &Y2[j]);
        scanf("%ld", &A4[j]);
        scanf("%ld", &B4[j]);
        scanf("%ld", &C4[j]);
        scanf("%ld", &M4[j]);

    }

    for (j=0; j<T; j++)
    {
        P = (long int *)malloc((N[j])*sizeof(long int));
        // P_xsorted = (long int *)malloc((N[j])*sizeof(long int));
        P_relev = (long int *)malloc((N[j])*sizeof(long int));
        H = (long int *)malloc((N[j])*sizeof(long int));
        // H_xsorted = (long int *)malloc((N[j])*sizeof(long int));
        H_relev = (long int *)malloc((N[j])*sizeof(long int));
        X = (long int *)malloc((K[j])*sizeof(long int));
        // X_xsorted = (long int *)malloc((K[j])*sizeof(long int));
        Y = (long int *)malloc((K[j])*sizeof(long int));
        // Y_xsorted = (long int *)malloc((K[j])*sizeof(long int));
        
        // P[0] = 0;
        P[0] = P1[j];
        P[1] = P2[j];

        // H[0] = 0;
        H[0] = H1[j];
        H[1] = H2[j];
        
        // X[0] = 0;
        X[0] = X1[j];
        X[1] = X2[j];
        
        // Y[0] = 0;
        Y[0] = Y1[j];
        Y[1] = Y2[j];
        
        
        long int i = 3;
        // printf("\ntest 2 \n");
        for (i=2; i<N[j]; i++) // while (i <= N[j])
        {
            
            P[i] = (A1[j]*P[i-1] + B1[j]*P[i-2] + C1[j]) % M1[j] + 1;
            // printf("P[%ld]=%ld\n", i+1, P[i]);
            
            H[i] = (A2[j]*H[i-1] + B2[j]*H[i-2] + C2[j]) % M2[j] + 1;
            // printf("H[%ld]=%ld\n", i+1, H[i]);
        }

        // CopyArray(P, 0, N[j], P_xsorted);
        // CopyArray(H, 0, N[j], H_xsorted);
        // TopDownMergeSort(P_xsorted, H_xsorted, N[j]);
        TopDownMergeSort(P, H, N[j]);
        
        long int N_relev = 0;
        
        int *irrelev = calloc(N[j],sizeof(int));

        long int ii = 1, temp = P[0]+H[0]-1;
        for (ii=0; ii<N[j]; ii++)
        {
            if (temp >= P[ii]+H[ii])
            {
                irrelev[ii] = 1;
            }
            else
            {
                temp = P[ii]+H[ii];
            }
        }
        temp = P[N[j]-1]-H[N[j]-1]+1;
        for (ii=N[j]-1; ii>=0; ii--)
        {
            if (ii!=0)
            {
                if (P[ii]==P[ii-1] && H[ii]==H[ii-1])
                {
                    continue;
                }
            }
            if (temp <= P[ii]-H[ii] )
            {
                irrelev[ii] = 1;
            }
            else
            {
                temp = P[ii]-H[ii];
            }
        }
        temp=1;
        for (ii=0; ii<N[j]; ii++)
        {
            if (irrelev[ii]!=1)
            {
                P_relev[N_relev] = P[ii];
                H_relev[N_relev] = H[ii];
                N_relev++;
            }
        }
        free(irrelev);
        /////////////// old /////////////////
        // for (ii=0; ii<N[j]; ii++)
        // {
        //     long int xdiff = 0;
        //     int ydiff = 1;
        //     for (i=0; i<N[j] && ydiff; i++)
        //     {
        //         if (i==ii) continue;
        //         if (P[i] == P[ii] && H[i]==H[ii] && i>ii) continue;
        //         // i+=(ii==i);
        //         // xdiff = (P_xsorted[i]>P_xsorted[ii]) * (P_xsorted[i]-P_xsorted[ii]) + (P_xsorted[ii]>P_xsorted[i]) * (P_xsorted[ii]-P_xsorted[i]) ;
        //         // ydiff = xdiff>(H_xsorted[i]-H_xsorted[ii]) ;
        //         xdiff = (P[i]>P[ii]) * (P[i]-P[ii]) + (P[ii]>P[i]) * (P[ii]-P[i]) ;
        //         ydiff = xdiff>(H[i]-H[ii]) ;
        //     }
        //     if(ydiff)
        //     {
        //         // P_relev[N_relev] = P_xsorted[ii];
        //         // H_relev[N_relev] = H_xsorted[ii];
        //         P_relev[N_relev] = P[ii];
        //         H_relev[N_relev] = H[ii];
        //         N_relev++;
        //         // printf("N_relev=%ld\n",N_relev);
        //     }
        // }
        // for (i=0; i<N_relev; i++)
        // {
        //     printf("P[%ld]=%ld,H[%ld]=%ld\n",i+1,P_relev[i],1+i,H_relev[i]);
        // }
        /////////////////////////////////////


        for (i=2; i<K[j]; i++) // while (i <= N[j])
        {
            
            X[i] = (A3[j]*X[i-1] + B3[j]*X[i-2] + C3[j]) % M3[j] + 1;
            
            Y[i] = (A4[j]*Y[i-1] + B4[j]*Y[i-2] + C4[j]) % M4[j] + 1;
        }
        // for (i=0; i<K[j]; i++) // while (i <= N[j])
        // {
        //     printf("X[%ld]=%ld,", i+1, X[i]);
        //     printf("Y[%ld]=%ld\n", i+1, Y[i]);
        // }
        

        if (N_relev<K[j])
        {
            for (ii=0; ii<K[j]; ii++)
            {
                i = BinarySearch(0, N_relev-1, X[ii], P_relev);
                long int xdiff = 0;
                int ydiff = 0; // int ydiff = 1;
                if (i<N_relev-1)
                {
                    // xdiff = (P_relev[i]>X_xsorted[ii]) * (P_relev[i]-X_xsorted[ii]) + (X_xsorted[ii]>P_relev[i]) * (X_xsorted[ii]-P_relev[i]) ;
                    // ydiff = xdiff>(H_relev[i]-Y_xsorted[ii]) ;
                    xdiff = (P_relev[i]>X[ii]) * (P_relev[i]-X[ii]) + (X[ii]>P_relev[i]) * (X[ii]-P_relev[i]) ;
                    ydiff = xdiff>(H_relev[i]-Y[ii]) ;
                
                    // xdiff = (P_relev[i-1]>X_xsorted[ii]) * (P_relev[i-1]-X_xsorted[ii]) + (X_xsorted[ii]>P_relev[i-1]) * (X_xsorted[ii]-P_relev[i-1]) ;
                    // ydiff &= xdiff>(H_relev[i-1]-Y_xsorted[ii]) ;
                    xdiff = (P_relev[i+1]>X[ii]) * (P_relev[i+1]-X[ii]) + (X[ii]>P_relev[i+1]) * (X[ii]-P_relev[i+1]) ;
                    ydiff &= xdiff>(H_relev[i+1]-Y[ii]) ;
                }
                else
                {
                    i=N_relev-2;
                    // xdiff = (P_relev[i]>X_xsorted[ii]) * (P_relev[i]-X_xsorted[ii]) + (X_xsorted[ii]>P_relev[i]) * (X_xsorted[ii]-P_relev[i]) ;
                    // ydiff = xdiff>(H_relev[i]-Y_xsorted[ii]) ;
                    xdiff = (P_relev[i]>X[ii]) * (P_relev[i]-X[ii]) + (X[ii]>P_relev[i]) * (X[ii]-P_relev[i]) ;
                    ydiff = xdiff>(H_relev[i]-Y[ii]) ;
                
                    // xdiff = (P_relev[i+1]>X_xsorted[ii]) * (P_relev[i+1]-X_xsorted[ii]) + (X_xsorted[ii]>P_relev[i+1]) * (X_xsorted[ii]-P_relev[i+1]) ;
                    // ydiff &= xdiff>(H_relev[i+1]-Y_xsorted[ii]) ;
                    xdiff = (P_relev[i+1]>X[ii]) * (P_relev[i+1]-X[ii]) + (X[ii]>P_relev[i+1]) * (X[ii]-P_relev[i+1]) ;
                    ydiff &= xdiff>(H_relev[i+1]-Y[ii]) ;
                }
                y[j] += !ydiff;
            }

        }
        else
        {
            // CopyArray(X, 0, K[j], X_xsorted);
            // CopyArray(Y, 0, K[j], Y_xsorted);
            // TopDownMergeSort(X_xsorted, Y_xsorted, K[j]);
            TopDownMergeSort(X, Y, K[j]);
            
            i=0;
            for (ii=0; ii<K[j]; ii++)
            {
                // while (P_relev[i] < X_xsorted[ii] && i<N_relev)
                while (P_relev[i] < X[ii] && i<N_relev)
                {
                    i++;
                }
                long int xdiff = 0;
                int ydiff = 0; // int ydiff = 1;
                if (i==0)
                {
                    // xdiff = (P_relev[i]>X_xsorted[ii]) * (P_relev[i]-X_xsorted[ii]) + (X_xsorted[ii]>P_relev[i]) * (X_xsorted[ii]-P_relev[i]) ;
                    // ydiff = xdiff>(H_relev[i]-Y_xsorted[ii]) ;
                    xdiff = (P_relev[i]>X[ii]) * (P_relev[i]-X[ii]) + (X[ii]>P_relev[i]) * (X[ii]-P_relev[i]) ;
                    ydiff = xdiff>(H_relev[i]-Y[ii]) ;
                }
                else if (i==N_relev)
                {
                    // xdiff = (P_relev[i-1]>X_xsorted[ii]) * (P_relev[i-1]-X_xsorted[ii]) + (X_xsorted[ii]>P_relev[i-1]) * (X_xsorted[ii]-P_relev[i-1]) ;
                    // ydiff = xdiff>(H_relev[i-1]-Y_xsorted[ii]) ;
                    xdiff = (P_relev[i-1]>X[ii]) * (P_relev[i-1]-X[ii]) + (X[ii]>P_relev[i-1]) * (X[ii]-P_relev[i-1]) ;
                    ydiff = xdiff>(H_relev[i-1]-Y[ii]) ;
                }
                else
                {
                    // xdiff = (P_relev[i]>X_xsorted[ii]) * (P_relev[i]-X_xsorted[ii]) + (X_xsorted[ii]>P_relev[i]) * (X_xsorted[ii]-P_relev[i]) ;
                    // ydiff = xdiff>(H_relev[i]-Y_xsorted[ii]) ;
                    xdiff = (P_relev[i]>X[ii]) * (P_relev[i]-X[ii]) + (X[ii]>P_relev[i]) * (X[ii]-P_relev[i]) ;
                    ydiff = xdiff>(H_relev[i]-Y[ii]) ;
                
                    // xdiff = (P_relev[i-1]>X_xsorted[ii]) * (P_relev[i-1]-X_xsorted[ii]) + (X_xsorted[ii]>P_relev[i-1]) * (X_xsorted[ii]-P_relev[i-1]) ;
                    // ydiff &= xdiff>(H_relev[i-1]-Y_xsorted[ii]) ;
                    xdiff = (P_relev[i-1]>X[ii]) * (P_relev[i-1]-X[ii]) + (X[ii]>P_relev[i-1]) * (X[ii]-P_relev[i-1]) ;
                    ydiff &= xdiff>(H_relev[i-1]-Y[ii]) ;
                }
                
                // for (; i<=N_relev[j] && ydiff; i++)
                // {
                //     xdiff = (P_relev[i]>X_xsorted[ii]) * (P_relev[i]-X_xsorted[ii]) + (X_xsorted[ii]>P_relev[i]) * (X_xsorted[ii]-P_relev[i]) ;
                //     ydiff = xdiff>(H_relev[i]-Y_xsorted[ii]) ;
                // }

                y[j] += !ydiff; // y[j] += !ydiff;
            }
        }
        

        printf("Case #%d: %ld\n", j+1, y[j]);

        // for (i=0; i<N[j]; i++)
        // {
        //     printf("P[%ld]=%ld,H[%ld]=%ld\n", i+1, P[i], i+1, H[i]);
        // }
        // for (i=0; i<K[j]; i++)
        // {
        //     printf("X[%ld]=%ld,Y[%ld]=%ld\n", i+1, X[i], i+1, Y[i]);
        // }
        // TopDownMergeSort(P, H, N[j]);
        // TopDownMergeSort(X, Y, K[j]);
        // for (i=0; i<N[j]; i++)
        // {
        //     printf("P[%ld]=%ld,H[%ld]=%ld\n", i+1, P[i], i+1, H[i]);
        // }
        // for (i=0; i<K[j]; i++)
        // {
        //     printf("X[%ld]=%ld,Y[%ld]=%ld\n", i+1, X[i], i+1, Y[i]);
        // }
        free(P);
        // free(P_xsorted);
        free(P_relev);
        free(H);
        // free(H_xsorted);
        free(H_relev);
        free(X);
        // free(X_xsorted);
        free(Y);
        // free(Y_xsorted);
    }

    
    free(N);
    free(K);

    free(P1) ;
    free(P2) ;
    free(H1) ;
    free(H2) ;
    free(X1) ;
    free(X2) ;
    free(Y1) ;
    free(Y2) ;
    
    free(A1) ;
    free(A2) ;
    free(A3) ;
    free(A4) ;
    
    free(B1) ;
    free(B2) ;
    free(B3) ;
    free(B4) ;
    
    free(C1) ;
    free(C2) ;
    free(C3) ;
    free(C4) ;
    
    free(M1) ;
    free(M2) ;
    free(M3) ;
    free(M4) ;
    
    free(y);

    return 0;
}