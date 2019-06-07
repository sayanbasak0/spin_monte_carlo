#include <stdio.h>

int main()
{
    int T;
    // printf("Enter number of test cases : ");
    scanf("%d", &T);
    for (;T>0; T--)
    {
        int A, B, N;
        
        // printf("\nEnter A, B, N : ");
        scanf("%d %d %d", &A, &B, &N);
        A++;
        
        int C;
        char ans[32];

        do
        {
            C = B + A >> 1;
            
            printf("%d\n", C );
            fflush(stdout);
            // printf("\nIs this the number ? " );
            scanf("%s", ans);

            if (ans[4] == 'B')
            {
                B = C - 1;
            }
            else 
            {
                A = C + 1;
            }
            
        }
        while (ans[4] != 'E');
        
    }
    return 0;

}

// #include <stdio.h>
// #include <string.h>

// int main() {
//   int T; scanf("%d", &T);

//   for (int id = 1; id <= T; ++id) {
//     int A, B, N, done = 0;
//     scanf("%d %d %d", &A, &B, &N);
//     for (++A; !done;) {
//       int mid = A + B >> 1;
//       char result[32];
//       printf("%d\n", mid);
//       fflush(stdout);
//       scanf("%s", result);
//       if (!strcmp(result, "CORRECT")) done = 1;
//       else if (!strcmp(result, "TOO_SMALL")) A = mid + 1;
//       else B = mid - 1;
//     }
//   }
//   return 0;
// }
