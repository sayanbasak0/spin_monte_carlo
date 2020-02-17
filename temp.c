#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define dim_S 2
#define dim_L 2
#define BLACK_WHITE 0
#define CUTOFF_SPIN 0.000000000001
#define del_phi_max 0.0001
#define del_phi_min 0.00000001
#define del_m_cutoff 0.0001


long no_of_black_white_sites[2] = { 0, 0 };
double CHKPT_TIME = 0.0f;
double *spin;
double *spin_tmp;
int TYPE_INT = 1;
int TYPE_DOUBLE = 1;
FILE* pFile_chkpt;
double start_time;
long no_of_sites = 1;
long L_size[dim_L] = { 2, 2 };

int backup_chkpt(int start_stop, int type, int length, void* array, char* append_str) {
  if (start_stop == 0) {
    char chkpt_file_1[256] = "";
    strcat(chkpt_file_1, append_str);
    strcat(chkpt_file_1, "_chkpt.dat");
    pFile_chkpt = fopen(chkpt_file_1, "w"); }
  else if (start_stop == 1) {
    if (type == TYPE_INT) { // similar for other types (TYPE_LONG, TYPE_FLOAT, TYPE_DOUBLE)
      int *arr = (int *)array;
      for (int j_arr=0; j_arr<length; j_arr++) {
        fprintf(pFile_chkpt, "%d", arr[j_arr]); } } }
  else if (start_stop == 2) {
    fclose(pFile_chkpt);
    double time_now = omp_get_wtime();
    printf("\n---- Checkpoint after %lf seconds ----\n", time_now - start_time); }
  return 0;
}

int restore_chkpt(int start_stop, int type, int length, void* array, char* append_str) {
  static int restore_point_exist = 1;
  if (start_stop == -1) { // to reset static variable
    restore_point_exist = 1;
    return 0; }
  if (restore_point_exist == 0) { return 0; }
  if (start_stop == 0) {
    char chkpt_file_1[256] = "";
    strcat(chkpt_file_1, append_str);
    strcat(chkpt_file_1, "_chkpt.dat");
    pFile_chkpt = fopen(chkpt_file_1, "r");
    if (pFile_chkpt == NULL) {
      restore_point_exist = 0;
      printf("\n---- Starting from Initial Conditions ----\n");
      return 0; } }
  else if (start_stop == 1) {
    if (type == TYPE_INT) { // similar for other types (TYPE_LONG, TYPE_FLOAT, TYPE_DOUBLE)
      int *arr = (int *)array;
      for (int j_arr=0; j_arr<length; j_arr++) {
        fscanf(pFile_chkpt, "%d", &arr[j_arr]); } } }
  else if (start_stop == 2) {
    fclose(pFile_chkpt);
    printf("\n---- Resuming from Checkpoint ----\n"); } 
  return 1;
}

int main()
{
  start_time = omp_get_wtime();
  printf("Hello World!\n");
  int x=2;
  int phi = 1;
  int restore_ = 1;
  int is_complete = 0;
//...
  if (restore_ == 1) {
    restore_ = !restore_;
    restore_chkpt(0, TYPE_INT, 1, &is_complete, "filename");
    restore_chkpt(1, TYPE_DOUBLE, 1, &phi, "filename");
    // call to restore more variables ...
    restore_chkpt(2, TYPE_INT, 1, &is_complete, "filename");
    if (is_complete == 1) {
      printf("\n---- Already Completed ----\n");
      // free allocated memory and close open files
      return 1;
    }
  }
//...
  if (omp_get_wtime() - start_time > CHKPT_TIME) {
    backup_chkpt(0, TYPE_INT, 1, &is_complete, "filename");
    backup_chkpt(1, TYPE_DOUBLE, 1, &phi, "filename");
    // call to backup more variables ...
    backup_chkpt(2, TYPE_INT, 1, &is_complete, "filename");
    printf( "----- Checkpointed here. -----\n");
    // free allocated memory and close open files
    return 2;
  }
//...

  restore_chkpt(0, TYPE_INT, 1, &x, "filename");
  restore_chkpt(1, TYPE_INT, 1, &x, "filename");
  return restore_chkpt(2, TYPE_INT, 1, &x, "filename");
}

//-------------------------------------------------------------

int index_to_position(long index, long *x_L) {
  long prod_L = 1;
  for(int j_L=0; j_L<dim_L; j_L++) {
    x_L[j_L] = ((long)(index/prod_L)) % L_size[j_L];
    prod_L *= L_size[j_L];
  }
  return 0;
}
int position_to_index(long *x_L, long index) {
  long prod_L = 1;
  for(int j_L=0; j_L<dim_L; j_L++) {
    index += x_L[j_L] * prod_L;
    prod_L *= L_size[j_L];
  }
  return 0;
}

long nearest_neighbor(long index, int j_L, int k_L) {
  long neighbor;
  int jj_L;
  long prod_L = 1;
  for(jj_L=0; jj_L<j_L; jj_L++) {
    prod_L *= L_size[jj_L];
  }
  int x_L = ((long)(index/prod_L)) % L_size[j_L];
  neighbor = index - (x_L * prod_L);
  neighbor += ((x_L + ((k_L*2)-1) + L_size[j_L]) % L_size[j_L]) * prod_L;
  return neighbor;
}


int calculate_new_spin(int i){
    return 0;
}

int add_old_spin_norm(int i, double x){
    return 0;
}

int backup_spin(){
    return 0;
}

int restore_spin(){
    return 0;
}

int calculate_magnetization_change(){
    return 0;
}

int spin_relaxation_v0() {
  int cutoff_check = 0, j_S; long i = 0;
  do {
    cutoff_check = 0;
    for (i=0; i<no_of_sites; i++) {
      calculate_new_spin(i);
      for (j_S=0; j_S<dim_S; j_S++) {
        cutoff_check = cutoff_check || ( fabs(spin[i*dim_S+j_S] - spin_tmp[i*dim_S+j_S]) > CUTOFF_SPIN );
        spin[i*dim_S+j_S] = spin_tmp[i*dim_S+j_S];
      }
    }
  } while ( cutoff_check > 0 );
  return 0;
}

int spin_relaxation_v1() {
  int cutoff_check = 0, j_S; long i = 0;
  do {
    cutoff_check = 0;
    #pragma omp parallel 
    {
      #pragma omp for 
      for (i=0; i<no_of_sites; i++) {
        calculate_new_spin(i);
      }
      #pragma omp for private(j_S) reduction(||:cutoff_check)
      for (i=0; i<no_of_sites; i++) {
        for (j_S=0; j_S<dim_S; j_S++) {
          cutoff_check = cutoff_check || ( fabs(spin[i*dim_S+j_S] - spin_tmp[i*dim_S+j_S]) > CUTOFF_SPIN );
          spin[i*dim_S+j_S] = spin_tmp[i*dim_S+j_S];
        }
      }
    }
  } while ( cutoff_check > 0 );
  return 0;
}

int spin_relaxation_v2() {
  int cutoff_check = 0, j_S; long i = 0;
  do {
    cutoff_check = 0;
    #pragma omp parallel 
    {
      #pragma omp for 
      for (i=0; i<no_of_sites; i++) {
        calculate_new_spin(i);
        add_old_spin_norm(i, 2); 
      }
      #pragma omp for private(j_S) reduction(||:cutoff_check)
      for (i=0; i<no_of_sites; i++) {
        for (j_S=0; j_S<dim_S; j_S++) {
          cutoff_check = cutoff_check || ( fabs(spin[i*dim_S+j_S] - spin_tmp[i*dim_S+j_S]) > CUTOFF_SPIN );
          spin[i*dim_S+j_S] = spin_tmp[i*dim_S+j_S];
        }
      }
    }
  } while ( cutoff_check > 0 );
  return 0;
}

long *black_white_checker[2];
int initialize_checkerboard() {
  long black_white_index[2] = { 0, 0 };
  black_white_checker[0] = (long*)malloc(((long)(no_of_sites/2)+no_of_sites%2) * sizeof(long));
  black_white_checker[1] = (long*)malloc(((long)(no_of_sites/2)) * sizeof(long));
  for (long i=0; i<no_of_sites; i++) {
    long x_L[dim_L];
    index_to_position(i, x_L);
    long pos_index_sum = 0;
    for (int j_L=0; j_L<dim_L; j_L++) {
      pos_index_sum = pos_index_sum + x_L[j_L];
    }
    int bw = pos_index_sum % 2 ;
    black_white_checker[bw][black_white_index[bw]] = i;
    black_white_index[bw]++;
  }
  return 0;
}

int spin_relaxation_v3() {
  int cutoff_check = 0, j_S; long i;
  static int black_or_white = BLACK_WHITE;
  do {
    cutoff_check = 0;
    #pragma omp parallel 
    {
      #pragma omp for private(j_S) reduction(||:cutoff_check)
      for (i=0; i<no_of_black_white_sites[black_or_white]; i++) {
        long index = black_white_checker[black_or_white][i];
        calculate_new_spin(index);
        for (j_S=0; j_S<dim_S; j_S++) {
          cutoff_check = cutoff_check || ( fabs(spin[index*dim_S+j_S] - spin_tmp[index*dim_S+j_S]) > CUTOFF_SPIN );
          spin[index*dim_S+j_S] = spin_tmp[index*dim_S+j_S];
        }
      }
      black_or_white = !black_or_white;
    }
  } while ( cutoff_check > 0 );
  return 0;
}


int dynamic_rate_phi(double *phi, double *delta_phi) {
  double delta_m = calculate_magnetization_change();
  if (delta_m > del_m_cutoff) {
    if (delta_phi[0] > del_phi_min) {
      restore_spin();
      phi[0] -= delta_phi[0];
      delta_phi[0] /= 2;
      return 1;
    }
    else {
      backup_spin();
      return 0;
    }
  }
  else {
    backup_spin();
    if (delta_phi[0] < del_phi_max && delta_m < del_m_cutoff/2) {
      delta_phi[0] *= 2;
      if (delta_phi[0] > del_phi_max) { delta_phi[0] = del_phi_max; }
    }
    return 0;
  }
}


