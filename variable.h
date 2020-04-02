
FILE *pFile_1, *pFile_2, *pFile_phase, *pFile_output = NULL, *pFile_chkpt, *pFile_temp, *pFile_ising_spin, *pFile_ising_h;
char output_file_0[256];

    int lattice_size[dim_L] = { 10, 10, 10 }; // lattice_size[dim_L]
    long int no_of_sites;
    long int no_of_black_sites;
    long int no_of_white_sites;
    long int no_of_black_white_sites[2];