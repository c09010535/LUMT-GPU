# LUMT-GPU
Fast sparse LU factorization using left-looking algorithm and the OpenMP-CUDA parallel.

The GPU-based LU refactorization has been added to the LUMT code. The newer code is named as LUMT-GPU. The 
LU factorization is performed on CPUs, and GPU device only performs refactorizations which occupy the vast 
majority of the simulation time.

You can type 'lu filename threads' to run the code. The input card is the same as that for the LUMT code.
