#ifndef LU_KERNEL_H_INCLUDED
#define LU_KERNEL_H_INCLUDED

#include "sparse.h"
#include "etree.h"
#include "lu_config.h"

typedef enum
{
    REAL,
    COMPLEX
} LUMode;

typedef struct lu_kernel
{
    int__t _mat_size; // on host
    int__t * _ap;
    int__t * _ai;
    double * _ax;
    int__t * _amdp;
    int__t * _p;
    int__t * _pinv;

    int__t _et_levels;
    int__t * _et_plev;
    int__t * _et_col_lists;

    int__t * _lp;
    int__t * _li;
    double * _lx;

    int__t * _up;
    int__t * _ui;
    double * _ux;

    int * _issucc;
    int * _statuses;
    int * _errs;
    double * _work_buff;

    double * _b;
    double * _x;
    
} Glu;

typedef struct
{
    int _mode;           // 0 is real, 1 is complex

    int _initflag;       // Initialization flag

    int _factflag;       // Factorization flag

    //int _colamd;         // COLAMD flag
    int _amd;            // AMD flag

    int _scaling;        // Scaling flag

    int _rmvzero;        // MC64 flag

    int__t _num_threads; // Number of threads used

    int__t _ava_threads; // NUmber of available threads

    double _pivtol;    // Tolerance of the pivoting

    int__t _thrlim;    // Task limit to do pipeline parallel

    int__t _mat_size;  // Matrix size

    int__t _nnzs;      // Number of non-zero entries

    int__t * _ap;      // Pointer of columns

    int__t * _ai0;        // Original row indexes
    
    int__t * _ai;         // Permuted row indexes

    //double * _ax0;        // Original values of non-zero entries
    void * _ax0;        // Original values of non-zero entries

    //double * _ax;         // Scaled values of non-zero entries
    void * _ax;         // Scaled values of non-zero entries

    //double * _x;          // Solution vector x
    void * _x;          // Solution vector x

    //double * _rhs;        // Right Hand Side (RHS) of the linear system Ax = b
    void * _rhs;        // Right Hand Side (RHS) of the linear system Ax = b

    //double * _rhs0;       // Original RHS
    void * _rhs0;       // Original RHS

    //double * _axlen;       // Length of the complex entries

    int__t * _p;          // Pivoting permutation vector

    int__t * _pinv;       // Inverse pivoting permutation vector

    int__t * _amdp;       // COLAMD permutation vector

    int__t * _mc64pinv;      // MC64 permutation vector
    
    Etree * _et;          // Elimination tree

    CscMat * _L;          // Lower matrix (REAL mode)

    CscMat * _U;          // Upper matrix (REAL mode)

    CscMatComp * _Lcomp;  // Lower matrix (COMPLEX mode)

    CscMatComp * _Ucomp;  // Upper matrix (COMPLEX mode)

    double * _sr;         // Row scaling factors

    double * _sc;         // Column scaling factors

    Glu * _glu;           // GPU-based refactorization

} LU;


LU * lu_ctor(void);

LU * lu_free(LU * lu);

void lu_set_mode(LU * lu, LUMode mode);

void lu_init(LU * lu, int amd, int rmvzero, int scaling, int num_threads, double pivtol);

void lu_read_coo_real(LU * lu, const char * filename);

void lu_read_coo_complex(LU * lu, const char * filename);

void lu_read_coo(LU * lu, const char * filename);

int lu_fact(LU * lu);

int lu_fact_complex(LU * lu);

int lu_refact(LU * lu, double * nax, double * nrhs);

int lu_refact_complex(LU * lu, double (*nax)[2], double (*nrhs)[2]);

int glu_refact(LU * lu, double * nax, double * nrhs, int__t num_blocks, int__t num_threads_per_block);

void lu_read_ms(LU * lu, int__t size, int__t nnzs, int__t * ms_p, int__t * ms_rows, double * ms_vals, double * b);

void rmv_zero_entry_diag_mc64(LU * lu);

void rmv_zero_entry_diag_mc64_complex(LU * lu);

#endif