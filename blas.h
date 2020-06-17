typedef float CCOMPLEX;
typedef double ZCOMPLEX; 
typedef int CBLAS_INDEX;


/*
 * Enumerated and derived types
 */

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/* 
 * Routines with standard 4 prefixes (s, d, c, z)
 */
void cblas_sswap(const int N, float *X, const int incX, 
                 float *Y, const int incY);
void cblas_scopy(const int N, const float *X, const int incX, 
                 float *Y, const int incY);
void cblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY);

void cblas_dswap(const int N, double *X, const int incX, 
                 double *Y, const int incY);
void cblas_dcopy(const int N, const double *X, const int incX, 
                 double *Y, const int incY);
void cblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY);

void cblas_cswap(const int N, CCOMPLEX *X, const int incX, 
                 CCOMPLEX *Y, const int incY);
void cblas_ccopy(const int N, const CCOMPLEX *X, const int incX, 
                 CCOMPLEX *Y, const int incY);
void cblas_caxpy(const int N, const CCOMPLEX *alpha, const CCOMPLEX *X,
                 const int incX, CCOMPLEX *Y, const int incY);

void cblas_zswap(const int N, ZCOMPLEX *X, const int incX, 
                 ZCOMPLEX *Y, const int incY);
void cblas_zcopy(const int N, const ZCOMPLEX *X, const int incX, 
                 ZCOMPLEX *Y, const int incY);
void cblas_zaxpy(const int N, const ZCOMPLEX *alpha, const ZCOMPLEX *X,
                 const int incX, ZCOMPLEX *Y, const int incY);


/* 
 * Routines with S and D prefix only
 */
void cblas_srotg(float *a, float *b, float *c, float *s);
void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
void cblas_srot(const int N, float *X, const int incX,
                float *Y, const int incY, const float c, const float s);
void cblas_srotm(const int N, float *X, const int incX,
                 float *Y, const int incY, const float *P);

void cblas_drotg(double *a, double *b, double *c, double *s);
void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
void cblas_drot(const int N, double *X, const int incX,
                double *Y, const int incY, const double c, const double  s);
void cblas_drotm(const int N, double *X, const int incX,
                 double *Y, const int incY, const double *P);


/* 
 * Routines with S D C Z CS and ZD prefixes
 */
void cblas_sscal(const int N, const float alpha, float *X, const int incX);
void cblas_dscal(const int N, const double alpha, double *X, const int incX);
void cblas_cscal(const int N, const CCOMPLEX *alpha, CCOMPLEX *X, const int incX);
void cblas_zscal(const int N, const ZCOMPLEX *alpha, ZCOMPLEX *X, const int incX);
void cblas_csscal(const int N, const float alpha, CCOMPLEX *X, const int incX);
void cblas_zdscal(const int N, const double alpha, ZCOMPLEX *X, const int incX);


/* Offset variations of the copy, axpy routines */

void sicopy(const int N, const float *X, const int incX, const
            int offsetX, float *Y, const int incY, const int offsetY)
{
  cblas_scopy (N, X+offsetX, incX, Y+offsetY, incY);
}

void dicopy(const int N, const double *X, const int incX, const
            int offsetX, double *Y, const int incY, const int offsetY)
{
  cblas_dcopy (N, X+offsetX, incX, Y+offsetY, incY);
}


void cicopy(const int N, const CCOMPLEX *X, const int incX, const
            int offsetX, CCOMPLEX *Y, const int incY, const int offsetY)
{
  cblas_ccopy (N, X+(2*offsetX), incX, Y+(2*offsetY), incY);
}

void zicopy(const int N, const ZCOMPLEX *X, const int incX, const
            int offsetX, ZCOMPLEX *Y, const int incY, const int offsetY)
{
  cblas_zcopy (N, X+(2*offsetX), incX, Y+(2*offsetY), incY);
}


void cblas_siaxpy(const int N, const float alpha, 
                  const float *X, const int incX, const int offsetX, 
                  float *Y, const int incY, const int offsetY)
{
  
  cblas_saxpy(N, alpha, X+offsetX, incX, Y+offsetY, incY);
}


void cblas_diaxpy(const int N, const double alpha, 
                  const double *X, const int incX, const int offsetX, 
                  double *Y, const int incY, const int offsetY)
{

  cblas_daxpy(N, alpha, X+offsetX, incX, Y+offsetY, incY);
}


void cblas_ciaxpy(const int N, const CCOMPLEX *alpha, 
                  const CCOMPLEX *X, const int incX, const int offsetX, 
                  CCOMPLEX *Y, const int incY, const int offsetY)
{
  cblas_caxpy(N, alpha, X+(2*offsetX), incX, Y+(2*offsetY), incY);
}


void cblas_ziaxpy(const int N, const ZCOMPLEX *alpha, 
                  const ZCOMPLEX *X, const int incX, const int offsetX, 
                  ZCOMPLEX *Y, const int incY, const int offsetY)
{
  cblas_zaxpy(N, alpha, X+(2*offsetX), incX, Y+(2*offsetY), incY);
}


/*
 * ===========================================================================
 * Prototypes for level 2 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */
float  cblas_sdsdot(const int N, const float alpha, const float *X,
                    const int incX, const float *Y, const int incY);
double cblas_dsdot(const int N, const float *X, const int incX, const float *Y,
                   const int incY);
float  cblas_sdot(const int N, const float  *X, const int incX,
                  const float  *Y, const int incY);
double cblas_ddot(const int N, const double *X, const int incX,
                  const double *Y, const int incY);

/*
 * Functions having prefixes Z and C only
 */
void   cblas_cdotu_sub(const int N, const CCOMPLEX *X, const int incX,
                       const CCOMPLEX *Y, const int incY, CCOMPLEX *dotu);
void   cblas_cdotc_sub(const int N, const CCOMPLEX *X, const int incX,
                       const CCOMPLEX *Y, const int incY, CCOMPLEX *dotc);

void   cblas_zdotu_sub(const int N, const ZCOMPLEX *X, const int incX,
                       const ZCOMPLEX *Y, const int incY, ZCOMPLEX *dotu);
void   cblas_zdotc_sub(const int N, const ZCOMPLEX *X, const int incX,
                       const ZCOMPLEX *Y, const int incY, ZCOMPLEX *dotc);


/*
 * Functions having prefixes S D SC DZ
 */
float  cblas_snrm2(const int N, const float *X, const int incX);
float  cblas_sasum(const int N, const float *X, const int incX);

double cblas_dnrm2(const int N, const double *X, const int incX);
double cblas_dasum(const int N, const double *X, const int incX);

float  cblas_scnrm2(const int N, const CCOMPLEX *X, const int incX);
float  cblas_scasum(const int N, const CCOMPLEX *X, const int incX);

double cblas_dznrm2(const int N, const ZCOMPLEX *X, const int incX);
double cblas_dzasum(const int N, const ZCOMPLEX *X, const int incX);



/*
 * Functions having standard 4 prefixes (S D C Z)
 */
CBLAS_INDEX cblas_isamax(const int N, const float  *X, const int incX);
CBLAS_INDEX cblas_idamax(const int N, const double *X, const int incX);
CBLAS_INDEX cblas_icamax(const int N, const void   *X, const int incX);
CBLAS_INDEX cblas_izamax(const int N, const void   *X, const int incX);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/* 
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY);
void cblas_sgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const float alpha,
                 const float *A, const int lda, const float *X,
                 const int incX, const float beta, float *Y, const int incY);
void cblas_strmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *A, const int lda, 
                 float *X, const int incX);
void cblas_stbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda, 
                 float *X, const int incX);
void cblas_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX);
void cblas_strsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *A, const int lda, float *X,
                 const int incX);
void cblas_stbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const float *A, const int lda,
                 float *X, const int incX);
void cblas_stpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const float *Ap, float *X, const int incX);

void cblas_dgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);
void cblas_dgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const double alpha,
                 const double *A, const int lda, const double *X,
                 const int incX, const double beta, double *Y, const int incY);
void cblas_dtrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *A, const int lda, 
                 double *X, const int incX);
void cblas_dtbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda, 
                 double *X, const int incX);
void cblas_dtpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX);
void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *A, const int lda, double *X,
                 const int incX);
void cblas_dtbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const double *A, const int lda,
                 double *X, const int incX);
void cblas_dtpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const double *Ap, double *X, const int incX);

void cblas_cgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const CCOMPLEX *alpha, const CCOMPLEX *A, const int lda,
                 const CCOMPLEX *X, const int incX, const CCOMPLEX *beta,
                 CCOMPLEX *Y, const int incY);
void cblas_cgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const CCOMPLEX *alpha,
                 const CCOMPLEX *A, const int lda, const CCOMPLEX *X,
                 const int incX, const CCOMPLEX *beta, CCOMPLEX *Y, const int incY);
void cblas_ctrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const CCOMPLEX *A, const int lda, 
                 CCOMPLEX *X, const int incX);
void cblas_ctbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const CCOMPLEX *A, const int lda, 
                 CCOMPLEX *X, const int incX);
void cblas_ctpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const CCOMPLEX *Ap, CCOMPLEX *X, const int incX);
void cblas_ctrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const CCOMPLEX *A, const int lda, CCOMPLEX *X,
                 const int incX);
void cblas_ctbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const CCOMPLEX *A, const int lda,
                 CCOMPLEX *X, const int incX);
void cblas_ctpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const CCOMPLEX *Ap, CCOMPLEX *X, const int incX);

void cblas_zgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const ZCOMPLEX *alpha, const ZCOMPLEX *A, const int lda,
                 const ZCOMPLEX *X, const int incX, const ZCOMPLEX *beta,
                 ZCOMPLEX *Y, const int incY);
void cblas_zgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const int KL, const int KU, const ZCOMPLEX *alpha,
                 const ZCOMPLEX *A, const int lda, const ZCOMPLEX *X,
                 const int incX, const ZCOMPLEX *beta, ZCOMPLEX *Y, const int incY);
void cblas_ztrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const ZCOMPLEX *A, const int lda, 
                 ZCOMPLEX *X, const int incX);
void cblas_ztbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const ZCOMPLEX *A, const int lda, 
                 ZCOMPLEX *X, const int incX);
void cblas_ztpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const ZCOMPLEX *Ap, ZCOMPLEX *X, const int incX);
void cblas_ztrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const ZCOMPLEX *A, const int lda, ZCOMPLEX *X,
                 const int incX);
void cblas_ztbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const int K, const ZCOMPLEX *A, const int lda,
                 ZCOMPLEX *X, const int incX);
void cblas_ztpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const int N, const ZCOMPLEX *Ap, ZCOMPLEX *X, const int incX);


/* 
 * Routines with S and D prefixes only
 */
void cblas_ssymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *A,
                 const int lda, const float *X, const int incX,
                 const float beta, float *Y, const int incY);
void cblas_ssbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const float alpha, const float *A,
                 const int lda, const float *X, const int incX,
                 const float beta, float *Y, const int incY);
void cblas_sspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *Ap,
                 const float *X, const int incX,
                 const float beta, float *Y, const int incY);
void cblas_sger(const enum CBLAS_ORDER order, const int M, const int N,
                const float alpha, const float *X, const int incX,
                const float *Y, const int incY, float *A, const int lda);
void cblas_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float *X,
                const int incX, float *A, const int lda);
void cblas_sspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const float *X,
                const int incX, float *Ap);
void cblas_ssyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *X,
                 const int incX, const float *Y, const int incY, float *A,
                 const int lda);
void cblas_sspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *X,
                 const int incX, const float *Y, const int incY, float *A);

void cblas_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *A,
                 const int lda, const double *X, const int incX,
                 const double beta, double *Y, const int incY);
void cblas_dsbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const double alpha, const double *A,
                 const int lda, const double *X, const int incX,
                 const double beta, double *Y, const int incY);
void cblas_dspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *Ap,
                 const double *X, const int incX,
                 const double beta, double *Y, const int incY);
void cblas_dger(const enum CBLAS_ORDER order, const int M, const int N,
                const double alpha, const double *X, const int incX,
                const double *Y, const int incY, double *A, const int lda);
void cblas_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double *X,
                const int incX, double *A, const int lda);
void cblas_dspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const double *X,
                const int incX, double *Ap);
void cblas_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *X,
                 const int incX, const double *Y, const int incY, double *A,
                 const int lda);
void cblas_dspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const double alpha, const double *X,
                 const int incX, const double *Y, const int incY, double *A);


/* 
 * Routines with C and Z prefixes only
 */
void cblas_chemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const CCOMPLEX *alpha, const CCOMPLEX *A,
                 const int lda, const CCOMPLEX *X, const int incX,
                 const CCOMPLEX *beta, CCOMPLEX *Y, const int incY);
void cblas_chbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const CCOMPLEX *alpha, const CCOMPLEX *A,
                 const int lda, const CCOMPLEX *X, const int incX,
                 const CCOMPLEX *beta, CCOMPLEX *Y, const int incY);
void cblas_chpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const CCOMPLEX *alpha, const CCOMPLEX *Ap,
                 const CCOMPLEX *X, const int incX,
                 const CCOMPLEX *beta, CCOMPLEX *Y, const int incY);
void cblas_cgeru(const enum CBLAS_ORDER order, const int M, const int N,
                 const CCOMPLEX *alpha, const CCOMPLEX *X, const int incX,
                 const CCOMPLEX *Y, const int incY, CCOMPLEX *A, const int lda);
void cblas_cgerc(const enum CBLAS_ORDER order, const int M, const int N,
                 const CCOMPLEX *alpha, const CCOMPLEX *X, const int incX,
                 const CCOMPLEX *Y, const int incY, CCOMPLEX *A, const int lda);
void cblas_cher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const CCOMPLEX *X, const int incX,
                CCOMPLEX *A, const int lda);
void cblas_chpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const float alpha, const CCOMPLEX *X,
                const int incX, CCOMPLEX *A);
void cblas_cher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                 const CCOMPLEX *alpha, const CCOMPLEX *X, const int incX,
                 const CCOMPLEX *Y, const int incY, CCOMPLEX *A, const int lda);
void cblas_chpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                 const CCOMPLEX *alpha, const CCOMPLEX *X, const int incX,
                 const CCOMPLEX *Y, const int incY, CCOMPLEX *Ap);

void cblas_zhemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const ZCOMPLEX *alpha, const ZCOMPLEX *A,
                 const int lda, const ZCOMPLEX *X, const int incX,
                 const ZCOMPLEX *beta, ZCOMPLEX *Y, const int incY);
void cblas_zhbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const int K, const ZCOMPLEX *alpha, const ZCOMPLEX *A,
                 const int lda, const ZCOMPLEX *X, const int incX,
                 const ZCOMPLEX *beta, ZCOMPLEX *Y, const int incY);
void cblas_zhpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const int N, const ZCOMPLEX *alpha, const ZCOMPLEX *Ap,
                 const ZCOMPLEX *X, const int incX,
                 const ZCOMPLEX *beta, ZCOMPLEX *Y, const int incY);
void cblas_zgeru(const enum CBLAS_ORDER order, const int M, const int N,
                 const ZCOMPLEX *alpha, const ZCOMPLEX *X, const int incX,
                 const ZCOMPLEX *Y, const int incY, ZCOMPLEX *A, const int lda);
void cblas_zgerc(const enum CBLAS_ORDER order, const int M, const int N,
                 const ZCOMPLEX *alpha, const ZCOMPLEX *X, const int incX,
                 const ZCOMPLEX *Y, const int incY, ZCOMPLEX *A, const int lda);
void cblas_zher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const ZCOMPLEX *X, const int incX,
                ZCOMPLEX *A, const int lda);
void cblas_zhpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const int N, const double alpha, const ZCOMPLEX *X,
                const int incX, ZCOMPLEX *A);
void cblas_zher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                 const ZCOMPLEX *alpha, const ZCOMPLEX *X, const int incX,
                 const ZCOMPLEX *Y, const int incY, ZCOMPLEX *A, const int lda);
void cblas_zhpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const int N,
                 const ZCOMPLEX *alpha, const ZCOMPLEX *X, const int incX,
                 const ZCOMPLEX *Y, const int incY, ZCOMPLEX *Ap);

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/* 
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);
void cblas_ssymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *B, const int ldb, const float beta,
                 float *C, const int ldc);
void cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const float alpha, const float *A, const int lda,
                 const float beta, float *C, const int ldc);
void cblas_ssyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const float alpha, const float *A, const int lda,
                  const float *B, const int ldb, const float beta,
                  float *C, const int ldc);
void cblas_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 float *B, const int ldb);
void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 float *B, const int ldb);

void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);
void cblas_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *B, const int ldb, const double beta,
                 double *C, const int ldc);
void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const double alpha, const double *A, const int lda,
                 const double beta, double *C, const int ldc);
void cblas_dsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const double alpha, const double *A, const int lda,
                  const double *B, const int ldb, const double beta,
                  double *C, const int ldc);
void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb);
void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 double *B, const int ldb);

void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const CCOMPLEX *alpha, const CCOMPLEX *A,
                 const int lda, const CCOMPLEX *B, const int ldb,
                 const CCOMPLEX *beta, CCOMPLEX *C, const int ldc);
void cblas_csymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const CCOMPLEX *alpha, const CCOMPLEX *A, const int lda,
                 const CCOMPLEX *B, const int ldb, const CCOMPLEX *beta,
                 CCOMPLEX *C, const int ldc);
void cblas_csyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const CCOMPLEX *alpha, const CCOMPLEX *A, const int lda,
                 const CCOMPLEX *beta, CCOMPLEX *C, const int ldc);
void cblas_csyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const CCOMPLEX *alpha, const CCOMPLEX *A, const int lda,
                  const CCOMPLEX *B, const int ldb, const CCOMPLEX *beta,
                  CCOMPLEX *C, const int ldc);
void cblas_ctrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const CCOMPLEX *alpha, const CCOMPLEX *A, const int lda,
                 CCOMPLEX *B, const int ldb);
void cblas_ctrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const CCOMPLEX *alpha, const CCOMPLEX *A, const int lda,
                 CCOMPLEX *B, const int ldb);

void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const ZCOMPLEX *alpha, const ZCOMPLEX *A,
                 const int lda, const ZCOMPLEX *B, const int ldb,
                 const ZCOMPLEX *beta, ZCOMPLEX *C, const int ldc);
void cblas_zsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const ZCOMPLEX *alpha, const ZCOMPLEX *A, const int lda,
                 const ZCOMPLEX *B, const int ldb, const ZCOMPLEX *beta,
                 ZCOMPLEX *C, const int ldc);
void cblas_zsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const ZCOMPLEX *alpha, const ZCOMPLEX *A, const int lda,
                 const ZCOMPLEX *beta, ZCOMPLEX *C, const int ldc);
void cblas_zsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const ZCOMPLEX *alpha, const ZCOMPLEX *A, const int lda,
                  const ZCOMPLEX *B, const int ldb, const ZCOMPLEX *beta,
                  ZCOMPLEX *C, const int ldc);
void cblas_ztrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const ZCOMPLEX *alpha, const ZCOMPLEX *A, const int lda,
                 ZCOMPLEX *B, const int ldb);
void cblas_ztrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const int M, const int N,
                 const ZCOMPLEX *alpha, const ZCOMPLEX *A, const int lda,
                 ZCOMPLEX *B, const int ldb);


/* 
 * Routines with prefixes C and Z only
 */
void cblas_chemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const CCOMPLEX *alpha, const CCOMPLEX *A, const int lda,
                 const CCOMPLEX *B, const int ldb, const CCOMPLEX *beta,
                 CCOMPLEX *C, const int ldc);
void cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const float alpha, const CCOMPLEX *A, const int lda,
                 const float beta, CCOMPLEX *C, const int ldc);
void cblas_cher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const CCOMPLEX *alpha, const CCOMPLEX *A, const int lda,
                  const CCOMPLEX *B, const int ldb, const float beta,
                  CCOMPLEX *C, const int ldc);

void cblas_zhemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const int M, const int N,
                 const ZCOMPLEX *alpha, const ZCOMPLEX *A, const int lda,
                 const ZCOMPLEX *B, const int ldb, const ZCOMPLEX *beta,
                 ZCOMPLEX *C, const int ldc);
void cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                 const double alpha, const ZCOMPLEX *A, const int lda,
                 const double beta, ZCOMPLEX *C, const int ldc);
void cblas_zher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const int N, const int K,
                  const ZCOMPLEX *alpha, const ZCOMPLEX *A, const int lda,
                  const ZCOMPLEX *B, const int ldb, const double beta,
                  ZCOMPLEX *C, const int ldc);


/* Offset variants of ger routines */

void cblas_siger(const enum CBLAS_ORDER order, const int M, const int N,
                 const float alpha, 
                 const float *X, const int incX, const int offsetX,
                 const float *Y, const int incY, const int offsetY,
                 float *A, const int lda)
{
  
  cblas_sger(order, M, N, alpha, X+offsetX, incX, Y+offsetY, incY, A, lda);
}

void cblas_diger(const enum CBLAS_ORDER order, const int M, const int N,
                 const double alpha, 
                 const double *X, const int incX, const int offsetX,
                 const double *Y, const int incY, const int offsetY,
                                      double *A, const int lda)
{
  
  cblas_dger(order, M, N, alpha, X+offsetX, incX, Y+offsetY, incY, A, lda);
}


/* "Standardize" some procedure names */

float  cblas_cnrm2(const int N, const CCOMPLEX *X, const int incX)
{
  return cblas_scnrm2(N,X,incX);
}

double cblas_znrm2(const int N, const ZCOMPLEX *X, const int incX)
{
  return cblas_dznrm2(N,X,incX);
}


float  cblas_casum(const int N, const CCOMPLEX *X, const int incX)
{
  return cblas_scasum(N,X,incX);
}

double  cblas_zasum(const int N, const ZCOMPLEX *X, const int incX)
{
  return cblas_dzasum(N,X,incX);
}

CBLAS_INDEX cblas_samax(const int N, const float  *X, const int incX)
{
  return cblas_isamax(N,X,incX);
}

CBLAS_INDEX cblas_damax(const int N, const double *X, const int incX)
{
  return cblas_idamax(N,X,incX);
}

CBLAS_INDEX cblas_camax(const int N, const void   *X, const int incX)
{
  return cblas_icamax(N,X,incX);
}

CBLAS_INDEX cblas_zamax(const int N, const void   *X, const int incX)
{
  return cblas_izamax(N,X,incX);
}

void   cblas_cdotu(const int N, const CCOMPLEX *X, const int incX,
                   const CCOMPLEX *Y, const int incY, CCOMPLEX *dotu)
{
  cblas_cdotu_sub(N,X,incX,Y,incY,dotu);
}

void   cblas_cdotc(const int N, const CCOMPLEX *X, const int incX,
                   const CCOMPLEX *Y, const int incY, CCOMPLEX *dotc)
{
  cblas_cdotc_sub(N,X,incX,Y,incY,dotc);
}

void   cblas_zdotu(const int N, const ZCOMPLEX *X, const int incX,
                   const ZCOMPLEX *Y, const int incY, ZCOMPLEX *dotu)
{
  cblas_zdotu_sub(N,X,incX,Y,incY,dotu);
}

void   cblas_zdotc(const int N, const ZCOMPLEX *X, const int incX,
                   const ZCOMPLEX *Y, const int incY, ZCOMPLEX *dotc)
{
  cblas_zdotc_sub(N,X,incX,Y,incY,dotc);
}

