 HPGMG: High-performance Geometric Multigrid
===========================================

This is a heterogeneous implementation of HPGMG-FV using CUDA with Unified
Memory. The code is multi-GPU ready and uses one rank per GPU.

HPGMG implements full multigrid (FMG) algorithms using finite-volume and
finite-element methods.  Different algorithmic variants adjust the arithmetic
intensity and architectural properties that are tested. These FMG methods
converge up to discretization error in one F-cycle, thus may be considered
direct solvers.  An F-cycle visits the finest level a total of two times, the
first coarsening (8x smaller) 4 times, the second coarsening 6 times, etc.

#General installation

Use build.sh script as a reference for configure and make.  Note that currently
only the finite-volume solver is enabled on GPU.  NVIDIA Kepler architecture
GPU and CUDA >= 6.0 is required to run this code.  There are ready scripts
available for ORNL Titan cluster: use build_titan.sh to compile,
finite-volume/example_jobs/job.titan to submit a job.  Default is the 4th order
scheme (fv4) using GSRB smoother.  It is possible to compile the 2nd order
(fv2) by updating local.mk and specify a different smoother by using
--fv-smoother config option (see build.sh).

# HPGMG-FV: Finite Volume solver

The finite-volume solver uses cell-centered methods with constant or variable
coefficients.  This implementation requires CUDA >= 6.0 and OpenMP and cannot
be configured at run-time.  Be sure to pass suitable NVCC and OpenMP flags.
See build.sh for recommended GPU settings.  More details about the GPU
implementation and a brief description of various options is available in the
corresponding [finite-volume readme](finite-volume/source/README).

## Running

For multi-GPU configurations it is recommended to run as many MPI ranks as you
have GPUs in your system.  Please note that if peer mappings are not available
between GPUs then the system will fall back to using zero-copy memory which can
perform very slowly.  This issue can be resolved by setting
CUDA_VISIBLE_DEVICES environment variable to constrain which GPUs are visible
for the system, or by setting CUDA_MANAGED_FORCE_DEVICE_ALLOC to a non-zero
value. See [CUDA Programming
Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-managed-memory)
for more details.

Below is a sample application output using NVIDIA Tesla K20:

```
$ export OMP_NUM_THREADS=4
$ ./build/bin/hpgmg-fv 7 8

rank 0:  Number of visible GPUs:  1
rank 0:  GPU0 name Tesla K20c


********************************************************************************
***                            HPGMG-FV Benchmark                            ***
********************************************************************************
Requested MPI_THREAD_FUNNELED, got MPI_THREAD_FUNNELED
1 MPI Tasks of 4 threads


===== Benchmark setup ==========================================================

attempting to create a 256^3 level from 8 x 128^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the GPU
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000044 seconds)
  Calculating boxes per process... target=8.000, max=8
  Creating Poisson (a=0.000000, b=1.000000) test problem
  calculating D^{-1} exactly for level h=3.906250e-03 using 64 colors...  done (3.253787 seconds)
  estimating  lambda_max... <2.223326055334546e+00

attempting to create a 128^3 level from 8 x 64^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the GPU
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000036 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 64^3 level from 8 x 32^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the GPU
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000043 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 32^3 level from 8 x 16^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the GPU
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000013 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 16^3 level from 8 x 8^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the host
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000023 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 8^3 level from 1 x 8^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the host
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000012 seconds)
  Calculating boxes per process... target=1.000, max=1

attempting to create a 4^3 level from 1 x 4^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the host
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000011 seconds)
  Calculating boxes per process... target=1.000, max=1

attempting to create a 2^3 level from 1 x 2^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the host
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000009 seconds)
  Calculating boxes per process... target=1.000, max=1

  Building restriction and interpolation lists... done

  Building MPI subcommunicator for level 1... done (0.000029 seconds)
  Building MPI subcommunicator for level 2... done (0.000011 seconds)
  Building MPI subcommunicator for level 3... done (0.000008 seconds)
  Building MPI subcommunicator for level 4... done (0.000006 seconds)
  Building MPI subcommunicator for level 5... done (0.000007 seconds)
  Building MPI subcommunicator for level 6... done (0.000006 seconds)
  Building MPI subcommunicator for level 7... done (0.000007 seconds)

  calculating D^{-1} exactly for level h=7.812500e-03 using 64 colors...  done (0.477303 seconds)
  estimating  lambda_max... <2.223332976449110e+00
  calculating D^{-1} exactly for level h=1.562500e-02 using 64 colors...  done (0.078353 seconds)
  estimating  lambda_max... <2.223387382550970e+00
  calculating D^{-1} exactly for level h=3.125000e-02 using 64 colors...  done (0.021781 seconds)
  estimating  lambda_max... <2.223793919679896e+00
  calculating D^{-1} exactly for level h=6.250000e-02 using 64 colors...  done (0.004676 seconds)
  estimating  lambda_max... <2.226274210000863e+00
  calculating D^{-1} exactly for level h=1.250000e-01 using 64 colors...  done (0.000957 seconds)
  estimating  lambda_max... <2.230456244760858e+00
  calculating D^{-1} exactly for level h=2.500000e-01 using 64 colors...  done (0.000515 seconds)
  estimating  lambda_max... <2.232895109443501e+00
  calculating D^{-1} exactly for level h=5.000000e-01 using 8 colors...  done (0.000045 seconds)
  estimating  lambda_max... <1.375886524822695e+00



===== Warming up by running 10 solves ==========================================
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.449285 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.449055 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.448706 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.448903 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447468 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447508 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447609 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447216 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447783 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447501 seconds)


===== Running 10 solves ========================================================
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447407 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447450 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447126 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447691 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447396 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447678 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447265 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447090 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447588 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447583 seconds)


===== Timing Breakdown =========================================================


level                                0            1            2            3            4            5            6            7 
level dimension                  256^3        128^3         64^3         32^3         16^3          8^3          4^3          2^3 
box dimension                    128^3         64^3         32^3         16^3          8^3          8^3          4^3          2^3        total
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
smooth                        0.000122     0.000234     0.000347     0.000463     0.001918     0.000389     0.000234     0.000000     0.003707
residual                      0.000021     0.000020     0.000029     0.000039     0.000218     0.000042     0.000021     0.000013     0.000403
applyOp                       0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000031     0.000031
BLAS1                         0.023472     0.000006     0.000012     0.000017     0.000046     0.000030     0.000027     0.000233     0.023844
BLAS3                         0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Boundary Conditions           0.018046     0.012454     0.008626     0.008353     0.000701     0.000288     0.000275     0.000070     0.048814
Restriction                   0.022185     0.006088     0.001300     0.010350     0.000020     0.000016     0.000009     0.000000     0.039967
  local restriction           0.000060     0.000086     0.000110     0.010056     0.000019     0.000015     0.000008     0.000000     0.010354
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Interpolation                 0.008293     0.001090     0.000245     0.000533     0.000216     0.000065     0.000026     0.000000     0.010468
  local interpolation         0.008293     0.001089     0.000245     0.000532     0.000215     0.000065     0.000025     0.000000     0.010463
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Ghost Zone Exchange           0.240134     0.061751     0.013678     0.004098     0.000375     0.000004     0.000004     0.000002     0.320046
  local exchange              0.000160     0.000286     0.000402     0.000538     0.000367     0.000000     0.000000     0.000000     0.001752
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
MPI_collectives               0.000001     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000012     0.000014
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
Total by level                0.317742     0.077481     0.023448     0.023495     0.003442     0.000848     0.000608     0.000344     0.447408

   Total time in MGBuild      3.138746 seconds
   Total time in MGSolve      0.447423 seconds
      number of v-cycles             1
Bottom solver iterations            14




===== Warming up by running 10 solves ==========================================
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.089112 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.082957 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083226 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083064 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083830 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.082918 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083286 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083205 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083255 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083176 seconds)


===== Running 10 solves ========================================================
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083151 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083004 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083312 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083029 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083228 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.082985 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083218 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.082948 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083441 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.082983 seconds)


===== Timing Breakdown =========================================================


level                                0            1            2            3            4            5            6 
level dimension                  128^3         64^3         32^3         16^3          8^3          4^3          2^3 
box dimension                     64^3         32^3         16^3          8^3          8^3          4^3          2^3        total
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
smooth                        0.000116     0.000228     0.000347     0.001521     0.000323     0.000199     0.000000     0.002734
residual                      0.000020     0.000019     0.000029     0.000174     0.000035     0.000018     0.000010     0.000306
applyOp                       0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000026     0.000026
BLAS1                         0.003066     0.000006     0.000011     0.000033     0.000024     0.000022     0.000201     0.003363
BLAS3                         0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Boundary Conditions           0.006434     0.005806     0.006297     0.000563     0.000237     0.000236     0.000060     0.019633
Restriction                   0.002846     0.000925     0.008350     0.000016     0.000013     0.000008     0.000000     0.012159
  local restriction           0.000058     0.000082     0.008123     0.000015     0.000012     0.000007     0.000000     0.008297
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Interpolation                 0.001083     0.000239     0.000450     0.000194     0.000060     0.000023     0.000000     0.002049
  local interpolation         0.001083     0.000239     0.000449     0.000193     0.000059     0.000022     0.000000     0.002046
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Ghost Zone Exchange           0.030326     0.009057     0.003062     0.000298     0.000003     0.000003     0.000002     0.042751
  local exchange              0.000143     0.000265     0.000404     0.000292     0.000000     0.000000     0.000000     0.001105
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
MPI_collectives               0.000001     0.000000     0.000000     0.000000     0.000000     0.000000     0.000011     0.000012
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
Total by level                0.044822     0.015763     0.018245     0.002761     0.000708     0.000520     0.000294     0.083113

   Total time in MGBuild      3.138746 seconds
   Total time in MGSolve      0.083126 seconds
      number of v-cycles             1
Bottom solver iterations            12




===== Warming up by running 10 solves ==========================================
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025581 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025271 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025359 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025547 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025173 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.026175 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.026232 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.026606 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025709 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025811 seconds)


===== Running 10 solves ========================================================
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025718 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025613 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025405 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025592 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025192 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025358 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025819 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025159 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025752 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025551 seconds)


===== Timing Breakdown =========================================================


level                                0            1            2            3            4            5 
level dimension                   64^3         32^3         16^3          8^3          4^3          2^3 
box dimension                     32^3         16^3          8^3          8^3          4^3          2^3        total
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
smooth                        0.000115     0.000230     0.001150     0.000260     0.000164     0.000000     0.001920
residual                      0.000020     0.000020     0.000131     0.000028     0.000015     0.000009     0.000223
applyOp                       0.000000     0.000000     0.000000     0.000000     0.000000     0.000022     0.000022
BLAS1                         0.000597     0.000006     0.000023     0.000018     0.000018     0.000169     0.000831
BLAS3                         0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Boundary Conditions           0.003010     0.004253     0.000433     0.000190     0.000197     0.000056     0.008138
Restriction                   0.000483     0.006209     0.000013     0.000011     0.000007     0.000000     0.006723
  local restriction           0.000057     0.006045     0.000013     0.000010     0.000006     0.000000     0.006131
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Interpolation                 0.000233     0.000378     0.000171     0.000052     0.000021     0.000000     0.000855
  local interpolation         0.000233     0.000378     0.000170     0.000052     0.000020     0.000000     0.000853
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Ghost Zone Exchange           0.004446     0.002037     0.000235     0.000003     0.000003     0.000001     0.006724
  local exchange              0.000136     0.000274     0.000230     0.000000     0.000000     0.000000     0.000640
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
MPI_collectives               0.000001     0.000000     0.000000     0.000000     0.000000     0.000009     0.000010
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ 
Total by level                0.009219     0.012888     0.002130     0.000574     0.000435     0.000252     0.025499

   Total time in MGBuild      3.138746 seconds
   Total time in MGSolve      0.025512 seconds
      number of v-cycles             1
Bottom solver iterations            10




===== Performance Summary ======================================================
  h=3.906250000000000e-03  DOF=1.677721600000000e+07  time=0.447423  DOF/s=3.750e+07  MPI=1  OMP=4
  h=7.812500000000000e-03  DOF=2.097152000000000e+06  time=0.083126  DOF/s=2.523e+07  MPI=1  OMP=4
  h=1.562500000000000e-02  DOF=2.621440000000000e+05  time=0.025512  DOF/s=1.028e+07  MPI=1  OMP=4


===== Richardson error analysis ================================================
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447343 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.083149 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025538 seconds)
  h=3.906250000000000e-03  ||error||=1.486406621094630e-08
  order=3.978


===== Deallocating memory ======================================================
attempting to free the restriction and interpolation lists... done
attempting to free the     2^3 level... done
attempting to free the     4^3 level... done
attempting to free the     8^3 level... done
attempting to free the    16^3 level... done
attempting to free the    32^3 level... done
attempting to free the    64^3 level... done
attempting to free the   128^3 level... done
attempting to free the   256^3 level... done


===== Done =====================================================================
```
