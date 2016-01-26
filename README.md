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
Requested MPI_THREAD_FUNNELED, got MPI_THREAD_FUNNELED
1 MPI Tasks of 4 threads


===== Benchmark setup ===============================================

attempting to create a 256^3 level from 8 x 128^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the GPU
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000043 seconds)
  Calculating boxes per process... target=8.000, max=8
  Creating Poisson (a=0.000000, b=1.000000) test problem
  calculating D^{-1} exactly for level h=3.906250e-03 using 64 colors...  done (3.186553 seconds)
  estimating  lambda_max... <2.223326055334546e+00

attempting to create a 128^3 level from 8 x 64^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the GPU
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000033 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 64^3 level from 8 x 32^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  This level will be run on the GPU
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000048 seconds)
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
  Duplicating MPI_COMM_WORLD... done (0.000021 seconds)
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
  Duplicating MPI_COMM_WORLD... done (0.000008 seconds)
  Calculating boxes per process... target=1.000, max=1

  Building restriction and interpolation lists... done

  Building MPI subcommunicator for level 1... done (0.000029 seconds)
  Building MPI subcommunicator for level 2... done (0.000011 seconds)
  Building MPI subcommunicator for level 3... done (0.000006 seconds)
  Building MPI subcommunicator for level 4... done (0.000007 seconds)
  Building MPI subcommunicator for level 5... done (0.000008 seconds)
  Building MPI subcommunicator for level 6... done (0.000006 seconds)
  Building MPI subcommunicator for level 7... done (0.000007 seconds)

  calculating D^{-1} exactly for level h=7.812500e-03 using 64 colors...  done (0.473554 seconds)
  estimating  lambda_max... <2.223332976449110e+00
  calculating D^{-1} exactly for level h=1.562500e-02 using 64 colors...  done (0.082038 seconds)
  estimating  lambda_max... <2.223387382550970e+00
  calculating D^{-1} exactly for level h=3.125000e-02 using 64 colors...  done (0.021434 seconds)
  estimating  lambda_max... <2.223793919679896e+00
  calculating D^{-1} exactly for level h=6.250000e-02 using 64 colors...  done (0.007268 seconds)
  estimating  lambda_max... <2.226274210000863e+00
  calculating D^{-1} exactly for level h=1.250000e-01 using 64 colors...  done (0.001783 seconds)
  estimating  lambda_max... <2.230456244760858e+00
  calculating D^{-1} exactly for level h=2.500000e-01 using 64 colors...  done (0.000652 seconds)
  estimating  lambda_max... <2.232895109443501e+00
  calculating D^{-1} exactly for level h=5.000000e-01 using 8 colors...  done (0.000046 seconds)
  estimating  lambda_max... <1.375886524822695e+00



===== Warming up by running 10 solves ===============================
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.448556 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.448806 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.448563 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447226 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447042 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447016 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447093 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.446898 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447330 seconds)
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.446914 seconds)


===== Running 1 solves =============================================
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.447450 seconds)


===== Timing Breakdown ==============================================


                                     0            1            2            3            4            5            6            7
box dimension                    128^3         64^3         32^3         16^3          8^3          8^3          4^3          2^3        total
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------
smooth                        0.000122     0.000232     0.000350     0.000463     0.001905     0.000435     0.000230     0.000000     0.003737
residual                      0.000020     0.000019     0.000029     0.000038     0.000217     0.000044     0.000022     0.000012     0.000402
applyOp                       0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000030     0.000030
BLAS1                         0.023467     0.000006     0.000012     0.000017     0.000038     0.000029     0.000027     0.000234     0.023830
BLAS3                         0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Boundary Conditions           0.018082     0.012425     0.008622     0.008376     0.000704     0.000332     0.000270     0.000068     0.048879
Restriction                   0.022162     0.006077     0.001302     0.010322     0.000021     0.000016     0.000009     0.000000     0.039909
  local restriction           0.000059     0.000084     0.000109     0.010027     0.000020     0.000015     0.000008     0.000000     0.010322
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Interpolation                 0.008297     0.001088     0.000245     0.000514     0.000218     0.000067     0.000027     0.000000     0.010457
  local interpolation         0.008297     0.001088     0.000244     0.000513     0.000217     0.000067     0.000025     0.000000     0.010452
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Ghost Zone Exchange           0.240168     0.061740     0.013662     0.004099     0.000383     0.000004     0.000004     0.000002     0.320062
  local exchange              0.000165     0.000278     0.000389     0.000526     0.000376     0.000000     0.000000     0.000000     0.001734
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
MPI_collectives               0.000001     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000012     0.000013
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------
Total by level                0.317788     0.077422     0.023437     0.023477     0.003432     0.000934     0.000600     0.000341     0.447431

   Total time in MGBuild      3.037430 seconds
   Total time in MGSolve      0.447445 seconds
      number of v-cycles             1
Bottom solver iterations            14

            Performance      3.750e+07 DOF/s




===== Performing Richardson error analysis ==========================
FMGSolve... f-cycle     norm=5.144230278419926e-07  rel=5.155086150658313e-07  done (0.446816 seconds)
FMGSolve... f-cycle     norm=7.454872257728340e-06  rel=7.517954010323053e-06  done (0.082831 seconds)
FMGSolve... f-cycle     norm=6.934706239802857e-05  rel=7.171778037444580e-05  done (0.025491 seconds)
  h =  3.906250000000000e-03  ||error|| =  1.486406621094630e-08
  order = 3.978


===== Deallocating memory ===========================================
attempting to destroy the   256^3 level... done
attempting to destroy the   128^3 level... done
attempting to destroy the    64^3 level... done
attempting to destroy the    32^3 level... done
attempting to destroy the    16^3 level... done
attempting to destroy the     8^3 level... done
attempting to destroy the     4^3 level... done
attempting to destroy the     2^3 level... done


===== done ==========================================================
```
