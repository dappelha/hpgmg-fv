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
finite-volume/example_jobs/job.titan to submit a job.

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

Below is a sample application output using NVIDIA GeForce Titan X:

```
$ export OMP_NUM_THREADS=4
$ ./build/bin/hpgmg-fv 7 8

Requested MPI_THREAD_FUNNELED, got MPI_THREAD_FUNNELED
# of devices:  1
1 MPI Tasks of 4 threads


===== Benchmark setup ===============================================

attempting to create a 256^3 level from 8 x 128^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000057 seconds)
  Calculating boxes per process... target=8.000, max=8
  Creating Poisson (a=0.000000, b=1.000000) test problem
  rebuilding operator for level...  h=3.906250e-03  eigenvalue_max<2.000000e+00

attempting to create a 128^3 level from 8 x 64^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000031 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 64^3 level from 8 x 32^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000032 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 32^3 level from 8 x 16^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000022 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 16^3 level from 8 x 8^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000018 seconds)
  Calculating boxes per process... target=8.000, max=8

attempting to create a 8^3 level from 1 x 8^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000013 seconds)
  Calculating boxes per process... target=1.000, max=1

attempting to create a 4^3 level from 1 x 4^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000012 seconds)
  Calculating boxes per process... target=1.000, max=1

attempting to create a 2^3 level from 1 x 2^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000012 seconds)
  Calculating boxes per process... target=1.000, max=1

attempting to create a 1^3 level from 1 x 1^3 boxes distributed among 1 tasks...
  boundary condition = BC_DIRICHLET
  Decomposing level via Z-mort ordering... done
  Allocating vectors... done
  Duplicating MPI_COMM_WORLD... done (0.000012 seconds)
  Calculating boxes per process... target=1.000, max=1

  Building restriction and interpolation lists... done

  Building MPI subcommunicator for level 1... done (0.000048 seconds)
  Building MPI subcommunicator for level 2... done (0.000007 seconds)
  Building MPI subcommunicator for level 3... done (0.000008 seconds)
  Building MPI subcommunicator for level 4... done (0.000007 seconds)
  Building MPI subcommunicator for level 5... done (0.000006 seconds)
  Building MPI subcommunicator for level 6... done (0.000008 seconds)
  Building MPI subcommunicator for level 7... done (0.000006 seconds)
  Building MPI subcommunicator for level 8... done (0.000007 seconds)

  rebuilding operator for level...  h=7.812500e-03  eigenvalue_max<2.000000e+00
  rebuilding operator for level...  h=1.562500e-02  eigenvalue_max<2.000000e+00
  rebuilding operator for level...  h=3.125000e-02  eigenvalue_max<2.000000e+00
  rebuilding operator for level...  h=6.250000e-02  eigenvalue_max<2.000000e+00
  rebuilding operator for level...  h=1.250000e-01  eigenvalue_max<2.000000e+00
  rebuilding operator for level...  h=2.500000e-01  eigenvalue_max<2.000000e+00
  rebuilding operator for level...  h=5.000000e-01  eigenvalue_max<1.286089e+00
  rebuilding operator for level...  h=1.000000e+00  eigenvalue_max<1.000000e+00



===== Warming up by running 10 solves ===============================
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.086176 seconds)
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.083930 seconds)
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.082946 seconds)
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.082315 seconds)
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.081531 seconds)
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.081480 seconds)
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.081571 seconds)
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.082118 seconds)
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.081608 seconds)
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.081482 seconds)


===== Running 1 solves =============================================
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.081494 seconds)


===== Timing Breakdown ==============================================


                                     0            1            2            3            4            5            6            7            8
box dimension                    128^3         64^3         32^3         16^3          8^3          8^3          4^3          2^3          1^3        total
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------
smooth                        0.000046     0.000095     0.000168     0.000228     0.000990     0.000211     0.000142     0.000078     0.000000     0.001958
residual                      0.000011     0.000011     0.000019     0.000023     0.000093     0.000022     0.000017     0.000009     0.000010     0.000216
applyOp                       0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000010     0.000010
BLAS1                         0.005402     0.000005     0.000009     0.000015     0.000123     0.000034     0.000030     0.000025     0.000149     0.005792
BLAS3                         0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Boundary Conditions           0.000098     0.000170     0.000289     0.000397     0.000146     0.000116     0.000126     0.000135     0.000035     0.001511
Restriction                   0.000011     0.000016     0.000023     0.000028     0.000057     0.000018     0.000012     0.000010     0.000000     0.000175
  local restriction           0.000011     0.000016     0.000022     0.000027     0.000056     0.000017     0.000010     0.000009     0.000000     0.000168
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Interpolation                 0.000011     0.000024     0.000031     0.000092     0.000051     0.000025     0.000011     0.000012     0.000000     0.000258
  local interpolation         0.000011     0.000024     0.000031     0.000091     0.000051     0.000024     0.000011     0.000010     0.000000     0.000251
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
Ghost Zone Exchange           0.042454     0.016665     0.005276     0.003941     0.000188     0.000003     0.000004     0.000004     0.000001     0.068535
  local exchange              0.042453     0.016662     0.005271     0.003935     0.000182     0.000000     0.000000     0.000000     0.000000     0.068503
  pack MPI buffers            0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  unpack MPI buffers          0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Isend                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Irecv                   0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
  MPI_Waitall                 0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000
MPI_collectives               0.000001     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000009     0.000011
------------------        ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------
Total by level                0.048799     0.016405     0.005722     0.007581     0.001598     0.000467     0.000382     0.000329     0.000200     0.081483

   Total time in MGBuild      3.028144 seconds
   Total time in MGSolve      0.081493 seconds
      number of v-cycles             1
Bottom solver iterations             9

            Performance      2.059e+08 DOF/s




===== Performing Richardson error analysis ==========================
FMGSolve... f-cycle     norm=3.655730380627276e-04  rel=1.712155260789041e-03  done (0.081630 seconds)
FMGSolve... f-cycle     norm=7.373370047273620e-04  rel=3.454783598249116e-03  done (0.019513 seconds)
FMGSolve... f-cycle     norm=1.461751956422175e-03  rel=6.862671774280970e-03  done (0.007436 seconds)
  h =  3.906250000000000e-03  ||error|| =  1.312645480964873e-07
  order = 1.967


===== Deallocating memory ===========================================
attempting to destroy the   256^3 level... done
attempting to destroy the   128^3 level... done
attempting to destroy the    64^3 level... done
attempting to destroy the    32^3 level... done
attempting to destroy the    16^3 level... done
attempting to destroy the     8^3 level... done
attempting to destroy the     4^3 level... done
attempting to destroy the     2^3 level... done
attempting to destroy the     1^3 level... done


===== done ==========================================================
```
