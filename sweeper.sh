#!/bin/bash -e

export MANPATH="$MANPATH":/home/scratch.alotfi_gpu_1/specHPC/hpc_sdk/Linux_x86_64/22.7/compilers/man/
export PATH=/home/scratch.alotfi_gpu_1/specHPC/hpc_sdk/Linux_x86_64/22.7/compilers/bin:$PATH

export PGI_PATH=/home/scratch.alotfi_gpu_1/specHPC/hpc_sdk/Linux_x86_64/22.7
export MPI_HOME=${PGI_PATH}/comm_libs/openmpi4/openmpi-4.0.5/
export MPI_ROOT=${MPI_HOME}
export PATH=${MPI_HOME}/bin:$PATH
export MANPATH=${MPI_HOME}/share/man:$MANPATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH

export PATH=${PGI_PATH}/compilers/bin:$PATH
export MANPATH=${PGI_PATH}/compilers/man:$MANPATH
export LD_LIBRARY_PATH=${PGI_PATH}/compilers/lib:$LD_LIBRARY_PATH

set NVHPC_CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/cuda-11.6.46-30759134/
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/cuda-11.5.55-30433912/bin/:$PATH  #was this one
#

# find MPI compiler
CC=`which mpicc`
#CC=`which mpiicc`

# find NVCC compiler
NVCC=`which nvcc`

# set gpu architectures to compile for
CUDA_ARCH+="-gencode arch=compute_60,code=sm_60 "
#CUDA_ARCH+="-gencode arch=compute_70,code=sm_70 "
#CUDA_ARCH+="-gencode arch=compute_80,code=sm_80 "

# main tile size
OPTS+="-DBLOCKCOPY_TILE_I=32 "
OPTS+="-DBLOCKCOPY_TILE_J=4 "
OPTS+="-DBLOCKCOPY_TILE_K=8 "

# special tile size for boundary conditions
OPTS+="-DBOUNDARY_TILE_I=64 "
OPTS+="-DBOUNDARY_TILE_J=16 "
OPTS+="-DBOUNDARY_TILE_K=16 "

# host level threshold: number of grid elements
#OPTS+="-DHOST_LEVEL_SIZE_THRESHOLD=10000 "
OPTS+="-DHOST_LEVEL_SIZE_THRESHOLD=${THRESHOLD} "

# max number of solves after warmup
OPTS+="-DMAX_SOLVES=10 "

# unified memory allocation options
OPTS+="-DCUDA_UM_ALLOC "
OPTS+="-DCUDA_UM_ZERO_COPY "

# MPI buffers allocation policy
OPTS+="-DMPI_ALLOC_ZERO_COPY "
#OPTS+="-DMPI_ALLOC_PINNED "

# stencil optimizations
OPTS+="-DUSE_REG "
OPTS+="-DUSE_TEX "
#OPTS+="-DUSE_SHM "

# GSRB smoother options
#OPTS+="-DGSRB_FP "
#OPTS+="-DGSRB_STRIDE2 "
#OPTS+="-DGSRB_BRANCH "
#OPTS+="-DGSRB_OOP "

# tools
#OPTS+="-DUSE_PROFILE "
OPTS+="-DUSE_NVTX "
#OPTS+="-DUSE_ERROR "

# override MVAPICH flags for C++
OPTS+="-DMPICH_IGNORE_CXX_SEEK "
OPTS+="-DMPICH_SKIP_MPICXX "

rm -rf build

# GSRB smoother (default)
./configure --CC=$CC --NVCC=$NVCC --CFLAGS="-O2 -fopenmp $OPTS" --NVCCFLAGS="-O2 -lineinfo -lnvToolsExt $OPTS" --CUDAARCH="$CUDA_ARCH" --no-fe

# Chebyshev smoother
#./configure --CC=$CC --NVCC=$NVCC --CFLAGS="-O2 -fopenmp $OPTS" --NVCCFLAGS="-O2 -lineinfo -lnvToolsExt $OPTS" --CUDAARCH="$CUDA_ARCH" --fv-smoother="cheby" --no-fe

#make clean -C build
make V=1 -j3 -C build

# alternatively set CUDA_VISIBLE_DEVICES appropriately, see README for details
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

# number of CPU threads executing coarse levels
export OMP_NUM_THREADS=4

# enable threads for MVAPICH
export MV2_ENABLE_AFFINITY=0

# Single GPU
#./build/bin/hpgmg-fv 7 8 

mpirun -np 1 ./build/bin/hpgmg-fv 7 8
