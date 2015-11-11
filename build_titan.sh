module swap PrgEnv-pgi PrgEnv-gnu
module load cudatoolkit

# find compilers
CC=`which cc`
NVCC=`which nvcc`

# set gpu architectures to compile for
CUDA_ARCH+="-gencode code=sm_35,arch=compute_35 "

# main tile size
OPTS+="-DBLOCKCOPY_TILE_I=64 "
OPTS+="-DBLOCKCOPY_TILE_J=2 "
OPTS+="-DBLOCKCOPY_TILE_K=8 "

# special tile size for boundary conditions
OPTS+="-DBC_TILE_I=64 "
OPTS+="-DBC_TILE_J=16 "
OPTS+="-DBC_TILE_K=16 "

# host level threshold: number of grid elements
OPTS+="-DHOST_LEVEL_SIZE_THRESHOLD=10000 "

# use naive interpolation in fv2
OPTS+="-DUSE_NAIVE_INTERP "

# max number of solves after warmup
OPTS+="-DMAX_SOLVES=10 "

# unified memory allocation options
OPTS+="-DCUDA_UM_ALLOC "
OPTS+="-DCUDA_UM_ZERO_COPY "
OPTS+="-DCUDA_UM_HOST_ATTACH "

# stencil kernel optimizations
OPTS+="-DCUDA_STENCIL_OPT_TEX "

# override MVAPICH flags for C++
OPTS+="-DMPICH_IGNORE_CXX_SEEK "
OPTS+="-DMPICH_SKIP_MPICXX "

# enable NVTX profiling
#OPTS+="-DUSE_NVTX "

./configure --CC=$CC --NVCC=$NVCC --CFLAGS="-O2 -fopenmp $OPTS" --NVCCFLAGS="-O2 -lineinfo -lnvToolsExt $OPTS" --CUDAARCH="$CUDA_ARCH" --no-fe

make clean -C build
make -j3 -C build
