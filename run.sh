
export MANPATH="$MANPATH":/home/scratch.alotfi_gpu_1/specHPC/hpc_sdk/Linux_x86_64/22.7/compilers/man/
export PATH=/home/scratch.alotfi_gpu_1/specHPC/hpc_sdk/Linux_x86_64/22.7/compilers/bin:$PATH

export PGI_PATH=/home/scratch.alotfi_gpu_1/specHPC/hpc_sdk/Linux_x86_64/22.7
export MPI_HOME=${PGI_PATH}/comm_libs/openmpi4/openmpi-4.0.5/
export MPI_ROOT=${MPI_HOME}
export PATH=${MPI_HOME}/bin:$PATH
export MANPATH=${MPI_HOME}/share/man:$MANPATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH

#export PATH=${PGI_PATH}/compilers/bin:$PATH
export MANPATH=${PGI_PATH}/compilers/man:$MANPATH
export LD_LIBRARY_PATH=${PGI_PATH}/compilers/lib:$LD_LIBRARY_PATH

set NVHPC_CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/cuda-11.6.46-30759134/
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/cuda-11.5.55-30433912/bin/:$PATH  #was this one


# alternatively set CUDA_VISIBLE_DEVICES appropriately, see README for details
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

# number of CPU threads executing coarse levels
export OMP_NUM_THREADS=4

# enable threads for MVAPICH
export MV2_ENABLE_AFFINITY=0

# Single GPU
#./build/bin/hpgmg-fv 7 8

# MPI, one rank per GPU
#mpirun -np 1 ./build/bin/hpgmg-fv 7 8
mpirun -np 1 nsys profile -t nvtx,cuda,mpi -f true --kill none --sampling-period=500000 -o myprof-gpu ./build/bin/hpgmg-fv 4 8


