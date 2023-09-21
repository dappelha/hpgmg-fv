
#set NVHPC_CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/internal/cuda-12.1.15/
set NVHPC_CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/2023/cuda/12.2/
# hpgmg config uses location of nvcc to assume relative path of ../include (for cuda_runtime.h)
# So put the better location of nvcc first in path if using nvhpc containers:
export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/2023/cuda/12.2/bin/:$PATH




export PGI_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/23.7
export MPI_HOME=${PGI_PATH}/comm_libs/mpi/  # think this is generic, so will be whatever mpi container has loaded (hpcx probably).
#export MPI_HOME=${PGI_PATH}/comm_libs/openmpi4/openmpi-4.0.5/

export MPI_ROOT=${MPI_HOME}
export PATH=${MPI_HOME}/bin:$PATH
export MANPATH=${MPI_HOME}/share/man:$MANPATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH

#export PATH=${PGI_PATH}/compilers/bin:$PATH
export MANPATH=${PGI_PATH}/compilers/man:$MANPATH
export LD_LIBRARY_PATH=${PGI_PATH}/compilers/lib:$LD_LIBRARY_PATH


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
#mpirun -np 1 nsys profile -t nvtx,cuda,mpi -f true --kill none --sampling-period=500000 -o altera-h100 ./build/bin/hpgmg-fv 4 8
mpirun -np 1 nsys profile -t nvtx,cuda,mpi --cuda-memory-usage true -f true --kill none --sampling-period=500000 -o altera-h100-1 ./build/bin/hpgmg-fv 7 8


